#!/usr/bin/env python3
"""Losslessly expand a short ACE-Step 1.5 vocal demo into a full single (+ optional B-track BGM).

Why not `task_type="repaint"`?
In ACE-Step 1.5, `repaint` disables the 5Hz LM (`thinking=False`), causing long vocal continuations
to either repeat Verse 1 or exhaust new lyrics in 25s. Instead, this script:
  1. `lm-continue`: Feeds prefix 5Hz `audio_codes` into `cond` & `uncond` prompts of `acestep-5Hz-lm-1.7B`
     (`FSMState.CODES_GENERATION`) to plan the full remaining song structure across candidate seeds.
  2. `dit-stitch`: Renders in a separate DiT process with `reference_audio` locked to the base demo,
     phase-aligns (`±40ms`) with an equal-power cosine crossfade, and splices `0..crossfade_start_sec`
     bit-for-bit (`max_diff = 0.0`) from the base demo.
  3. `master`: Applies smooth tail fade-out, streaming loudness normalization (`-12.8 LUFS / -1.0 dBTP`),
     `48kHz / 24-bit WAV` + `320kbps MP3`, and silence-gate verification.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from _studio_core import (
    check_silence_gates,
    cleanup_cwd_cache,
    configure_offline_env,
    ensure_acestep_importable,
    load_wav_audio,
    stitch_equal_power_crossfade,
)


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_spec_asset(raw_path: str, spec_file: Path) -> Path:
    p = Path(raw_path).expanduser()
    if p.exists():
        return p.resolve()
    rel = spec_file.resolve().parent / p
    if rel.exists():
        return rel.resolve()
    return p


def rms_tail_profile(audio: np.ndarray, sr: int, start_sec: float) -> list[dict]:
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    win = max(1, sr // 2)
    n_bins = len(mono) // win
    start_bin = max(0, int(start_sec * 2))
    out = []
    for i in range(start_bin, n_bins):
        seg = mono[i * win : (i + 1) * win]
        rms = float(np.sqrt(np.mean(seg**2)))
        db = 20.0 * math.log10(max(rms, 1e-7))
        out.append(
            {
                "t0": round(i * 0.5, 1),
                "t1": round((i + 1) * 0.5, 1),
                "rms": round(rms, 5),
                "dbfs": round(db, 2),
            }
        )
    return out


def cmd_lm_continue(
    spec: dict,
    spec_file: Path,
    out_root: Path,
    ace_root: Path,
    lm_model: str,
    device: str,
    overwrite: bool,
) -> None:
    from acestep.constrained_logits_processor import FSMState
    from acestep.llm_inference import LLMHandler

    base_lm_path = resolve_spec_asset(spec["base_lm_json"], spec_file)
    if not base_lm_path.exists():
        raise FileNotFoundError(f"Base demo lm.json not found: {base_lm_path}")

    orig_lm = json.loads(base_lm_path.read_text(encoding="utf-8"))
    orig_codes = re.findall(r"<\|audio_code_\d+\|>", orig_lm["result"]["audio_codes"] or "")
    prefix_count = int(spec["prefix_codes_count"])
    if prefix_count <= 0 or prefix_count > len(orig_codes):
        raise ValueError(f"prefix_codes_count={prefix_count} out of range for base lm.json ({len(orig_codes)} codes)")

    total_duration = int(spec["target_duration"])
    total_codes = total_duration * 5
    if total_codes <= prefix_count:
        raise ValueError(f"target_duration ({total_duration}s = {total_codes} codes) must exceed prefix ({prefix_count})")
    remain_codes = total_codes - prefix_count
    remain_duration = remain_codes / 5.0
    prefix_str = "".join(orig_codes[:prefix_count])

    full_caption = spec["full_caption"]
    full_lyrics = spec["full_lyrics"]
    cot_text = (
        "<think>\n"
        f"bpm: {spec['bpm']}\n"
        f"caption: {full_caption}\n"
        f"duration: {total_duration}\n"
        f"keyscale: {spec['keyscale']}\n"
        f"language: {spec['vocal_language']}\n"
        f"timesignature: {spec['timesignature']}\n"
        "</think>"
    )

    lm = LLMHandler()
    status, ok = lm.initialize(
        checkpoint_dir=str(ace_root / "checkpoints"),
        lm_model_path=lm_model,
        backend="pt",
        device=device,
    )
    if not ok:
        raise RuntimeError(f"LM init failed: {status}")

    lm_dir = out_root / "lm_continued"
    lm_dir.mkdir(parents=True, exist_ok=True)

    cond_prompt = (
        lm.build_formatted_prompt_with_cot(full_caption, full_lyrics, cot_text, is_negative_prompt=False)
        + prefix_str
    )
    uncond_prompt = (
        lm.build_formatted_prompt_with_cot(
            full_caption,
            full_lyrics,
            cot_text,
            is_negative_prompt=True,
            negative_prompt="NO USER INPUT",
        )
        + prefix_str
    )

    orig_pad_side = lm.llm_tokenizer.padding_side
    lm.llm_tokenizer.padding_side = "left"
    batch_tok = lm.llm_tokenizer(
        [cond_prompt, uncond_prompt],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    lm.llm_tokenizer.padding_side = orig_pad_side
    batch_tok = {k: v.to(lm.device) for k, v in batch_tok.items()}
    input_len = batch_tok["input_ids"].shape[1]

    for seed in spec["candidate_seeds"]:
        seed = int(seed)
        out_json = lm_dir / f"codes_seed_{seed}.json"
        if out_json.exists() and not overwrite:
            print(f"SKIP_EXISTING {out_json}", flush=True)
            continue

        set_all_seeds(seed)
        proc = lm._setup_constrained_processor(
            use_constrained_decoding=True,
            constrained_decoding_debug=False,
            target_duration=remain_duration,
            user_metadata=None,
            stop_at_reasoning=False,
            skip_genres=True,
            skip_caption=True,
            skip_language=True,
            generation_phase="codes",
            is_batch=False,
        )
        proc.state = FSMState.CODES_GENERATION
        proc.codes_count = 0

        t0 = time.perf_counter()
        with lm._load_model_context():
            outputs = lm._generate_with_cfg_custom(
                batch_input_ids=batch_tok["input_ids"],
                batch_attention_mask=batch_tok.get("attention_mask"),
                max_new_tokens=remain_codes + 8,
                temperature=float(spec.get("temperature", 0.82)),
                cfg_scale=float(spec.get("cfg_scale", 2.0)),
                top_k=None,
                top_p=0.9,
                repetition_penalty=1.02,
                pad_token_id=lm.llm_tokenizer.pad_token_id or lm.llm_tokenizer.eos_token_id,
                streamer=None,
                constrained_processor=proc,
            )
        elapsed = time.perf_counter() - t0
        gen_ids = outputs[0, input_len:].cpu()
        tail_text = lm.llm_tokenizer.decode(gen_ids, skip_special_tokens=False)
        tail_codes = re.findall(r"<\|audio_code_\d+\|>", tail_text)
        merged_codes = orig_codes[:prefix_count] + tail_codes
        payload = {
            "seed": seed,
            "prefix_codes_count": prefix_count,
            "tail_codes_count": len(tail_codes),
            "total_codes_count": len(merged_codes),
            "target_duration": total_duration,
            "generation_seconds": round(elapsed, 3),
            "full_caption": full_caption,
            "full_lyrics": full_lyrics,
            "audio_codes": "".join(merged_codes),
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"LM_CONTINUE_OK seed={seed} prefix={prefix_count} tail={len(tail_codes)} "
            f"total={len(merged_codes)} elapsed={elapsed:.2f}s -> {out_json}",
            flush=True,
        )

    bgm = spec.get("bgm_b_track")
    if bgm:
        bgm_dir = out_root / "bgm_b_track"
        bgm_dir.mkdir(parents=True, exist_ok=True)
        bgm_lm_path = bgm_dir / "lm.json"
        if not bgm_lm_path.exists() or overwrite:
            set_all_seeds(int(bgm["seed"]))
            t0 = time.perf_counter()
            res = lm.generate_with_stop_condition(
                caption=bgm["caption"],
                lyrics=bgm.get("lyrics", "[Instrumental]"),
                infer_type="llm_dit",
                target_duration=int(bgm["duration"]),
                user_metadata={
                    "bpm": bgm["bpm"],
                    "duration": bgm["duration"],
                    "keyscale": bgm["keyscale"],
                    "timesignature": str(bgm.get("timesignature", "4")),
                    "language": bgm.get("vocal_language", "en"),
                },
                use_cot_caption=False,
                use_cot_language=False,
            )
            elapsed = time.perf_counter() - t0
            if not res.get("success"):
                raise RuntimeError(f"BGM B-track LM failed: {res.get('error')}")
            b_codes = re.findall(r"<\|audio_code_\d+\|>", res.get("audio_codes") or "")
            expected_b = int(bgm["duration"]) * 5
            if len(b_codes) != expected_b:
                raise ValueError(f"BGM expected {expected_b} codes, got {len(b_codes)}")
            bgm_lm_path.write_text(
                json.dumps(
                    {
                        "request": bgm,
                        "model": lm_model,
                        "code_count": len(b_codes),
                        "generation_seconds": round(elapsed, 3),
                        "result": res,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"LM_BGM_OK {bgm['id']}: {len(b_codes)} codes in {elapsed:.2f}s -> {bgm_lm_path}", flush=True)


def cmd_dit_stitch(
    spec: dict,
    spec_file: Path,
    out_root: Path,
    ace_root: Path,
    dit_model: str,
    device: str,
    overwrite: bool,
) -> None:
    import soundfile as sf
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    base_wav = resolve_spec_asset(spec["base_wav"], spec_file)
    if not base_wav.exists():
        raise FileNotFoundError(f"Base demo WAV not found: {base_wav}")

    orig_audio, orig_sr = sf.read(base_wav, dtype="float32", always_2d=True)
    if orig_sr != 48000:
        raise ValueError(f"Expected 48000 Hz base WAV, got {orig_sr} Hz: {base_wav}")

    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(ace_root),
        config_path=dit_model,
        device=device,
    )
    if not ok:
        raise RuntimeError(f"DiT init failed: {status}")

    lm_dir = out_root / "lm_continued"
    render_root = out_root / "rendered_candidates"
    render_root.mkdir(parents=True, exist_ok=True)

    xf_start_sec = float(spec["crossfade_start_sec"])
    xf_end_sec = float(spec["crossfade_end_sec"])
    rseed = int(spec["render_seed"])
    infer_steps = int(spec.get("inference_steps", 8))

    for seed in spec["candidate_seeds"]:
        seed = int(seed)
        code_file = lm_dir / f"codes_seed_{seed}.json"
        if not code_file.exists():
            raise FileNotFoundError(f"Missing LM continuation codes: {code_file} (run `lm-continue` first)")
        cdata = json.loads(code_file.read_text(encoding="utf-8"))
        cand_dir = render_root / f"seed_{seed}"
        stitched_wav = cand_dir / f"stitched_seed_{seed}.wav"
        if stitched_wav.exists() and not overwrite:
            print(f"SKIP_EXISTING {stitched_wav}", flush=True)
            continue

        cand_dir.mkdir(parents=True, exist_ok=True)
        set_all_seeds(rseed)
        params = GenerationParams(
            task_type="text2music",
            reference_audio=str(base_wav),
            audio_codes=cdata["audio_codes"],
            caption=cdata["full_caption"],
            lyrics=cdata["full_lyrics"],
            bpm=spec["bpm"],
            keyscale=spec["keyscale"],
            timesignature=str(spec["timesignature"]),
            vocal_language=spec["vocal_language"],
            duration=int(cdata["target_duration"]),
            inference_steps=infer_steps,
            seed=rseed,
            instrumental=False,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            use_cot_lyrics=False,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[rseed],
            audio_format="flac",
        )
        t0 = time.perf_counter()
        res = generate_music(dit, LLMHandler(), params, config, save_dir=str(cand_dir / "raw_coded"))
        render_sec = time.perf_counter() - t0
        if not res.success or not res.audios:
            raise RuntimeError(f"DiT render failed for seed {seed}: {res.error}")

        coded_audio, sr = sf.read(res.audios[0]["path"], dtype="float32", always_2d=True)
        coded_wav = cand_dir / f"coded_full_seed_{seed}.wav"
        sf.write(coded_wav, coded_audio, sr, subtype="PCM_24")

        stitched, metrics = stitch_equal_power_crossfade(
            orig_audio=orig_audio,
            coded_audio=coded_audio,
            sr=sr,
            xf_start_sec=xf_start_sec,
            xf_end_sec=xf_end_sec,
        )
        sf.write(stitched_wav, stitched, sr, subtype="PCM_24")

        tail_start = max(0.0, metrics["duration_sec"] - 18.0)
        summary = {
            "seed": seed,
            "coded_render_sec": round(render_sec, 3),
            **metrics,
            "stitched_wav": str(stitched_wav),
            "tail_profile": rms_tail_profile(stitched, sr, start_sec=tail_start),
        }
        (cand_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"DIT_STITCH_OK seed={seed} dur={summary['duration_sec']}s gain={metrics['rms_gain_applied']:.3f} "
            f"lag={metrics['best_lag_ms']}ms prefix_diff={metrics['prefix_bit_exact_max_diff']:.6f} -> {stitched_wav}",
            flush=True,
        )

    bgm = spec.get("bgm_b_track")
    if bgm:
        bgm_dir = out_root / "bgm_b_track"
        bgm_raw_wav = bgm_dir / "raw_bgm.wav"
        if not bgm_raw_wav.exists() or overwrite:
            bgm_lm_file = bgm_dir / "lm.json"
            if not bgm_lm_file.exists():
                raise FileNotFoundError(f"Missing BGM LM codes: {bgm_lm_file}")
            bgm_saved = json.loads(bgm_lm_file.read_text(encoding="utf-8"))
            bseed = int(bgm["seed"])
            set_all_seeds(bseed)
            b_params = GenerationParams(
                task_type="text2music",
                audio_codes=bgm_saved["result"]["audio_codes"],
                caption=bgm["caption"],
                lyrics="[Instrumental]",
                bpm=bgm["bpm"],
                keyscale=bgm["keyscale"],
                timesignature=str(bgm.get("timesignature", "4")),
                vocal_language=bgm.get("vocal_language", "en"),
                duration=int(bgm["duration"]),
                inference_steps=infer_steps,
                seed=bseed,
                instrumental=True,
                thinking=False,
                use_cot_metas=False,
                use_cot_caption=False,
                use_cot_language=False,
                use_cot_lyrics=False,
            )
            b_cfg = GenerationConfig(
                batch_size=1,
                use_random_seed=False,
                seeds=[bseed],
                audio_format="flac",
            )
            t0 = time.perf_counter()
            b_res = generate_music(dit, LLMHandler(), b_params, b_cfg, save_dir=str(bgm_dir / "render"))
            elapsed = time.perf_counter() - t0
            if not b_res.success or not b_res.audios:
                raise RuntimeError(f"BGM B-track DiT failed: {b_res.error}")
            ba, bsr = sf.read(b_res.audios[0]["path"], dtype="float32", always_2d=True)
            sf.write(bgm_raw_wav, ba, bsr, subtype="PCM_24")

            deliv = bgm_dir / "deliverables"
            deliv.mkdir(parents=True, exist_ok=True)
            stem = re.sub(r"[^\w\-]+", "_", bgm.get("title", bgm["id"])).strip("_") or "B_Track_BGM"
            dur_trim = max(4.0, float(bgm["duration"]) - 0.5)
            fade_st = max(1.0, dur_trim - 3.0)
            bed_wav = deliv / f"{stem}_{int(round(dur_trim))}s.wav"
            bed_mp3 = deliv / f"{stem}_{int(round(dur_trim))}s.mp3"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(bgm_raw_wav),
                    "-af", f"atrim=0:{dur_trim:.2f},afade=t=in:st=0:d=0.08,afade=t=out:st={fade_st:.2f}:d=3.0:curve=qsin,loudnorm=I=-16.0:TP=-1.5:LRA=9:linear=true",
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le",
                    str(bed_wav),
                ],
                check=True,
            )
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(bed_wav), "-codec:a", "libmp3lame", "-b:a", "320k", str(bed_mp3)],
                check=True,
            )
            stinger_sec = float(bgm.get("stinger_seconds", 6.2))
            if stinger_sec > 1.5:
                stinger_fade = max(0.8, stinger_sec - 1.4)
                stinger_wav = deliv / f"{stem}_Outro_{int(round(stinger_sec))}s.wav"
                stinger_mp3 = deliv / f"{stem}_Outro_{int(round(stinger_sec))}s.mp3"
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(bgm_raw_wav),
                        "-af", f"atrim=0:{stinger_sec:.2f},afade=t=in:st=0:d=0.06,afade=t=out:st={stinger_fade:.2f}:d=1.4:curve=qsin,loudnorm=I=-14.0:TP=-1.0:LRA=7:linear=true",
                        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le",
                        str(stinger_wav),
                    ],
                    check=True,
                )
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(stinger_wav), "-codec:a", "libmp3lame", "-b:a", "320k", str(stinger_mp3)],
                    check=True,
                )
            print(f"DIT_BGM_OK {bgm['id']} rendered in {elapsed:.2f}s -> {deliv}", flush=True)


def master_deliverable(
    src_wav: Path,
    dst_wav: Path,
    dst_mp3: Path,
    fade_out_start: float,
    total_end: float,
    target_lufs: float = -12.8,
) -> dict:
    if total_end <= fade_out_start:
        raise ValueError(f"total_end ({total_end}) must exceed fade_out_start ({fade_out_start})")
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    fade_dur = max(1.0, total_end - fade_out_start)
    af = (
        f"atrim=0:{total_end:.3f},"
        f"afade=t=in:st=0:d=0.06,"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_dur:.3f}:curve=qsin,"
        f"loudnorm=I={target_lufs}:TP=-1.0:LRA=11:linear=true"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src_wav),
            "-af", af,
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le",
            str(dst_wav),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(dst_wav),
            "-codec:a", "libmp3lame", "-b:a", "320k",
            str(dst_mp3),
        ],
        check=True,
    )
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(dst_wav),
            "-filter_complex", "ebur128=peak=true", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    audio, sr = load_wav_audio(dst_wav)
    gates = check_silence_gates(audio, sr)
    return {
        "wav": str(dst_wav),
        "mp3": str(dst_mp3),
        "duration_sec": round(len(audio) / sr, 3),
        "sample_rate": sr,
        "channels": int(audio.shape[1]),
        "peak_linear": round(float(np.max(np.abs(audio))), 5),
        "target_lufs": target_lufs,
        "silence_gates": gates,
        "ebur128_tail": "\n".join(proc.stderr.splitlines()[-12:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for stage in ("lm-continue", "dit-stitch"):
        p = sub.add_parser(stage)
        p.add_argument("--spec", type=Path, required=True, help="Expansion spec JSON")
        p.add_argument("--out-dir", type=Path, default=Path("output/full"), help="Output directory")
        p.add_argument("--acestep-root", default=None, help="Explicit ACE-Step-1.5 root path")
        p.add_argument("--lm-model", default="acestep-5Hz-lm-1.7B", help="5Hz LM checkpoint directory name")
        p.add_argument("--dit-model", default="acestep-v15-turbo", help="DiT checkpoint directory name")
        p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
        p.add_argument("--online", action="store_true", help="Allow online HuggingFace requests")
        p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    p_master = sub.add_parser("master", help="Master a stitched candidate WAV for streaming release")
    p_master.add_argument("--src-wav", type=Path, required=True)
    p_master.add_argument("--out-dir", type=Path, default=Path("output/full/final"))
    p_master.add_argument("--out-stem", type=str, required=True)
    p_master.add_argument("--fade-start", type=float, required=True)
    p_master.add_argument("--total-end", type=float, required=True)
    p_master.add_argument("--target-lufs", type=float, default=-12.8)

    args = parser.parse_args()

    if args.cmd == "master":
        dst_wav = args.out_dir / f"{args.out_stem}.wav"
        dst_mp3 = args.out_dir / f"{args.out_stem}.mp3"
        info = master_deliverable(
            src_wav=args.src_wav,
            dst_wav=dst_wav,
            dst_mp3=dst_mp3,
            fade_out_start=args.fade_start,
            total_end=args.total_end,
            target_lufs=args.target_lufs,
        )
        (args.out_dir / f"{args.out_stem}.master.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        return

    cache_existed = (Path.cwd() / ".cache").exists()
    configure_offline_env(online=args.online)
    ace_root = ensure_acestep_importable(args.acestep_root)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.cmd == "lm-continue":
            cmd_lm_continue(spec, args.spec, args.out_dir, ace_root, args.lm_model, args.device, args.overwrite)
        elif args.cmd == "dit-stitch":
            cmd_dit_stitch(spec, args.spec, args.out_dir, ace_root, args.dit_model, args.device, args.overwrite)
    finally:
        cleanup_cwd_cache(cache_existed)


if __name__ == "__main__":
    main()
