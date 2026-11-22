#!/usr/bin/env python3
"""Generate 45s catchy chorus demos in split LM / DiT processes, or inspect WAV envelopes.

Hard rule on consumer GPUs (<=16GB VRAM):
Never initialize `acestep-5Hz-lm-1.7B` and `acestep-v15-turbo` in the same Python process.
Run `lm` first (writes `lm.json` and exits), then run `dit`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from _studio_core import (
    cleanup_cwd_cache,
    configure_offline_env,
    ensure_acestep_importable,
    load_wav_audio,
    recommend_cut_window,
    rms_envelope_half_sec,
)


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_demos(spec_file: Path, only_id: str | None = None) -> list[dict]:
    payload = json.loads(spec_file.read_text(encoding="utf-8"))
    demos = payload["demos"]
    if only_id:
        demos = [d for d in demos if d["id"] == only_id]
        if not demos:
            raise ValueError(f"Unknown demo id `{only_id}` in {spec_file}")
    return demos


def run_lm_stage(
    demos: list[dict],
    out_root: Path,
    ace_root: Path,
    lm_model: str,
    device: str,
    overwrite: bool,
) -> None:
    from acestep.llm_inference import LLMHandler

    lm = LLMHandler()
    status, ok = lm.initialize(
        checkpoint_dir=str(ace_root / "checkpoints"),
        lm_model_path=lm_model,
        backend="pt",
        device=device,
    )
    if not ok:
        raise RuntimeError(f"LM initialization failed: {status}")

    for spec in demos:
        out_dir = out_root / spec["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        lm_path = out_dir / "lm.json"
        if lm_path.exists() and not overwrite:
            print(f"SKIP_EXISTING_LM {spec['id']}: {lm_path}", flush=True)
            continue

        set_all_seeds(int(spec["seed"]))
        t0 = time.perf_counter()
        result = lm.generate_with_stop_condition(
            caption=spec["caption"],
            lyrics=spec["lyrics"],
            infer_type="llm_dit",
            target_duration=int(spec["duration"]),
            user_metadata={
                "bpm": spec["bpm"],
                "duration": spec["duration"],
                "keyscale": spec["keyscale"],
                "timesignature": str(spec["timesignature"]),
                "language": spec["vocal_language"],
            },
            use_cot_caption=False,
            use_cot_language=False,
        )
        elapsed = time.perf_counter() - t0
        if not result.get("success"):
            raise RuntimeError(f"LM failed for {spec['id']}: {result.get('error')}")
        codes = result.get("audio_codes") or ""
        count = len(re.findall(r"<\|audio_code_\d+\|>", codes))
        expected = int(spec["duration"]) * 5
        if count != expected:
            raise ValueError(f"{spec['id']}: expected {expected} 5Hz codes, got {count}")

        lm_path.write_text(
            json.dumps(
                {
                    "request": spec,
                    "model": lm_model,
                    "backend": "pt",
                    "generation_seconds": round(elapsed, 3),
                    "code_count": count,
                    "result": result,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"LM_OK {spec['id']} ({spec['title']}): {count} codes in {elapsed:.2f}s -> {lm_path}", flush=True)


def run_dit_stage(
    demos: list[dict],
    out_root: Path,
    ace_root: Path,
    dit_model: str,
    device: str,
    overwrite: bool,
) -> None:
    import soundfile as sf
    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(ace_root),
        config_path=dit_model,
        device=device,
    )
    if not ok:
        raise RuntimeError(f"DiT initialization failed: {status}")

    device_label = torch.cuda.get_device_name() if device.startswith("cuda") and torch.cuda.is_available() else device

    for spec in demos:
        out_dir = out_root / spec["id"]
        lm_path = out_dir / "lm.json"
        if not lm_path.exists():
            raise FileNotFoundError(f"Missing LM output for {spec['id']}: {lm_path} (run `lm` stage first)")
        wav_path = out_dir / f"{spec['id']}.wav"
        mp3_path = out_dir / f"{spec['id']}.mp3"
        if wav_path.exists() and not overwrite:
            print(f"SKIP_EXISTING_DIT {spec['id']}: {wav_path}", flush=True)
            continue

        saved = json.loads(lm_path.read_text(encoding="utf-8"))
        seed = int(spec["seed"])
        set_all_seeds(seed)
        render_dir = out_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)

        is_inst = bool(spec.get("instrumental", False))
        params = GenerationParams(
            caption=spec["caption"],
            lyrics="[Instrumental]" if is_inst else spec["lyrics"],
            duration=int(spec["duration"]),
            bpm=spec["bpm"],
            keyscale=spec["keyscale"],
            timesignature=str(spec["timesignature"]),
            vocal_language=spec["vocal_language"],
            seed=seed,
            audio_codes=saved["result"]["audio_codes"],
            instrumental=is_inst,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            use_cot_lyrics=False,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="flac")
        t0 = time.perf_counter()
        result = generate_music(dit, LLMHandler(), params, config, save_dir=str(render_dir))
        elapsed = time.perf_counter() - t0
        if not result.success or len(result.audios) != 1:
            raise RuntimeError(f"DiT render failed for {spec['id']}: {result.error}")

        src_flac = Path(result.audios[0]["path"])
        audio, rate = sf.read(src_flac, always_2d=True)
        duration = len(audio) / rate
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        if rate != 48000 or audio.shape[1] != 2:
            raise ValueError(f"Unexpected output format: {rate}Hz shape={audio.shape}")
        if not np.isfinite(audio).all() or peak <= 0 or rms <= 0.0005:
            raise ValueError(f"Invalid audio values: peak={peak}, rms={rms}")

        target_ceiling = 0.89125  # -1.0 dBFS linear ceiling
        if peak > target_ceiling:
            audio = audio * (target_ceiling / peak)
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio**2)))

        sf.write(wav_path, audio, rate, subtype="PCM_24")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav_path),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "320k",
                str(mp3_path),
            ],
            check=True,
        )
        cut_rec = recommend_cut_window(audio, rate)
        env_stats = rms_envelope_half_sec(audio, rate)
        env_summary = {k: v for k, v in env_stats.items() if k != "all_bins_rms"}
        report = {
            "id": spec["id"],
            "title": spec["title"],
            "dit_model": dit_model,
            "device": device_label,
            "wav": str(wav_path),
            "mp3": str(mp3_path),
            "duration_seconds": round(duration, 3),
            "sample_rate": rate,
            "channels": int(audio.shape[1]),
            "peak": round(peak, 5),
            "rms": round(rms, 5),
            "render_seconds": round(elapsed, 3),
            "envelope": env_summary,
            "continuation_recommendation": cut_rec,
        }
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"DIT_OK {spec['id']} ({spec['title']}): {duration:.2f}s peak={peak:.4f} rms={rms:.4f} "
            f"active={env_summary['first_active_sec']}s..{env_summary['last_active_sec']}s -> {mp3_path}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for stage in ("lm", "dit"):
        p = sub.add_parser(stage)
        p.add_argument("--spec", type=Path, required=True, help="Path to demos spec JSON")
        p.add_argument("--out-dir", type=Path, default=Path("output/demos"), help="Output directory")
        p.add_argument("--only", default=None, help="Optional single demo ID to run")
        p.add_argument("--acestep-root", default=None, help="Explicit ACE-Step-1.5 root path")
        p.add_argument("--lm-model", default="acestep-5Hz-lm-1.7B", help="5Hz LM checkpoint directory name")
        p.add_argument("--dit-model", default="acestep-v15-turbo", help="DiT checkpoint directory name")
        p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
        p.add_argument("--online", action="store_true", help="Allow online HuggingFace requests")
        p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    p_env = sub.add_parser("inspect-envelope", help="Scan a WAV in 0.5s RMS bins and recommend continuation cut point")
    p_env.add_argument("--wav", type=Path, required=True, help="Path to WAV file")

    args = parser.parse_args()

    if args.cmd == "inspect-envelope":
        audio, sr = load_wav_audio(args.wav)
        rec = recommend_cut_window(audio, sr)
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return

    cache_existed = (Path.cwd() / ".cache").exists()
    configure_offline_env(online=args.online)
    ace_root = ensure_acestep_importable(args.acestep_root)
    demos = load_demos(args.spec, args.only)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.cmd == "lm":
            run_lm_stage(demos, args.out_dir, ace_root, args.lm_model, args.device, args.overwrite)
        elif args.cmd == "dit":
            run_dit_stage(demos, args.out_dir, ace_root, args.dit_model, args.device, args.overwrite)
    finally:
        cleanup_cwd_cache(cache_existed)


if __name__ == "__main__":
    main()
