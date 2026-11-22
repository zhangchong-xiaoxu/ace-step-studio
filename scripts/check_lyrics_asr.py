#!/usr/bin/env python3
"""Offline sliding-window sung-lyrics ASR verification for 45s demos and full singles.

Supports:
- `sherpa_onnx` FireRed `zh_en` CTC in <=18s overlapping windows (with 0.3s zero lead-in so
  opening syllables are never dropped, and long tracks never suffer CTC tail truncation).
- Optional `faster_whisper` dual-engine cross-check (`--use-whisper`), automatically flagging
  pure-synth intro/outro windows where Whisper hallucinates phrases while CTC detects silence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from _studio_core import candidate_roots_from_ancestors, load_wav_audio


def locate_firered_model(explicit: str | None = None) -> Path | None:
    model_sub = "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    for env_name in ("ASR_MODEL_DIR", "KOUBO_ASR_MODEL"):
        if os.environ.get(env_name):
            cands.append(Path(os.environ[env_name]).expanduser())
    cands.append(Path.home() / ".cache" / "koubo" / model_sub)
    for ai_dir in candidate_roots_from_ancestors("ai-models"):
        cands.append(ai_dir / "koubo" / model_sub)
    for c in cands:
        if (c / "model.int8.onnx").exists() and (c / "tokens.txt").exists():
            return c.resolve()
    return None


def extract_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"\[[^\]]*\]", " ", text.lower())
    en_words = re.findall(r"[a-z0-9']+", cleaned)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", cleaned)
    return en_words + cjk_chars


def resample_to_16k_mono(src_wav: Path, dst_wav: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src_wav),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst_wav),
        ],
        check=True,
    )


def decode_ctc_windows(
    recognizer: Any,
    wav_16k: Path,
    window_sec: float = 18.0,
    hop_sec: float = 14.0,
) -> list[dict[str, Any]]:
    audio_2d, sr = load_wav_audio(wav_16k)
    audio = audio_2d.mean(axis=1)
    total_sec = len(audio) / sr
    windows: list[dict[str, Any]] = []
    start = 0.0
    idx = 0
    while start < total_sec - 0.3:
        end = min(total_sec, start + window_sec)
        seg = audio[int(start * sr) : int(end * sr)]
        if idx == 0:
            seg = np.concatenate([np.zeros(int(0.3 * sr), dtype=np.float32), seg])
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, seg)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        windows.append(
            {
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "text": text,
                "is_silent_vocal": len(extract_tokens(text)) == 0,
            }
        )
        if end >= total_sec:
            break
        start += hop_sec
        idx += 1
    return windows


def decode_whisper_segments(wav_path: Path, model_size: str) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _ = model.transcribe(str(wav_path), word_timestamps=True, vad_filter=False)
    out: list[dict[str, Any]] = []
    for seg in segments:
        out.append(
            {
                "start_sec": round(float(seg.start), 2),
                "end_sec": round(float(seg.end), 2),
                "text": seg.text.strip(),
            }
        )
    return out


def analyze_single_track(
    wav_path: Path,
    expected_lyrics: str,
    recognizer: Any | None,
    use_whisper: bool,
    whisper_model: str,
    tmp_dir: Path,
) -> dict[str, Any]:
    ctc_windows: list[dict[str, Any]] = []
    if recognizer is not None:
        wav_16k = tmp_dir / f"{wav_path.stem}_16k.wav"
        resample_to_16k_mono(wav_path, wav_16k)
        ctc_windows = decode_ctc_windows(recognizer, wav_16k)

    whisper_segs: list[dict[str, Any]] = []
    if use_whisper:
        whisper_segs = decode_whisper_segments(wav_path, whisper_model)

    suspected_hallucinations: list[dict[str, Any]] = []
    if ctc_windows and whisper_segs:
        for wseg in whisper_segs:
            overlapping = [
                cw for cw in ctc_windows if cw["start_sec"] <= wseg["end_sec"] and cw["end_sec"] >= wseg["start_sec"]
            ]
            if overlapping and all(cw["is_silent_vocal"] for cw in overlapping) and wseg["text"]:
                suspected_hallucinations.append(
                    {
                        "whisper_segment": wseg,
                        "reason": "Whisper emitted text while FireRed CTC detected zero sung tokens (<sil>); verify pure-synth intro/outro.",
                    }
                )

    combined_texts = [w["text"] for w in ctc_windows] + [s["text"] for s in whisper_segs]
    combined_str = " ".join(t for t in combined_texts if t)
    expected_vocab = sorted(set(extract_tokens(expected_lyrics)))
    recognized_vocab = set(extract_tokens(combined_str))
    hit_tokens = [tok for tok in expected_vocab if tok in recognized_vocab]
    hit_ratio = round(len(hit_tokens) / max(1, len(expected_vocab)), 3) if expected_vocab else None

    return {
        "wav": str(wav_path),
        "ctc_windows": ctc_windows,
        "whisper_segments": whisper_segs,
        "suspected_intro_outro_hallucinations": suspected_hallucinations,
        "expected_unique_tokens_count": len(expected_vocab),
        "hit_unique_tokens_count": len(hit_tokens),
        "vocab_hit_ratio": hit_ratio,
    }


def build_ctc_recognizer(model_dir_arg: str | None) -> Any | None:
    try:
        import sherpa_onnx  # type: ignore
    except ImportError:
        return None
    model_dir = locate_firered_model(model_dir_arg)
    if model_dir is None:
        return None
    return sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None, help="Batch mode: demos spec JSON")
    parser.add_argument("--demos-dir", type=Path, default=Path("output/demos"), help="Batch mode demos root")
    parser.add_argument("--only", default=None, help="Optional single demo ID in batch mode")
    parser.add_argument("--wav", type=Path, default=None, help="Single-track mode: WAV/MP3 path")
    parser.add_argument("--lyrics-file", type=Path, default=None, help="Expected lyrics text file")
    parser.add_argument("--out-json", type=Path, default=None, help="Output JSON report path for single-track mode")
    parser.add_argument("--model-dir", default=None, help="Explicit FireRed CTC model directory")
    parser.add_argument("--use-whisper", action="store_true", help="Also run faster-whisper word/segment read-back")
    parser.add_argument("--whisper-model", default="medium", help="faster-whisper model size (default: medium)")
    args = parser.parse_args()

    if not args.spec and not args.wav:
        parser.error("Provide either `--spec <demos.json>` (batch) or `--wav <audio.wav>` (single track).")

    recognizer = build_ctc_recognizer(args.model_dir)
    if recognizer is None and not args.use_whisper:
        raise RuntimeError(
            "Neither `sherpa_onnx` (+ FireRed model) nor `--use-whisper` is available in this Python environment. "
            "Run `python scripts/doctor.py` to see detected ASR environments."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if args.spec:
            demos = json.loads(args.spec.read_text(encoding="utf-8"))["demos"]
            if args.only:
                demos = [d for d in demos if d["id"] == args.only]
            for spec in demos:
                out_dir = args.demos_dir / spec["id"]
                wav_path = out_dir / f"{spec['id']}.wav"
                if not wav_path.exists():
                    raise FileNotFoundError(f"Missing demo WAV: {wav_path}")
                res = analyze_single_track(
                    wav_path=wav_path,
                    expected_lyrics=spec.get("lyrics", ""),
                    recognizer=recognizer,
                    use_whisper=args.use_whisper,
                    whisper_model=args.whisper_model,
                    tmp_dir=tmp_dir,
                )
                res["id"] = spec["id"]
                res["title"] = spec["title"]
                asr_path = out_dir / "asr.json"
                asr_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                pct = f"{res['vocab_hit_ratio']:.0%}" if res["vocab_hit_ratio"] is not None else "n/a"
                print(
                    f"ASR_OK {spec['id']} ({spec['title']}): hit={res['hit_unique_tokens_count']}/"
                    f"{res['expected_unique_tokens_count']} ({pct}) -> {asr_path}",
                    flush=True,
                )
        else:
            expected_text = args.lyrics_file.read_text(encoding="utf-8") if args.lyrics_file else ""
            res = analyze_single_track(
                wav_path=args.wav,
                expected_lyrics=expected_text,
                recognizer=recognizer,
                use_whisper=args.use_whisper,
                whisper_model=args.whisper_model,
                tmp_dir=tmp_dir,
            )
            text_out = json.dumps(res, ensure_ascii=False, indent=2)
            if args.out_json:
                args.out_json.parent.mkdir(parents=True, exist_ok=True)
                args.out_json.write_text(text_out, encoding="utf-8")
                print(f"ASR_SINGLE_OK -> {args.out_json}", flush=True)
            else:
                print(text_out, flush=True)


if __name__ == "__main__":
    main()
