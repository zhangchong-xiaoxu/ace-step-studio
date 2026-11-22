#!/usr/bin/env python3
"""Validate streaming QC gates and package final WAV, MP3, LRC, plain lyrics, and optional cover.

Checks enforced before writing release artifacts:
1. Audio format: 48,000 Hz stereo WAV, finite non-silent waveform, internal `< -42 dBFS` silence <= 3.0s,
   trailing `< -42 dBFS` silence <= 1.5s.
2. Synced lyrics: non-empty, monotonic `[MM:SS.cc]` timestamps within track duration, zero leaked
   section tags (`[Verse]`, `[Chorus]`, `[Bridge]`, `[Instrumental]`).
3. Optional cover image (`--cover`): 3000x3000 px, 3-channel RGB/sRGB (rejects CMYK/RGBA), <= 10 MB.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from _studio_core import (
    check_silence_gates,
    format_lrc_time,
    inspect_cover_image,
    load_wav_audio,
    validate_synced_lyrics,
)


def sanitize_filename_stem(title: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip()
    if not cleaned:
        raise ValueError(f"Invalid track title for filename: {title!r}")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True, help="Mastered 48kHz/24-bit WAV")
    parser.add_argument("--mp3", type=Path, default=None, help="Optional 320kbps MP3 preview")
    parser.add_argument("--synced-json", type=Path, required=True, help="Synced lyrics JSON (`lines: [{t, line}]`)")
    parser.add_argument("--title", required=True, help="Exact release track title")
    parser.add_argument("--artist", required=True, help="Exact release artist name")
    parser.add_argument("--album", default=None, help="Album title (defaults to track title for singles)")
    parser.add_argument("--cover", type=Path, default=None, help="Optional 3000x3000 sRGB cover JPEG/PNG to validate & copy")
    parser.add_argument("--out-dir", type=Path, default=Path("output/full/release"), help="Release output directory")
    parser.add_argument("--allow-silence-warnings", action="store_true", help="Warn instead of failing on silence gates")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    if not args.wav.exists():
        raise FileNotFoundError(f"Master WAV not found: {args.wav}")
    audio, sr = load_wav_audio(args.wav)
    duration_sec = round(len(audio) / sr, 3)
    channels = int(audio.shape[1])
    peak = float(np.max(np.abs(audio)))

    if sr != 48000:
        errors.append(f"Expected 48000 Hz sample rate for streaming master, got {sr} Hz.")
    if channels != 2:
        errors.append(f"Expected 2-channel stereo master, got {channels} channel(s).")
    if not np.isfinite(audio).all() or peak <= 0.01:
        errors.append(f"Master audio is silent or non-finite (peak={peak:.5f}).")

    gates = check_silence_gates(audio, sr)
    if not gates["passes_internal_3s_gate"]:
        msg = f"Internal quiet section ({gates['max_internal_quiet_sec']}s < -42 dBFS) exceeds 3.0s streaming QC gate."
        (warnings if args.allow_silence_warnings else errors).append(msg)
    if not gates["passes_trailing_1p5s_gate"]:
        msg = f"Trailing quiet tail ({gates['trailing_quiet_sec']}s < -42 dBFS) exceeds 1.5s recommended trim."
        (warnings if args.allow_silence_warnings else errors).append(msg)

    synced = json.loads(args.synced_json.read_text(encoding="utf-8"))
    lines = synced.get("lines", [])
    errors.extend(validate_synced_lyrics(lines, duration_sec=duration_sec))

    cover_report = None
    if args.cover:
        if not args.cover.exists():
            errors.append(f"Cover file not found: {args.cover}")
        else:
            cover_report = inspect_cover_image(args.cover)
            errors.extend(cover_report["errors"])

    if errors:
        print("RELEASE_QC_FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename_stem(args.title)
    album = args.album or args.title

    dst_wav = args.out_dir / f"{stem}.wav"
    shutil.copy2(args.wav, dst_wav)

    dst_mp3 = None
    if args.mp3:
        if not args.mp3.exists():
            raise FileNotFoundError(f"MP3 file not found: {args.mp3}")
        dst_mp3 = args.out_dir / f"{stem}.mp3"
        shutil.copy2(args.mp3, dst_mp3)

    dst_cover = None
    if args.cover and cover_report:
        dst_cover = args.out_dir / f"{stem}_cover{args.cover.suffix.lower()}"
        shutil.copy2(args.cover, dst_cover)

    lrc_lines = [
        f"[ti:{args.title}]",
        f"[ar:{args.artist}]",
        f"[al:{album}]",
    ]
    plain_lines: list[str] = []
    for item in lines:
        text = str(item["line"]).strip()
        lrc_lines.append(f"{format_lrc_time(float(item['t']))}{text}")
        plain_lines.append(text)

    lrc_path = args.out_dir / f"{stem}.lrc"
    txt_path = args.out_dir / f"{stem}_lyrics.txt"
    lrc_path.write_text("\n".join(lrc_lines) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(plain_lines) + "\n", encoding="utf-8")

    manifest = {
        "title": args.title,
        "artist": args.artist,
        "album": album,
        "duration_sec": duration_sec,
        "sample_rate": sr,
        "channels": channels,
        "peak_linear": round(peak, 5),
        "synced_lines_count": len(plain_lines),
        "silence_gates": gates,
        "cover": cover_report,
        "warnings": warnings,
        "files": {
            "wav": str(dst_wav),
            "mp3": str(dst_mp3) if dst_mp3 else None,
            "lrc": str(lrc_path),
            "plain_lyrics": str(txt_path),
            "cover": str(dst_cover) if dst_cover else None,
        },
        "aigc_compliance_reminder": "Check the platform's AI-generated / AIGC declaration box during upload; never print AIGC or model watermarks on the 3000x3000 single cover.",
    }
    manifest_path = args.out_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RELEASE_PACK_OK -> {args.out_dir} ({len(plain_lines)} synced lines)", flush=True)


if __name__ == "__main__":
    main()
