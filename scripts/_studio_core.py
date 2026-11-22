"""Core audio, splice, LRC, cover-inspection, and path-discovery utilities for ace-step-studio.

Designed so lightweight commands (`doctor.py`, `package_release.py`, and unit tests) run without
requiring PyTorch or GPU drivers, while generation scripts share the exact same tested math.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import struct
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

STRUCTURE_TAG_RE = re.compile(
    r"^\s*\[(?:intro|verse|pre-chorus|chorus|bridge|outro|instrumental|drop|build|breakdown|hook|solo|fade[^\]]*)[^\]]*\]\s*$",
    re.IGNORECASE,
)


def candidate_roots_from_ancestors(folder_name: str, extra_rel: tuple[str, ...] = ()) -> list[Path]:
    """Walk ancestors of CWD and repo root to discover sibling directories without hardcoded paths."""
    anchors = [Path.cwd().resolve(), Path(__file__).resolve().parent.parent]
    seen: set[Path] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        try:
            rp = p.expanduser().resolve()
        except OSError:
            return
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    for anchor in anchors:
        for base in [anchor, *anchor.parents]:
            add(base / folder_name)
            for rel in extra_rel:
                add(base / rel / folder_name)
    add(Path.home() / folder_name)
    for rel in extra_rel:
        add(Path.home() / rel / folder_name)
    return out


def find_acestep_root(explicit: str | Path | None = None) -> Path | None:
    """Locate a local ACE-Step-1.5 checkout via CLI arg, ACESTEP_ROOT env, or ancestor scan."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_val = os.environ.get("ACESTEP_ROOT")
    if env_val:
        candidates.append(Path(env_val).expanduser())
    candidates.extend(
        candidate_roots_from_ancestors("ACE-Step-1.5", extra_rel=("programming", "projects", "src"))
    )

    for cand in candidates:
        if (cand / "acestep").is_dir() and (cand / "checkpoints").is_dir():
            return cand.resolve()
    return None


def find_acestep_python(acestep_root: Path | None) -> Path | None:
    """Return the Python executable inside ACE-Step-1.5's virtualenv if present."""
    if not acestep_root:
        return None
    rels = (
        Path(".venv/Scripts/python.exe"),
        Path(".venv/bin/python"),
        Path("venv/Scripts/python.exe"),
        Path("venv/bin/python"),
    )
    for rel in rels:
        py = acestep_root / rel
        if py.exists():
            return py
    return None


def ensure_acestep_importable(explicit_root: str | Path | None = None) -> Path:
    """Ensure `acestep` can be imported and return the resolved ACE-Step-1.5 root directory."""
    root = find_acestep_root(explicit_root)
    if root is None:
        raise FileNotFoundError(
            "Could not locate ACE-Step-1.5 checkout (containing `acestep/` and `checkpoints/`). "
            "Pass `--acestep-root /path/to/ACE-Step-1.5` or set `ACESTEP_ROOT`."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def configure_offline_env(online: bool = False) -> None:
    """Set UTF-8 and offline HuggingFace flags unless `--online` is explicitly requested."""
    os.environ.setdefault("PYTHONUTF8", "1")
    if not online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def cleanup_cwd_cache(existed_before: bool) -> None:
    """Remove transient `./.cache` created by ACE-Step only if it did not exist beforehand."""
    if existed_before:
        return
    cache_dir = Path.cwd() / ".cache"
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)


def load_wav_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV/FLAC file as float32 stereo/2D array `(frames, channels)` using soundfile or stdlib wave."""
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return audio, int(sr)
    except ImportError:
        pass

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        raw_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        packed = (
            raw_u8[:, 0].astype(np.int32)
            | (raw_u8[:, 1].astype(np.int32) << 8)
            | (raw_u8[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        samples = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sampwidth} bytes in {path}")

    return samples.reshape(-1, channels), int(sr)


def rms_envelope_half_sec(
    audio: np.ndarray,
    rate: int,
    active_threshold: float = 0.015,
) -> dict[str, Any]:
    """Compute 0.5-second RMS bins and detect active intro/outro boundaries."""
    mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    win = max(1, rate // 2)
    n_bins = len(mono) // win
    vals = [
        round(float(np.sqrt(np.mean(mono[i * win : (i + 1) * win] ** 2))), 5)
        for i in range(n_bins)
    ]
    dbfs_vals = [round(20.0 * math.log10(max(v, 1e-7)), 2) for v in vals]
    first_active_sec = next((round(i * 0.5, 2) for i, v in enumerate(vals) if v > active_threshold), None)
    last_active_sec = next(
        (round((i + 1) * 0.5, 2) for i in range(len(vals) - 1, -1, -1) if vals[i] > active_threshold),
        None,
    )
    quiet_bins_below_neg42 = [round(i * 0.5, 1) for i, db in enumerate(dbfs_vals) if db < -42.0]
    return {
        "bin_seconds": 0.5,
        "num_bins": n_bins,
        "first_active_sec": first_active_sec,
        "last_active_sec": last_active_sec,
        "first_6_bins_rms": vals[:6],
        "last_12_bins_rms": vals[-12:],
        "quiet_bins_below_neg42dbfs": quiet_bins_below_neg42,
        "all_bins_rms": vals,
    }


def recommend_cut_window(
    audio: np.ndarray,
    rate: int,
    crossfade_duration_sec: float = 0.8,
    min_sustain_ratio: float = 0.30,
) -> dict[str, Any]:
    """Recommend a safe lossless continuation cut window before end-of-demo decay."""
    env = rms_envelope_half_sec(audio, rate)
    vals: list[float] = env["all_bins_rms"]
    if len(vals) < 10:
        raise ValueError("Audio too short to recommend a continuation cut window (< 5s)")

    body_vals = vals[min(4, len(vals) // 4) : max(6, len(vals) - 6)]
    median_body_rms = float(np.median(body_vals)) if body_vals else float(np.max(vals))
    sustain_floor = max(0.020, median_body_rms * min_sustain_ratio)

    last_sustained_bin = len(vals) - 1
    while last_sustained_bin > 4 and vals[last_sustained_bin] < sustain_floor:
        last_sustained_bin -= 1

    xf_end_sec = round((last_sustained_bin + 1) * 0.5, 2)
    xf_end_sec = min(xf_end_sec, round(len(audio) / rate - 0.4, 2))
    prefix_codes_count = int(math.floor(xf_end_sec * 5.0 + 1e-6))
    xf_end_aligned = round(prefix_codes_count / 5.0, 2)
    xf_start_aligned = round(max(1.0, xf_end_aligned - crossfade_duration_sec), 2)

    return {
        "median_body_rms": round(median_body_rms, 5),
        "sustain_floor_rms": round(sustain_floor, 5),
        "recommended_crossfade_start_sec": xf_start_aligned,
        "recommended_crossfade_end_sec": xf_end_aligned,
        "recommended_prefix_codes_count": prefix_codes_count,
        "first_active_sec": env["first_active_sec"],
        "last_active_sec": env["last_active_sec"],
        "last_12_bins_rms": env["last_12_bins_rms"],
    }


def stitch_equal_power_crossfade(
    orig_audio: np.ndarray,
    coded_audio: np.ndarray,
    sr: int,
    xf_start_sec: float,
    xf_end_sec: float,
    max_lag_sec: float = 0.040,
    lag_step: int = 24,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match body RMS, phase-align within ±max_lag_sec, and splice 0..xf_start_sec bit-for-bit."""
    if orig_audio.ndim != 2 or coded_audio.ndim != 2 or orig_audio.shape[1] != coded_audio.shape[1]:
        raise ValueError(f"Shape mismatch: orig={orig_audio.shape}, coded={coded_audio.shape}")
    if xf_end_sec <= xf_start_sec:
        raise ValueError(f"crossfade_end_sec ({xf_end_sec}) must exceed crossfade_start_sec ({xf_start_sec})")

    body_start = int(min(3.0, xf_start_sec * 0.2) * sr)
    body_end = int(max(body_start + sr, xf_start_sec - 1.0) * sr)
    body_end = min(body_end, len(orig_audio), len(coded_audio))

    rms_orig = float(np.sqrt(np.mean(orig_audio[body_start:body_end] ** 2)))
    rms_coded = float(np.sqrt(np.mean(coded_audio[body_start:body_end] ** 2)))
    gain = float(np.clip(rms_orig / max(rms_coded, 1e-6), 0.75, 1.35))
    coded_scaled = np.clip(coded_audio * gain, -0.98, 0.98).astype(orig_audio.dtype, copy=False)

    xf_start = int(round(xf_start_sec * sr))
    xf_end = int(round(xf_end_sec * sr))
    xf_len = xf_end - xf_start
    if xf_end > len(orig_audio):
        raise ValueError(f"crossfade_end_sec ({xf_end_sec}s) exceeds base audio ({len(orig_audio)/sr:.2f}s)")

    max_lag = int(round(max_lag_sec * sr))
    step = max(1, lag_step)
    ref_seg = orig_audio[xf_start:xf_end].mean(axis=1)
    best_lag = 0
    best_corr = -1e18
    for lag in range(-max_lag, max_lag + 1, step):
        c0 = xf_start + lag
        c1 = xf_end + lag
        if c0 < 0 or c1 > len(coded_scaled):
            continue
        cand_seg = coded_scaled[c0:c1].mean(axis=1)
        if len(cand_seg) == len(ref_seg):
            corr = float(np.dot(ref_seg, cand_seg))
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

    theta = np.linspace(0.0, math.pi / 2.0, xf_len, dtype=np.float32)[:, None]
    w_orig = np.cos(theta)
    w_coded = np.sin(theta)

    head = orig_audio[:xf_start]
    mid = (
        w_orig * orig_audio[xf_start:xf_end]
        + w_coded * coded_scaled[xf_start + best_lag : xf_end + best_lag]
    ).astype(orig_audio.dtype, copy=False)
    tail = coded_scaled[xf_end + best_lag :]
    stitched = np.concatenate([head, mid, tail], axis=0)

    max_prefix_diff = float(np.max(np.abs(stitched[:xf_start] - orig_audio[:xf_start]))) if xf_start > 0 else 0.0
    metrics = {
        "rms_gain_applied": round(gain, 4),
        "best_lag_samples": int(best_lag),
        "best_lag_ms": round(best_lag * 1000.0 / sr, 2),
        "prefix_bit_exact_max_diff": max_prefix_diff,
        "duration_sec": round(len(stitched) / sr, 3),
    }
    return stitched, metrics


def check_silence_gates(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Evaluate streaming silence gates: internal < -42 dBFS run <= 3.0s, trailing <= 1.5s."""
    env = rms_envelope_half_sec(audio, sr)
    vals: list[float] = env["all_bins_rms"]
    quiet_flags = [20.0 * math.log10(max(v, 1e-7)) < -42.0 for v in vals]

    trailing_bins = 0
    for q in reversed(quiet_flags):
        if q:
            trailing_bins += 1
        else:
            break
    trailing_sec = round(trailing_bins * 0.5, 2)

    max_internal_bins = 0
    cur_run = 0
    limit_idx = len(quiet_flags) - trailing_bins
    for q in quiet_flags[:limit_idx]:
        if q:
            cur_run += 1
            max_internal_bins = max(max_internal_bins, cur_run)
        else:
            cur_run = 0
    max_internal_sec = round(max_internal_bins * 0.5, 2)

    return {
        "max_internal_quiet_sec": max_internal_sec,
        "trailing_quiet_sec": trailing_sec,
        "quiet_bins_below_neg42dbfs": env["quiet_bins_below_neg42dbfs"],
        "passes_internal_3s_gate": max_internal_sec <= 3.0,
        "passes_trailing_1p5s_gate": trailing_sec <= 1.5,
    }


def format_lrc_time(sec: float) -> str:
    if sec < 0:
        raise ValueError(f"Negative LRC timestamp: {sec}")
    total_cs = int(round(sec * 100))
    m, rem_cs = divmod(total_cs, 6000)
    s = rem_cs / 100.0
    return f"[{m:02d}:{s:05.2f}]"


def validate_synced_lyrics(lines: list[dict[str, Any]], duration_sec: float | None = None) -> list[str]:
    """Return a list of validation errors for synced lyrics lines (empty list = valid)."""
    errors: list[str] = []
    if not lines:
        errors.append("Synced lyrics `lines` list is empty.")
        return errors

    prev_t = -1.0
    for idx, item in enumerate(lines):
        if "t" not in item or "line" not in item:
            errors.append(f"Line #{idx} missing required keys `t` or `line`.")
            continue
        t = float(item["t"])
        text = str(item["line"]).strip()
        if not text:
            errors.append(f"Line #{idx} at {t:.2f}s has empty text.")
        if STRUCTURE_TAG_RE.match(text) or re.search(r"\[(?:Verse|Chorus|Bridge|Intro|Outro|Instrumental)", text, re.I):
            errors.append(f"Line #{idx} ('{text}') contains a structure tag; strip section headers from LRC.")
        if t < 0:
            errors.append(f"Line #{idx} has negative timestamp {t:.2f}s.")
        if t < prev_t:
            errors.append(f"Line #{idx} timestamp {t:.2f}s is not monotonic (previous {prev_t:.2f}s).")
        if duration_sec is not None and t > duration_sec + 0.25:
            errors.append(f"Line #{idx} timestamp {t:.2f}s exceeds audio duration {duration_sec:.2f}s.")
        prev_t = t
    return errors


def inspect_cover_image(path: Path) -> dict[str, Any]:
    """Pure-stdlib PNG/JPEG header inspector enforcing 3000x3000 px, RGB/sRGB (not CMYK), <= 10 MB."""
    size_bytes = path.stat().st_size
    data = path.read_bytes()
    fmt = None
    width = height = 0
    color_mode = "unknown"

    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 26:
        fmt = "PNG"
        width, height = struct.unpack(">II", data[16:24])
        color_type = data[25]
        color_mode = {0: "Grayscale", 2: "RGB", 3: "Indexed", 4: "Grayscale+Alpha", 6: "RGBA"}.get(
            color_type, f"PNG_color_{color_type}"
        )
    elif data.startswith(b"\xff\xd8"):
        fmt = "JPEG"
        pos = 2
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while pos < len(data) - 1:
            if data[pos] != 0xFF:
                pos += 1
                continue
            while pos < len(data) and data[pos] == 0xFF:
                pos += 1
            if pos >= len(data):
                break
            marker = data[pos]
            pos += 1
            if marker in {0xD8, 0xD9} or (0xD0 <= marker <= 0xD7):
                continue
            if pos + 2 > len(data):
                break
            seg_len = struct.unpack(">H", data[pos : pos + 2])[0]
            if marker in sof_markers and pos + 8 <= len(data):
                height, width, components = struct.unpack(">HHB", data[pos + 3 : pos + 8])
                color_mode = {1: "Grayscale", 3: "RGB", 4: "CMYK"}.get(components, f"{components}-ch")
                break
            pos += seg_len

    errors: list[str] = []
    if fmt is None:
        errors.append("Unsupported cover format (must be JPEG or PNG).")
    if (width, height) != (3000, 3000):
        errors.append(f"Cover dimensions must be 3000x3000 px (got {width}x{height}).")
    if color_mode == "CMYK":
        errors.append("Cover color mode is CMYK; music platforms require sRGB (3-channel RGB).")
    elif color_mode not in {"RGB"}:
        errors.append(f"Cover should be 3-channel RGB/sRGB without alpha (got {color_mode}).")
    if size_bytes > 10 * 1024 * 1024:
        errors.append(f"Cover file size {size_bytes / (1024*1024):.2f} MB exceeds 10 MB limit.")

    return {
        "path": str(path),
        "format": fmt,
        "width": width,
        "height": height,
        "color_mode": color_mode,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "ok": len(errors) == 0,
        "errors": errors,
    }
