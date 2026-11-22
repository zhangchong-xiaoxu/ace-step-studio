#!/usr/bin/env python3
"""Step 0: Read-only environment check for ace-step-studio.

Usage:
    python scripts/doctor.py [--acestep-root /path/to/ACE-Step-1.5] [--json]

Standard library only. Never installs packages, drivers, or model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

OS_KEY = {"Windows": "win", "Darwin": "mac"}.get(platform.system(), "linux")
INSTALL_HINTS = {
    "ffmpeg": {
        "win": "winget install Gyan.FFmpeg",
        "mac": "brew install ffmpeg",
        "linux": "sudo apt-get install ffmpeg",
    }
}


def run_quiet(*cmd: str, timeout: int = 10) -> str | None:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return res.stdout.strip() if res.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_gpu() -> dict | None:
    smi = run_quiet("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
    if smi:
        first = smi.splitlines()[0]
        if "," in first:
            name, mem = first.rsplit(",", 1)
            vram_mb = int(float(mem.strip()))
            return {
                "backend": "cuda",
                "name": name.strip(),
                "vram_mb": vram_mb,
                "split_process_required": vram_mb <= 20480,
            }
    if OS_KEY == "mac" and platform.machine() == "arm64":
        return {
            "backend": "mps",
            "name": "Apple Silicon (MPS)",
            "vram_mb": None,
            "split_process_required": True,
        }
    return None


def scan_ancestor_dirs(folder_name: str, extra_rel: tuple[str, ...] = ()) -> list[Path]:
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


def locate_acestep(explicit: str | None) -> Path | None:
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    if os.environ.get("ACESTEP_ROOT"):
        cands.append(Path(os.environ["ACESTEP_ROOT"]).expanduser())
    cands.extend(scan_ancestor_dirs("ACE-Step-1.5", ("programming", "projects", "src")))
    for c in cands:
        if (c / "acestep").is_dir():
            return c.resolve()
    return None


def locate_venv_python(root: Path | None) -> Path | None:
    if not root:
        return None
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python", "venv/Scripts/python.exe", "venv/bin/python"):
        cand = root / rel
        if cand.exists():
            return cand
    return None


def has_module(py_exe: str, mod: str) -> bool:
    return (
        subprocess.run(
            [py_exe, "-c", f"import {mod}"],
            capture_output=True,
            timeout=15,
        ).returncode
        == 0
    )


def locate_firered_model() -> Path | None:
    model_sub = "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"
    cands: list[Path] = []
    for env_name in ("ASR_MODEL_DIR", "KOUBO_ASR_MODEL"):
        if os.environ.get(env_name):
            cands.append(Path(os.environ[env_name]).expanduser())
    cands.append(Path.home() / ".cache" / "koubo" / model_sub)
    for ai_dir in scan_ancestor_dirs("ai-models"):
        cands.append(ai_dir / "koubo" / model_sub)
    for c in cands:
        if (c / "model.int8.onnx").exists() and (c / "tokens.txt").exists():
            return c.resolve()
    return None


def detect_asr(ace_py: Path | None) -> dict:
    py_candidates: list[str] = []
    if os.environ.get("ASR_PYTHON"):
        py_candidates.append(os.environ["ASR_PYTHON"])
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        koubo_win = Path(local_app) / "koubo-venv" / "Scripts" / "python.exe"
        if koubo_win.exists():
            py_candidates.append(str(koubo_win))
    koubo_posix = Path.home() / ".local" / "share" / "koubo-venv" / "bin" / "python"
    if koubo_posix.exists():
        py_candidates.append(str(koubo_posix))
    if ace_py:
        py_candidates.append(str(ace_py))
    py_candidates.append(sys.executable)

    firered_dir = locate_firered_model()
    seen: set[str] = set()
    for py in py_candidates:
        if py in seen or not Path(py).exists():
            continue
        seen.add(py)
        sherpa_ok = has_module(py, "sherpa_onnx")
        whisper_ok = has_module(py, "faster_whisper")
        if sherpa_ok or whisper_ok:
            return {
                "available": bool((sherpa_ok and firered_dir) or whisper_ok),
                "asr_python": py,
                "sherpa_onnx": sherpa_ok,
                "firered_model_dir": str(firered_dir) if firered_dir else None,
                "faster_whisper": whisper_ok,
            }
    return {
        "available": False,
        "asr_python": None,
        "sherpa_onnx": False,
        "firered_model_dir": str(firered_dir) if firered_dir else None,
        "faster_whisper": False,
    }


def dir_nonempty(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acestep-root", default=None, help="Explicit path to ACE-Step-1.5 checkout")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    args = parser.parse_args()

    checks: list[dict] = []
    next_steps: list[str] = []

    def record(cid: str, ok: bool, detail: str, fix: str = "", required: bool = True) -> None:
        checks.append({"id": cid, "ok": bool(ok), "required": required, "detail": detail})
        if not ok and fix and fix not in next_steps:
            next_steps.append(fix)

    record("python", sys.version_info >= (3, 10), platform.python_version(), "Install Python 3.10+")
    ffmpeg_bin = shutil.which("ffmpeg")
    record("ffmpeg", bool(ffmpeg_bin), ffmpeg_bin or "not on PATH", INSTALL_HINTS["ffmpeg"][OS_KEY])

    gpu_info = detect_gpu()
    record(
        "gpu",
        gpu_info is not None,
        f"{gpu_info['name']} ({gpu_info['vram_mb']} MiB)" if gpu_info and gpu_info["vram_mb"] else (gpu_info["name"] if gpu_info else "No CUDA/MPS GPU detected"),
        "Ensure an NVIDIA GPU (12-16GB+ VRAM recommended) with working drivers is available",
    )

    ace_root = locate_acestep(args.acestep_root)
    record(
        "acestep_root",
        ace_root is not None,
        str(ace_root) if ace_root else "ACE-Step-1.5 directory not found",
        "git clone https://github.com/ace-step/ACE-Step-1.5 && set ACESTEP_ROOT=/path/to/ACE-Step-1.5",
    )

    ace_py = locate_venv_python(ace_root)
    if ace_py is None and ace_root is not None and has_module(sys.executable, "torch"):
        ace_py = Path(sys.executable)
    record(
        "acestep_python",
        ace_py is not None,
        str(ace_py) if ace_py else "No .venv Python found inside ACE-Step-1.5",
        "Set up ACE-Step-1.5 Python environment per its official README (`uv sync` / `.venv`)",
    )

    lm_ckpt = ace_root / "checkpoints" / "acestep-5Hz-lm-1.7B" if ace_root else None
    dit_ckpt = ace_root / "checkpoints" / "acestep-v15-turbo" if ace_root else None
    record(
        "checkpoint_lm",
        bool(lm_ckpt and dir_nonempty(lm_ckpt)),
        str(lm_ckpt) if lm_ckpt and dir_nonempty(lm_ckpt) else "Missing checkpoints/acestep-5Hz-lm-1.7B",
        "Download `acestep-5Hz-lm-1.7B` into ACE-Step-1.5/checkpoints/ (after user approval)",
    )
    record(
        "checkpoint_dit",
        bool(dit_ckpt and dir_nonempty(dit_ckpt)),
        str(dit_ckpt) if dit_ckpt and dir_nonempty(dit_ckpt) else "Missing checkpoints/acestep-v15-turbo",
        "Download `acestep-v15-turbo` into ACE-Step-1.5/checkpoints/ (after user approval)",
    )

    asr_info = detect_asr(ace_py)
    record(
        "local_asr (optional)",
        asr_info["available"],
        (
            f"python={asr_info['asr_python']} (sherpa_onnx={asr_info['sherpa_onnx']}, faster_whisper={asr_info['faster_whisper']})"
            if asr_info["available"]
            else "Not configured (optional; needed only for automated `check_lyrics_asr.py`)"
        ),
        required=False,
    )

    ready = all(c["ok"] for c in checks if c["required"])
    report = {
        "os": OS_KEY,
        "ready": ready,
        "gpu": gpu_info,
        "acestep_root": str(ace_root) if ace_root else None,
        "acestep_python": str(ace_py) if ace_py else None,
        "asr": asr_info,
        "checks": checks,
        "next_steps": next_steps,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"OS: {OS_KEY}   Ready: {'YES' if ready else 'NO'}")
    for c in checks:
        tag = "OK" if c["ok"] else ("OPT" if not c["required"] else "MISSING")
        print(f"  [{tag:7s}] {c['id']}: {c['detail']}")
    if gpu_info and gpu_info.get("split_process_required"):
        print("  [GUARD  ] <=20GB VRAM: always run `lm` and `dit` in separate processes (never same-process).")
    if next_steps:
        print("\nNext steps (ask the user for approval before installing/downloading):")
        for s in next_steps:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
