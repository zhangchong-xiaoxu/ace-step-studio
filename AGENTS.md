# AGENTS.md

Repository entry map for coding agents. Full workflow instructions live in [SKILL.md](SKILL.md); human overview lives in [README.md](README.md); cross-session status lives in [docs/PROGRESS.md](docs/PROGRESS.md) and [docs/README.md](docs/README.md).

## Hard Guardrails

1. **Read-only preflight first**: Run `python scripts/doctor.py --json` before running any generation command. Never install Python packages, system binaries (`ffmpeg`), or multi-GB model weights without explicit user consent.
2. **Split processes on ≤16GB GPUs**: Never import and initialize `LLMHandler` (`acestep-5Hz-lm-1.7B`) and `AceStepHandler` (`acestep-v15-turbo`) in the same Python process. Always run `lm` / `lm-continue` in one process, wait for exit, then run `dit` / `dit-stitch`.
3. **Do not judge music quality without human ears**: Envelopes, LUFS/dBTP, and ASR read-back only verify technical gates. Always pause after 45s hook demos so the user can listen and choose a take.
4. **Keep generated media out of Git**: `output/`, `raw/`, `.cache/`, and audio binaries (`*.wav`, `*.mp3`, `*.flac`) are git-ignored.

## Deterministic Verification

```bash
python -m unittest discover -s tests -v
python scripts/doctor.py --json
```
