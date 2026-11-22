---
name: ace-step-studio
description: Local AI songwriting, lossless full-song continuation, offline sung-lyrics LRC alignment, and streaming-ready mastering on consumer GPUs (16GB VRAM) with ACE-Step 1.5. Use when generating original vocal songs or instrumental BGM locally, expanding a 45s hook demo into a 2-3 min full single without repaint glitches, checking sung lyrics offline, or packaging tracks for streaming platforms (Qishui Music / Douyin / Spotify) and multi-platform video BGM.
---

# ACE-Step Studio Skill

Turn a one-line song direction into a **streaming-ready full vocal single** (and optional **copyright-isolated instrumental video BGM**) locally on a single 16GB consumer GPU using open-source [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (MIT) — **zero cloud API keys, zero subscription fees**.

## Why This Workflow Exists (vs. Upstream Defaults)

1. **16GB VRAM Split-Process Guard (`0xC0000005` fix)**: Initializing `acestep-5Hz-lm-1.7B` and `acestep-v15-turbo` in the same Python process crashes with `0xC0000005` on Windows 16GB GPUs (e.g. RTX 4060 Ti 16GB + PyTorch CUDA). Every generation in this skill splits **Stage 1 (`lm` → saves 5Hz `audio_codes` JSON and exits)** and **Stage 2 (`dit` → renders 48kHz stereo waveform)** into separate OS processes.
2. **5Hz Prefix Continuation + Bit-Exact Splice (replaces broken `repaint`)**: In ACE-Step 1.5, `task_type="repaint"` belongs to `DIRECT_CONDITIONING_TASKS` and **disables the 5Hz LM (`thinking=False`)**. Extending a 45s vocal demo to 2m20s via `repaint` either repeats Verse 1 or rushes all new lyrics into 25s followed by empty accompaniment. Instead, this skill:
   - Scans the 45s demo's `0.5s` RMS envelope to pick a full-energy cut point before tail decay;
   - Feeds the exact prefix `<|audio_code_*|>` tokens into **both `cond` and `uncond`** prompts of `acestep-5Hz-lm-1.7B` (`FSMState.CODES_GENERATION`) to autoregressively plan the remaining song;
   - Renders with `reference_audio` locked to the 45s demo, phase-aligns (`±40ms` cross-correlation) with an equal-power cosine crossfade, and **restores `0 .. crossfade_start_sec` bit-for-bit (`max_diff = 0.0`)** from the approved demo.
3. **Offline Sung-Lyrics Verification & A+B Copyright Isolation**: Uses local sliding-window ASR (`sherpa-onnx` FireRed CTC + optional `faster-whisper`) to catch skipped lines or synth-intro hallucinations before `.lrc` packaging, and supports **A+B dual-track output** so exclusive streaming singles never collide with multi-platform video BGM.

---

## Hard Rules

1. **Never load 5Hz LM and DiT in the same Python process.** Always run the `lm` command first, let the process exit, then run `dit`.
2. **Run `doctor.py` before first execution** and never install packages, drivers, FFmpeg, or multi-GB model weights without explicit user approval.
3. **Agents cannot hear audio.** RMS envelopes, LUFS/dBTP metrics, and ASR read-back are technical gates only — never claim a melody "sounds catchy" or "mixes well" without human listening. Pause after 45s demos for the user to pick a take.
4. **Original lyrics & AIGC transparency**: Write original lyrics only (never copy copyrighted lyrics or imitate named artists). Declare AI-generated music/cover on streaming and social platforms as required (e.g. China's AIGC Labeling Rules effective 2025-09-01).

---

## Step-by-Step Workflow

Resolve `{SKILL_DIR}` to the directory containing this `SKILL.md`.

### Step 0: Environment Preflight (Read-Only)

```bash
python "{SKILL_DIR}/scripts/doctor.py" --json
```

- Inspect `ready`, `acestep_python`, `acestep_root`, `gpu`, and `next_steps`.
- All inference scripts (`generate_demos.py`, `expand_song.py`) must be run with **`acestep_python`** (the Python inside ACE-Step-1.5's `.venv`), while `doctor.py` and `package_release.py` work with any Python 3.10+.
- Set `PYTHONUTF8=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` once checkpoints are downloaded so HuggingFace probes never hang offline or garble CJK output on Windows.

### Step 1: Draft & Render 2–3 Differentiated 45s Hook Demos

Read [references/songwriting-and-prompts.md](references/songwriting-and-prompts.md) before writing specs. Create a spec JSON modeled on [examples/demos-zh.json](examples/demos-zh.json) or [examples/demos-en.json](examples/demos-en.json):

- Give 2–3 distinct genre/tempo/vocal directions (45s = 225 5Hz codes, enough for `Chorus → Verse → Chorus` or `Verse → Chorus × 2`).
- Run LM stage, then DiT stage in separate processes:

```bash
<acestep_python> -B "{SKILL_DIR}/scripts/generate_demos.py" lm --spec demos.json --out-dir output/demos
<acestep_python> -B "{SKILL_DIR}/scripts/generate_demos.py" dit --spec demos.json --out-dir output/demos
```

### Step 2: Screen Envelopes + Local ASR, Then Let the User Pick

1. Check `report.json` in each demo folder: `first_active_sec` should be `0.0–1.0s` for short-video hooks, and `last_active_sec` shows where the tail starts decaying.
2. (Optional, if local ASR is available per `doctor.py`) Run sliding-window read-back to catch silent club-drop bugs:

```bash
<asr_python> -B "{SKILL_DIR}/scripts/check_lyrics_asr.py" --spec demos.json --demos-dir output/demos
```

3. Present the MP3/WAV paths and direction summary to the user. **Stop and wait for the user to listen and choose their favorite demo** (plus whether they want A-track streaming single only, B-track instrumental BGM only, or A+B dual output).

### Step 3: Lossless Expansion to Full Single (+ Optional B-Track BGM)

1. **Inspect the chosen demo's RMS envelope** to pick a clean cut window before premature fadeout:

```bash
<acestep_python> -B "{SKILL_DIR}/scripts/generate_demos.py" inspect-envelope --wav output/demos/<demo_id>/<demo_id>.wav
```

   Pick `crossfade_start_sec` / `crossfade_end_sec` inside a sustained chorus tail or instrumental bar (typically `38.5s–43.6s` on a 45s demo), and set `prefix_codes_count = round(crossfade_end_sec * 5)`.

2. **Write the expansion spec JSON** modeled on [examples/expand-full.example.json](examples/expand-full.example.json):
   - Keep `bpm`, `keyscale`, `timesignature`, and `vocal_language` identical to the base demo.
   - Provide `candidate_seeds` (3 seeds recommended) so the user has takes to compare.
   - If the song will sign an **exclusive streaming agreement** (e.g. Qishui Music Exclusive) AND the user also wants background music for multi-platform videos (Douyin + Xiaohongshu / YouTube), include `bgm_b_track` with a **fresh seed, shifted key/BPM, `instrumental: true`, and zero shared `audio_codes` or `reference_audio`** (see [references/streaming-release-and-copyright.md](references/streaming-release-and-copyright.md)).

3. **Run split-process continuation and bit-exact stitching**:

```bash
<acestep_python> -B "{SKILL_DIR}/scripts/expand_song.py" lm-continue --spec expand.json --out-dir output/full
<acestep_python> -B "{SKILL_DIR}/scripts/expand_song.py" dit-stitch --spec expand.json --out-dir output/full
```

   Verify in `rendered_candidates/seed_<N>/summary.json` that `prefix_bit_exact_max_diff` is `0.0`.

4. **Master the candidate(s)** to streaming loudness (`-12.8 LUFS`, `-1.0 dBTP`, `48kHz / 24-bit WAV` + `320kbps MP3`, tail `< -42 dBFS` trimmed):

```bash
<acestep_python> -B "{SKILL_DIR}/scripts/expand_song.py" master \
  --src-wav output/full/rendered_candidates/seed_<SEED>/stitched_seed_<SEED>.wav \
  --out-dir output/full/final \
  --out-stem takeA-s<SEED> \
  --fade-start 119.2 \
  --total-end 123.2
```

### Step 4: Verify Sung Lyrics & Package Streaming Release

Read [references/streaming-release-and-copyright.md](references/streaming-release-and-copyright.md) for streaming QC gates and cover art rules.

1. **Verify actual sung lines** (5Hz LM occasionally skips the last 1–2 Bridge lines or plays a pure synth intro before singing line 1):

```bash
<asr_python> -B "{SKILL_DIR}/scripts/check_lyrics_asr.py" \
  --wav output/full/final/takeA-s<SEED>.wav \
  --lyrics-file expected_lyrics.txt \
  --out-json output/full/asr_check.json
```

   Build `synced-lyrics.json` (see [examples/synced-lyrics.example.json](examples/synced-lyrics.example.json)) matching **100% what is actually sung** — omit any unsung intro/outro lines and strip all `[Verse]` / `[Chorus]` tags.

2. **Build the final release folder** (validates sample rate, silence gaps, monotonic LRC timestamps, and optional `3000×3000` sRGB cover image):

```bash
python "{SKILL_DIR}/scripts/package_release.py" \
  --wav output/full/final/takeA-s<SEED>.wav \
  --mp3 output/full/final/takeA-s<SEED>.mp3 \
  --synced-json synced-lyrics.json \
  --title "Song Title" \
  --artist "Artist Name" \
  --out-dir output/full/release
```

---

## Pitfalls Log (Battle-Tested)

- **Windows 16GB `0xC0000005` access violation**: Triggered whenever `LLMHandler` (1.7B) and `AceStepHandler` (DiT) initialize in one process, even if one targets CPU. Split-process `lm` → `dit` eliminates it 100%.
- **Premature silence at end of 45s demo**: A nominal `45.0s` demo often fades to `-57 dBFS` by `41.1s`. Splicing from `45.0s` leaves a 3.6s dead-air gap that fails streaming QC (`>3s` silence). Always run `inspect-envelope` and cut inside active sustain (`rms > 0.02`).
- **English electronic `Minimal Tech-House` + `groove bed` caption trap**: Combining those phrases causes 5Hz LM to sing one 14s opening hook and turn the remaining 31s into an instrumental club drop. Write `Catchy Melodic House Pop ... singing clearly upfront through opening chorus, full verse and final chorus, no long instrumental drop`.
- **Full-song CTC omission vs. Whisper intro hallucination**: Feeding a 140s song into CTC ASR in one chunk drops words — always decode in `≤18s` overlapping windows with `0.3s` lead-in zero padding. Conversely, `faster-whisper` excels at English word endings (`light / clear / disappears`) but hallucinates `Thanks for watching!` on loud brassy synth intros; cross-check with CTC `<sil>` to drop unsung intro lines.
- **Cover art rejection on music platforms**: Streaming covers (`3000×3000` sRGB, `≤10 MB`) are **not** video thumbnails — printing `AI生成`, `ACE-Step`, `2026`, `Demo`, lyrics, or social handles on the cover triggers automatic rejection. Check the AIGC box in the platform upload form instead.
