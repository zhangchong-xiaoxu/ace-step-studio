# Streaming Mastering, Cover Specs, and Dual-Track Copyright Isolation

Verified against Qishui Music / Douyin Music (`music.douyin.com`) automated and human review gates (first single approved and live same-day) and cross-platform video publishing workflows.

---

## 1. Audio Mastering & Silence Gates

Music streaming platforms run automated QC before human review. Three audio defects trigger immediate rejection:

1. **Integrated loudness out of range**: Target **`-13.2` to `-12.0 LUFS`** (streaming platforms typically accept `-14` to `-9 LUFS`) with **True Peak `≤ -1.0 dBTP`**.
2. **Mid-song or trailing dead air (`> 3.0s` silence)**: `expand_song.py master` scans `0.5s` windows and flags any `< -42 dBFS` window. Keep the musical fade-out smooth over `3.5–5.0s` (`afade=curve=qsin`) and trim `total_end` so trailing `< -42 dBFS` tail is **`≤ 1.5s`**.
3. **Sample format**: Export **`48kHz / 24-bit stereo PCM WAV`** as the primary master (plus `320kbps MP3` for preview/backup). Name final files cleanly with the exact track title (`<Song Title>.wav`), stripping `_takeA`, `_v1`, or `_seed` suffixes.

---

## 2. Sung-Lyrics Alignment (`.lrc` & Plain Lyrics)

Streaming review checks lyrics line-by-line against the vocal track:

- **100% match what is actually sung**:
  - If the 5Hz LM plays a 6.8s synth riff at the start and skips line 1 of your prompt, **delete line 1** from both `.lrc` and plain `.txt`.
  - If `[Bridge]` only sings 2 of 4 drafted lines before jumping into the final `[Chorus]`, trim the unsung lines from the lyrics files.
- **Zero structure tags in deliverables**: Never leave `[Verse]`, `[Chorus]`, `[Bridge]`, or `[Instrumental]` inside `.lrc` or the plain-text lyrics uploaded to streaming platforms.
- **Dual-ASR verification for English tracks**: `faster-whisper` captures English final consonants accurately, whereas `sherpa-onnx` FireRed CTC reliably outputs `<sil>` on pure instrumental intros where Whisper may hallucinate `Thanks for watching!`. Use both when available.

---

## 3. Single Cover Art Rules (`3000 × 3000` sRGB)

Music streaming cover rules are **much stricter** than YouTube/Douyin video thumbnails:

- **Technical spec**: `3000 × 3000 px` (1:1), **`sRGB` color space** (`RGB` 3-channel; `CMYK` is rejected), file size **`≤ 10 MB`** (high-grain 3000px PNGs often exceed 10 MB — use `JPEG quality=95..97, 4:4:4 subsampling` at ~2–4 MB).
- **Text red lines**:
  - Allowed: **Pure artwork with zero text**, OR **only the exact Song Title** (and exact Artist Name) matching the upload metadata character-for-character.
  - **Forbidden on artwork**: `AI生成`, `AIGC`, `ACE-Step`, `2026`, `Demo`, `Single`, lyric sentences, QR codes, URLs, or social handles. Declare AIGC via the platform's metadata checkbox instead.
- **Built-in platform AI cover tip (Qishui / Douyin Music)**: `music.douyin.com` provides a free built-in "Smart Cover" (智能封面) generator accepting **`≤ 50` Chinese characters** of comma-separated visual keywords plus compliant title typography templates. Providing 3–4 differentiated `≤ 45-char` visual prompts (subject + lighting + palette + medium/film texture) is faster and cleaner than procedural geometry scripts.

---

## 4. A+B Dual-Track Strategy (Streaming Exclusive vs. Multi-Platform Video BGM)

When a creator wants **both** streaming royalties/platform music-library integration **and** background music baked into videos posted across multiple platforms (e.g. Douyin + Xiaohongshu + Bilibili + YouTube):

| Track | Rights & Distribution | Audio Spec & Usage |
|---|---|---|
| **Track A (Vocal Single)** | Can sign **Streaming Exclusive** (e.g. Qishui Music Exclusive → auto-syncs to Douyin official music library) | `-13.2 ~ -12.0 LUFS / -1.0 dBTP`. Attach via native Douyin music picker on Douyin-only posts; **do NOT hard-bake into `.mp4` files uploaded to competing platforms** during the exclusive window. |
| **Track B (Video BGM & Stinger)** | **Keep local / non-exclusive** (100% self-owned cross-platform embedding rights) | Generate with a **completely fresh `seed`, shifted key & BPM** (e.g. Track A `D minor / 118 BPM` → Track B `A minor / 116 BPM`), `instrumental=True`, and **zero shared `audio_codes` or `reference_audio`**. Master a `60s` bed at `-16.0 LUFS` (duck to `volume=0.18..0.24` under voiceover) and a `6s` stinger at `-14.0 LUFS`. Verify `0..60s` has zero vocal hallucinations via ASR. |
