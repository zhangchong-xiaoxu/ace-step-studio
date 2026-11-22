# Songwriting, Captions, and Structure Guide for ACE-Step 1.5

Practical rules verified across Mandarin Pop/Electro/Funk/City-Pop and English Synthwave/House singles generated locally with `acestep-5Hz-lm-1.7B` + `acestep-v15-turbo`.

---

## 1. JSON Single Source of Truth

Always specify musical metadata explicitly instead of leaving `bpm` or `keyscale` vague when you plan to expand a 45s demo later:

| Field | Recommended Practice | Example |
|---|---|---|
| `bpm` | Integer matching genre pocket | `108` (City Pop), `118` (Synthwave), `122` (Nu-Disco), `126` (Electro-Pop) |
| `keyscale` | Explicit tonic + mode | `A minor`, `D minor`, `E minor`, `F major`, `D major` |
| `timesignature` | String beats per bar | `"4"` (4/4), `"3"` (3/4) |
| `vocal_language` | `"zh"` or `"en"` | `"zh"` |
| `duration` | `45` for hook demos (`225` 5Hz codes); `120–145` for full singles | `45` |

> **Do not** duplicate numeric BPM or key inside `caption` for short demos unless reinforcing a full-song continuation prompt; keep `caption` focused on genre, arrangement, vocal timbre, mix placement, and section flow.

---

## 2. Writing Captions That Sing Clearly

Write `caption` in **specific English production terms** covering 5 anchors:

1. **Genre + Groove**: e.g. `Upbeat catchy Mandarin Nu-Disco and Funk Pop anthem with instant hook from the very first beat, zero slow intro.`
2. **Lead Vocal Placement**: Always state gender, tone, articulation, and mix priority:
   - Mandarin: `Crisp bright solo female vocals with clear confident Mandarin articulation, keep the lead vocal clearly in front of the instruments, no spoken narration.`
   - English: `Bright soaring solo female vocals with crystal-clear English pronunciation and uplifting energy, keep the lead vocal upfront and articulate.`
3. **Core Rhythm & Harmony**: Name 4–5 non-competing instruments (e.g. `punchy electronic kick, tight sidechained synth bass, plucky lead synth hook, crisp hi-hats, uplifting supersaw chords`).
4. **Temporal Arc**: Match the exact tags in `lyrics` (e.g. `Starts immediately with an earworm chorus melody, flows into a groovy verse, and returns to a bigger final chorus`).

### Caption Pitfalls to Avoid

- **The "Groove Bed / Tech-House" Vocal Drop**: In English electronic demos (`vocal_language="en"`), combining `Minimal Tech-House` + `electronic groove bed` makes `acestep-5Hz-lm-1.7B` sing only the first 14s chorus and schedule a 31s instrumental drop. If you want full vocals, write `Melodic House Pop / Dance-Pop ... singing clearly upfront through opening chorus, full melodic verse and final chorus, no long instrumental drop`.
- **Conflicting Timbres**: Pairing `intimate whispered acoustic ballad` with `explosive stadium supersaw drop` without describing a transition degrades vocal intelligibility.

---

## 3. Lyric Rhythm & Rhyme Discipline

### Mandarin (`vocal_language: "zh"`)

- **Line length**: Keep lines within the same section uniform at **7–9 characters** (optionally split with a single space at the caesura, e.g. `别等风来 现在就出发`).
- **Open, resonant vowels**: End chorus lines on open or nasal finals (`a / ao / ang / ian / ou`) so high notes ring cleanly instead of closing the mouth early.
- **Avoid AI clichés**: Skip stacked hollow nouns (`代码与星辰大海 / 灵魂的共振`). Anchor lyrics in one concrete scene (midnight highway streetlights, a blinking terminal cursor at 2 AM, pulling open curtains for morning tea).

### English (`vocal_language: "en"`)

- **Line length**: **4–7 content words** (`6–9 syllables`) per line so the model aligns syllables cleanly to 4/4 bars without rushing consonant clusters.
- **Strong diphthong rhymes**: `-ai` (`sky / tonight / high / light / fly`), `-on/-one`, or `-ear` cut cleanly through dense synth mixes.

---

## 4. 45s Demo vs. 2m+ Full Single Structure

### 45-Second Catchy Demo (`225` 5Hz codes)

Aim for an immediate hook within `0–2s`:

```text
[Chorus]      (4 lines — immediate earworm hook)
[Verse]       (3–4 lines — rhythmic contrast)
[Chorus]      (3–4 lines — hook payoff)
```

or a 2-bar synth riff into `Verse (3 lines) → Chorus (4 lines) → Chorus (3 lines)`.

### Full Single Expansion (`600–720` 5Hz codes = `120–144s`)

When expanding a 45s demo whose `0..42s` already covers `Verse 1 + Chorus 1`:

```text
<Exact sung lyrics of the preserved 0..cut prefix>
[Instrumental]    (8–16s synth/guitar solo breather after Chorus 1)

[Verse 2]         (4 lines — develop the story)
[Pre-Chorus]      (optional 2–4 lines — harmonic lift)
[Chorus]          (4 lines — full hook)
[Bridge]          (2 lines — keep concise; 4+ Bridge lines often get truncated by 5Hz LM)
[Chorus]          (4 lines — climax chorus)
[Instrumental]    (clean 8–16s musical resolution outro)
```
