from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _studio_core import (  # noqa: E402
    check_silence_gates,
    format_lrc_time,
    inspect_cover_image,
    load_wav_audio,
    recommend_cut_window,
    stitch_equal_power_crossfade,
    validate_synced_lyrics,
)


def make_minimal_jpeg(width: int, height: int, components: int) -> bytes:
    sof0_payload = struct.pack(">BHHB", 8, height, width, components) + b"".join(
        struct.pack(">BBB", idx + 1, 0x11, 0) for idx in range(components)
    )
    sof0_seg = b"\xff\xc0" + struct.pack(">H", len(sof0_payload) + 2) + sof0_payload
    return b"\xff\xd8" + sof0_seg + b"\xff\xd9"


class StudioCoreTests(unittest.TestCase):
    def test_doctor_json_runs_cleanly(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-B", str(SCRIPTS_DIR / "doctor.py"), "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        report = json.loads(proc.stdout)
        self.assertIn("ready", report)
        self.assertIn("checks", report)
        check_ids = {c["id"] for c in report["checks"]}
        self.assertIn("python", check_ids)
        self.assertIn("acestep_root", check_ids)

    def test_recommend_cut_window_avoids_premature_tail_silence(self) -> None:
        sr = 48000
        t = np.arange(sr * 45, dtype=np.float32) / sr
        tone = 0.22 * np.sin(2.0 * math.pi * 220.0 * t)
        # Simulate real 45s demo decaying to silence at 41.0s..45.0s
        tone[int(41.0 * sr) :] *= 0.002
        stereo = np.stack([tone, tone], axis=1)

        rec = recommend_cut_window(stereo, sr)
        self.assertLessEqual(rec["recommended_crossfade_end_sec"], 41.0)
        self.assertGreaterEqual(rec["recommended_crossfade_start_sec"], 39.5)
        self.assertEqual(
            rec["recommended_prefix_codes_count"],
            int(round(rec["recommended_crossfade_end_sec"] * 5.0)),
        )

    def test_stitch_preserves_prefix_bit_for_bit_and_aligns_phase(self) -> None:
        sr = 48000
        dur_orig = 10.0
        dur_coded = 20.0
        t_orig = np.arange(int(dur_orig * sr), dtype=np.float32) / sr
        t_coded = np.arange(int(dur_coded * sr), dtype=np.float32) / sr

        orig_sig = 0.25 * np.sin(2.0 * math.pi * 110.0 * t_orig)
        # Shift coded signal by +240 samples (+5ms) and scale amplitude by 0.8x
        lag_samples = 240
        coded_sig = 0.20 * np.sin(2.0 * math.pi * 110.0 * (t_coded - lag_samples / sr))

        orig = np.stack([orig_sig, orig_sig], axis=1)
        coded = np.stack([coded_sig, coded_sig], axis=1)

        stitched, metrics = stitch_equal_power_crossfade(
            orig_audio=orig,
            coded_audio=coded,
            sr=sr,
            xf_start_sec=8.0,
            xf_end_sec=8.8,
        )
        self.assertEqual(metrics["prefix_bit_exact_max_diff"], 0.0)
        np.testing.assert_array_equal(stitched[: int(8.0 * sr)], orig[: int(8.0 * sr)])
        self.assertAlmostEqual(metrics["best_lag_ms"], 5.0, delta=1.0)

    def test_lrc_formatting_and_validation(self) -> None:
        self.assertEqual(format_lrc_time(6.92), "[00:06.92]")
        self.assertEqual(format_lrc_time(123.05), "[02:03.05]")

        valid_lines = [
            {"t": 6.92, "line": "Silver circuits start to glow"},
            {"t": 10.96, "line": "Leave the heavy doubts behind"},
        ]
        self.assertEqual(validate_synced_lyrics(valid_lines, duration_sec=120.0), [])

        bad_lines = [
            {"t": 5.0, "line": "[Chorus]"},
            {"t": 3.0, "line": "Out of order timestamp"},
            {"t": 130.0, "line": "Past track end"},
        ]
        errs = validate_synced_lyrics(bad_lines, duration_sec=120.0)
        self.assertEqual(len(errs), 3)

    def test_cover_header_inspector_rgb_vs_cmyk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            ok_jpg = p / "ok.jpg"
            ok_jpg.write_bytes(make_minimal_jpeg(3000, 3000, 3))
            rep_ok = inspect_cover_image(ok_jpg)
            self.assertTrue(rep_ok["ok"], rep_ok)

            cmyk_jpg = p / "cmyk.jpg"
            cmyk_jpg.write_bytes(make_minimal_jpeg(3000, 3000, 4))
            rep_cmyk = inspect_cover_image(cmyk_jpg)
            self.assertFalse(rep_cmyk["ok"])
            self.assertTrue(any("CMYK" in e for e in rep_cmyk["errors"]))

            small_jpg = p / "small.jpg"
            small_jpg.write_bytes(make_minimal_jpeg(1080, 1080, 3))
            rep_small = inspect_cover_image(small_jpg)
            self.assertFalse(rep_small["ok"])

    def test_24bit_pcm_wav_fallback_and_package_release_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            sr = 48000
            secs = 6
            t = np.arange(sr * secs, dtype=np.float32) / sr
            sig = 0.25 * np.sin(2.0 * math.pi * 220.0 * t)
            pcm24 = np.clip(np.rint(sig * 8388607.0), -8388608, 8388607).astype(np.int32)
            stereo = np.stack([pcm24, pcm24], axis=1).reshape(-1)
            b0 = (stereo & 0xFF).astype(np.uint8)
            b1 = ((stereo >> 8) & 0xFF).astype(np.uint8)
            b2 = ((stereo >> 16) & 0xFF).astype(np.uint8)
            packed = np.stack([b0, b1, b2], axis=1).tobytes()

            wav_path = p / "master.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(3)
                wf.setframerate(sr)
                wf.writeframes(packed)

            loaded, loaded_sr = load_wav_audio(wav_path)
            self.assertEqual(loaded_sr, 48000)
            self.assertEqual(loaded.shape, (sr * secs, 2))
            gates = check_silence_gates(loaded, loaded_sr)
            self.assertTrue(gates["passes_internal_3s_gate"])
            self.assertTrue(gates["passes_trailing_1p5s_gate"])

            synced_path = p / "synced.json"
            synced_path.write_text(
                json.dumps({"lines": [{"t": 0.8, "line": "Opening line"}, {"t": 3.2, "line": "Second line"}]}),
                encoding="utf-8",
            )
            cover_path = p / "cover.jpg"
            cover_path.write_bytes(make_minimal_jpeg(3000, 3000, 3))
            rel_dir = p / "release"

            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPTS_DIR / "package_release.py"),
                    "--wav",
                    str(wav_path),
                    "--synced-json",
                    str(synced_path),
                    "--title",
                    "Test Song",
                    "--artist",
                    "Test Artist",
                    "--cover",
                    str(cover_path),
                    "--out-dir",
                    str(rel_dir),
                ],
                check=True,
                capture_output=True,
            )
            self.assertTrue((rel_dir / "Test Song.wav").exists())
            self.assertTrue((rel_dir / "Test Song.lrc").exists())
            self.assertTrue((rel_dir / "Test Song_lyrics.txt").exists())
            self.assertTrue((rel_dir / "release_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
