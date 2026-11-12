"""Check the ambient-only feature builds are reproducible and idempotent."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio import ambient_features
from microwakeword.audio.licensed_data import digest


HAVE_AUDIO = all(importlib.util.find_spec(name) is not None
                 for name in ("soundfile", "librosa", "pymicro_features"))


def write_audio(path, seconds, rate=16000):
    """A clear tone, so generated features are non-trivial and deterministic."""
    import soundfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.sin(np.arange(int(round(seconds * rate))) / 7.0) * 0.3
    soundfile.write(str(path), samples.astype(np.float32), rate)
    return path


def make_prepared(root, files, *, durations=False):
    """Write audio plus a prepared-audio manifest.

    `files` maps a path relative to `root` to its duration in seconds. Paths mirror
    the licensed layout (`noise/testing/x.wav`) or CHiME-6 (`audio/testing/x.wav`).
    """
    root = Path(root)
    records = []
    for relative, seconds in files.items():
        path = write_audio(root / relative, seconds)
        record = {"path": relative, "sha256": digest(path), "dataset": "Fixture",
                  "split": relative.split("/")[1]}
        if durations:
            record["duration_s"] = float(seconds)
        records.append(record)
    (root / "manifest.json").write_text(json.dumps({"files": records}))
    return root


def build(prepared, output, **options):
    """Run a build and return (manifest, stdout)."""
    captured = io.StringIO()
    with redirect_stdout(captured):
        manifest = ambient_features.build_ambient_set(prepared, output, **options)
    return manifest, captured.getvalue()


def data_bytes(folder):
    return (Path(folder) / "data.ninja").read_bytes()


@unittest.skipUnless(HAVE_AUDIO, "needs soundfile, librosa and pymicro_features")
class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = make_prepared(self.root / "prepared", {
            "noise/testing/b_second.wav": 7.0,
            "noise/testing/a_first.wav": 7.0,
            "noise/testing/short.wav": 1.0,
            "noise/validation/dev.wav": 7.0,
        })
        self.output = self.root / "features" / "negative" / "ambient"

    def test_builds_only_ambient_sets_with_sorted_clips(self):
        manifest, output = build(self.prepared, self.output, corpora=("noise",))
        self.assertTrue(manifest["ambient_only"])
        self.assertEqual(sorted(manifest["sets"]), ["testing_ambient", "validation_ambient"])
        testing_clips = [clip["path"] for clip in manifest["clips"]
                         if clip["mode"] == "testing_ambient"]
        # Sorted by path, and the 1 s clip is below the 6 s floor.
        self.assertEqual(testing_clips, ["noise/testing/a_first.wav",
                                         "noise/testing/b_second.wav"])
        folder = self.output / "testing_ambient" / "noise_mmap"
        self.assertEqual(len(RaggedMmap(str(folder))), 2)
        self.assertIn("Built ambient features", output)
        self.assertGreater(manifest["hours"]["testing_ambient"], 0.0)

    def test_track_count_and_hours_match_the_manifest(self):
        manifest, _ = build(self.prepared, self.output, corpora=("noise",))
        info = manifest["sets"]["testing_ambient"]["noise"]
        tracks = RaggedMmap(str(self.output / "testing_ambient" / "noise_mmap"))
        self.assertEqual(info["tracks"], len(tracks))
        frames = sum(len(tracks[index]) for index in range(len(tracks)))
        self.assertAlmostEqual(info["hours"], frames * 10 / 1000 / 3600, places=9)
        self.assertEqual(len(info["clip_hashes"]), info["tracks"])

    def test_features_come_from_the_audio(self):
        build(self.prepared, self.output, corpora=("noise",))
        track = RaggedMmap(str(self.output / "testing_ambient" / "noise_mmap"))[0]
        self.assertEqual(track.shape[1], 40)
        self.assertGreater(track.shape[0], 400)  # ~7 s at 10 ms steps
        self.assertGreater(float(np.abs(track).sum()), 0.0)

    def test_rebuild_is_byte_identical(self):
        first, _ = build(self.prepared, self.output, corpora=("noise",))
        second, _ = build(self.prepared, self.root / "other" / "ambient",
                          corpora=("noise",), rebuild=True)
        self.assertEqual(first["sets"], second["sets"])
        self.assertEqual(first["clips"], second["clips"])
        self.assertEqual(
            data_bytes(self.output / "testing_ambient" / "noise_mmap"),
            data_bytes(self.root / "other" / "ambient" / "testing_ambient" / "noise_mmap"))

    def test_second_build_reuses_the_verified_tracks(self):
        manifest, _ = build(self.prepared, self.output, corpora=("noise",))
        data = self.output / "testing_ambient" / "noise_mmap" / "data.ninja"
        before = data.stat().st_mtime_ns
        again, output = build(self.prepared, self.output, corpora=("noise",))
        self.assertEqual(again, manifest)
        self.assertIn("Reusing verified ambient features", output)
        self.assertEqual(data.stat().st_mtime_ns, before)  # nothing was rewritten

    def test_verify_false_skips_hashing_but_still_reuses(self):
        build(self.prepared, self.output, corpora=("noise",))
        again, output = build(self.prepared, self.output, corpora=("noise",), verify=False)
        self.assertIn("Reusing verified ambient features", output)
        self.assertEqual(again["parameters"]["step_ms"], 10)

    def test_corrupted_track_forces_a_rebuild(self):
        manifest, _ = build(self.prepared, self.output, corpora=("noise",))
        folder = self.output / "testing_ambient" / "noise_mmap"
        data = folder / "data.ninja"
        data.write_bytes(b"\0" * data.stat().st_size)  # silent corruption
        again, output = build(self.prepared, self.output, corpora=("noise",))
        self.assertIn("no longer match their manifest", output)
        self.assertEqual(again["sets"], manifest["sets"])
        tracks = RaggedMmap(str(folder))
        for index, expected in enumerate(again["sets"]["testing_ambient"]["noise"]["clip_hashes"]):
            self.assertEqual(ambient_features.track_sha256(tracks[index]), expected)

    def test_parameter_change_rebuilds_with_a_new_selection(self):
        build(self.prepared, self.output, corpora=("noise",))
        manifest, output = build(self.prepared, self.output, corpora=("noise",),
                                 min_duration_s=0.0)
        self.assertIn("came from other data or parameters", output)
        paths = [clip["path"] for clip in manifest["clips"]
                 if clip["mode"] == "testing_ambient"]
        self.assertIn("noise/testing/short.wav", paths)

    def test_min_duration_is_strictly_longer(self):
        boundary = make_prepared(self.root / "boundary", {
            "noise/testing/exact.wav": 6.0,
            "noise/testing/longer.wav": 6.2,
        }, durations=True)
        manifest, _ = build(boundary, self.root / "boundary-features",
                           corpora=("noise",), splits=(("testing", "testing_ambient"),))
        self.assertEqual([clip["path"] for clip in manifest["clips"]],
                         ["noise/testing/longer.wav"])

    def test_new_source_audio_rebuilds(self):
        build(self.prepared, self.output, corpora=("noise",))
        write_audio(self.prepared / "noise/testing/c_third.wav", 7.0)
        records = json.loads((self.prepared / "manifest.json").read_text())["files"]
        records.append({"path": "noise/testing/c_third.wav", "split": "testing",
                        "sha256": digest(self.prepared / "noise/testing/c_third.wav"),
                        "dataset": "Fixture"})
        (self.prepared / "manifest.json").write_text(json.dumps({"files": records}))
        manifest, output = build(self.prepared, self.output, corpora=("noise",))
        self.assertIn("came from other data or parameters", output)
        self.assertEqual(len(manifest["sets"]["testing_ambient"]["noise"]["clip_hashes"]), 3)

    def test_chime6_style_manifest_uses_recorded_durations(self):
        prepared = make_prepared(self.root / "chime6", {
            "audio/testing/seg1.wav": 7.0,
            "audio/validation/seg0.wav": 7.0,
        }, durations=True)
        manifest, _ = build(prepared, self.root / "chime6-features", corpora=("chime6",))
        self.assertEqual(manifest["parameters"]["min_duration_s"], {"chime6": 0.0})
        self.assertEqual([clip["split"] for clip in manifest["clips"]],
                         ["validation", "testing"])
        info = manifest["sets"]["testing_ambient"]["chime6"]
        self.assertEqual(info["tracks"], 1)
        self.assertEqual(len(RaggedMmap(str(self.root / "chime6-features" /
                                           "testing_ambient" / "chime6_mmap"))), 1)

    def test_no_partial_directories_are_left_behind(self):
        build(self.prepared, self.output, corpora=("noise",))
        build(self.prepared, self.output, corpora=("noise",), rebuild=True)
        parent = self.output.parent
        self.assertFalse([path for path in parent.iterdir()
                          if path.name.startswith("ambient-build-") or path.suffix == ".stale"])

    def test_splits_select_which_sets_are_built(self):
        manifest, _ = build(self.prepared, self.output, corpora=("noise",),
                            splits=(("testing", "testing_ambient"),))
        self.assertEqual(list(manifest["sets"]), ["testing_ambient"])
        self.assertFalse((self.output / "validation_ambient").exists())

    def test_unknown_corpus_is_rejected(self):
        with self.assertRaises(ValueError):
            build(self.prepared, self.output, corpora=("speechless",))

    def test_other_step_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            build(self.prepared, self.output, corpora=("noise",), step_ms=20)

    def test_missing_manifest_is_reported(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError) as caught:
            build(empty, self.output, corpora=("noise",))
        self.assertIn("manifest.json", str(caught.exception))

    def test_missing_split_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            build(self.prepared, self.output, corpora=("noise",),
                  splits=(("training", "training_ambient"),))
        self.assertIn("training", str(caught.exception))


class BuildAllTests(unittest.TestCase):
    def test_requires_prepared_paths_for_the_requested_corpora(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ambient_features.build_all(directory, corpora=("noise",))
            with self.assertRaises(ValueError):
                ambient_features.build_all(directory, corpora=("chime6",))

    def test_unknown_corpus_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ambient_features.build_all(directory, corpora=("radio",))


if __name__ == "__main__":
    unittest.main()
