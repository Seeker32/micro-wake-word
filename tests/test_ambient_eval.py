"""Check ambient evaluation without a TFLite model or a trained checkpoint."""

import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

import numpy as np
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio import ambient_eval


HAVE_TF = importlib.util.find_spec("tensorflow") is not None
HAVE_AUDIO = all(importlib.util.find_spec(name) is not None
                 for name in ("soundfile", "scipy", "pymicro_features"))


def write_mmap(folder, lengths, value=0.0):
    """Write ragged tracks of `lengths` slices, 40 bins each, filled with `value`."""
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(
        str(folder),
        (np.full((length, 40), value, dtype=np.float32) for length in lengths),
        batch_size=1, verbose=False)
    return folder


def write_wav(path, seconds=1.0, rate=16000, value=0.25, channels=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(round(seconds * rate))
    payload = np.full((samples, channels), value, dtype=np.float32)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((payload * 32767).astype("<i2").tobytes())
    return path


def stub_player(stream=None, length=100, value=0.0):
    """A model stand-in: every spectrogram yields the same probability stream."""
    def player(_spectrogram):
        return np.full(length, value) if stream is None else np.asarray(stream)
    return player


class FindAmbientSetsTests(unittest.TestCase):
    def test_labels_pools_and_skips_other_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mmap(root / "negative/ambient/testing_ambient/noise_mmap", [10])
            write_mmap(root / "negative/ambient/testing_ambient/speech_mmap", [10])
            write_mmap(root / "negative/ambient_chime6/testing_ambient/chime6_mmap", [10])
            write_mmap(root / "negative/ambient/validation_ambient/noise_mmap", [10])
            sources = ambient_eval.find_ambient_sets(root / "negative/ambient",
                                                    root / "negative/ambient_chime6")
        self.assertEqual(sorted(sources), ["chime6", "noise", "speech"])
        self.assertEqual(sources["noise"],
                         [root / "negative/ambient/testing_ambient/noise_mmap"])

    def test_repeated_labels_keep_their_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mmap(root / "fsd/testing_ambient/noise_mmap", [10])
            write_mmap(root / "chime6/testing_ambient/noise_mmap", [10])
            sources = ambient_eval.find_ambient_sets(root)
        self.assertEqual(sorted(sources), ["chime6_noise", "fsd_noise"])

    def test_missing_sets_name_the_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError) as caught:
                ambient_eval.find_ambient_sets(Path(directory) / "nowhere")
        self.assertIn("testing_ambient", str(caught.exception))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ambient_eval.find_ambient_sets(".", mode="training")

    def test_validation_mode_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mmap(root / "ambient/validation_ambient/noise_mmap", [10])
            sources = ambient_eval.find_ambient_sets(root, mode="validation_ambient")
        self.assertEqual(sorted(sources), ["noise"])


class AudioFileTests(unittest.TestCase):
    def test_find_audio_files_walks_directories_and_takes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_wav(root / "a/one.wav")
            second = write_wav(root / "b/two.wav")
            (root / "notes.txt").write_text("not audio")
            self.assertEqual(ambient_eval.find_audio_files(root), [first, second])
            self.assertEqual(ambient_eval.find_audio_files(second), [second])

    def test_find_audio_files_names_missing_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError) as caught:
                ambient_eval.find_audio_files(Path(directory) / "empty")
        self.assertIn(".wav", str(caught.exception))

    @unittest.skipUnless(HAVE_AUDIO, "needs soundfile, scipy and pymicro_features")
    def test_audio_tracks_report_hours_and_features(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [write_wav(Path(directory) / "a.wav", seconds=2.0),
                     write_wav(Path(directory) / "b.wav", seconds=1.0)]
            tracks = ambient_eval.AudioTracks(paths)
            features, hours = tracks[0], tracks.hours
        self.assertEqual(len(tracks), 2)
        self.assertAlmostEqual(hours, 3.0 / 3600.0, places=9)
        self.assertEqual(features.shape[1], 40)
        self.assertEqual(features.dtype, np.float32)
        self.assertGreater(features.shape[0], 100)

    @unittest.skipUnless(HAVE_AUDIO, "needs soundfile, scipy and pymicro_features")
    def test_stereo_is_mixed_and_resampled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_wav(Path(directory) / "stereo.wav", seconds=1.0,
                             rate=48000, channels=2)
            samples = ambient_eval.read_audio(path)
        self.assertEqual(samples.ndim, 1)
        self.assertEqual(len(samples), 16000)
        self.assertEqual(samples.dtype, np.float32)

    @unittest.skipUnless(HAVE_AUDIO, "needs soundfile, scipy and pymicro_features")
    def test_features_match_the_training_pipeline(self):
        from microwakeword.audio.audio_utils import generate_features_for_clip

        samples = (np.sin(np.arange(16000) / 7.0) * 8000).astype(np.int16)
        np.testing.assert_allclose(ambient_eval.features_for_audio(samples),
                                   generate_features_for_clip(samples))

    def test_other_step_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            ambient_eval.features_for_audio(np.zeros(4000), step_ms=20)


class MovingAverageTests(unittest.TestCase):
    def test_averages_each_window(self):
        averaged = ambient_eval.moving_average(np.array([0.0, 1.0, 0.0, 1.0]), window=2)
        np.testing.assert_allclose(averaged, [0.5, 0.5, 0.5])

    def test_short_stream_is_rejected(self):
        with self.assertRaises(ValueError):
            ambient_eval.moving_average(np.zeros(3), window=5)

    def test_bad_window_is_rejected(self):
        with self.assertRaises(ValueError):
            ambient_eval.moving_average(np.zeros(9), window=0)


class EvaluateTracksTests(unittest.TestCase):
    def test_peaks_track_each_recording_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            tracks = write_mmap(Path(directory) / "testing_ambient/noise_mmap", [10, 20])
            calls = []

            def player(spectrogram):
                calls.append(len(calls))
                return np.full(len(spectrogram), 1.0 if len(calls) == 1 else 0.0)

            result = ambient_eval.evaluate_tracks(tracks, player)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["track_count"], 2)
        self.assertEqual(result["peaks"], [1.0, 0.0])
        self.assertEqual([len(track) for track in result["probabilities"]], [6, 16])

    def test_skip_first_drops_padded_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            tracks = write_mmap(Path(directory) / "testing/wakeword_mmap", [30])
            stream = np.concatenate([np.zeros(25), np.ones(10)])
            result = ambient_eval.evaluate_tracks(tracks, stub_player(stream), skip_first=25)
        self.assertEqual(len(result["probabilities"][0]), 6)
        self.assertEqual(result["peaks"], [1.0])

    def test_short_track_names_the_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            tracks = write_mmap(Path(directory) / "testing/wakeword_mmap", [8])
            with self.assertRaises(ValueError) as caught:
                ambient_eval.evaluate_tracks(tracks, stub_player(np.zeros(8)), skip_first=25)
        self.assertIn("Track 0", str(caught.exception))

    def test_hours_follow_the_stride(self):
        probabilities = [np.zeros(3600 * 100 - 4)]
        hours = ambient_eval.track_hours(probabilities, stride=3, step_s=0.01)
        self.assertAlmostEqual(hours, (3600 * 100 - 4) * 3 * 0.01 / 3600, places=9)

    def test_max_tracks_limits_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = write_mmap(Path(directory) / "testing_ambient/noise_mmap", [10, 10, 10])
            tracks = ambient_eval.open_tracks(folder, max_tracks=2)
        self.assertEqual(len(tracks), 2)

    def test_summarize_audio_tracks_uses_file_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [write_wav(Path(directory) / "a.wav", seconds=2.0)]
            summary = ambient_eval.summarize_tracks(ambient_eval.AudioTracks(paths))
        self.assertEqual(summary["tracks"], 1)
        self.assertAlmostEqual(summary["hours"], 2.0 / 3600.0, places=9)


class FalseNegativeRateTests(unittest.TestCase):
    def test_rates_are_share_of_missed_clips(self):
        rates = ambient_eval.false_negative_rates([0.10, 0.60, 0.99], [0.0, 0.50, 0.98])
        np.testing.assert_allclose(rates, [0.0, 1 / 3, 2 / 3])

    def test_empty_positives_are_rejected(self):
        with self.assertRaises(ValueError):
            ambient_eval.false_negative_rates([], [0.5])


class FalseAcceptTests(unittest.TestCase):
    def test_false_accepts_are_counted_once_per_cooldown(self):
        # One accept followed by an immediate second trigger: cooldown counts one.
        probabilities = [np.concatenate([np.zeros(100), np.ones(10), np.zeros(100)])]
        rates = ambient_eval.false_accepts_per_hour(probabilities, np.array([0.5]),
                                                    stride=1, step_s=0.01)
        hours = len(probabilities[0]) * 0.01 / 3600
        self.assertEqual(rates[0], 1 / hours)

    def test_quiet_audio_has_no_false_accepts(self):
        rates = ambient_eval.false_accepts_per_hour([np.zeros(1000)], np.array([0.1, 0.5]))
        np.testing.assert_allclose(rates, [0.0, 0.0])

    def test_cooldown_is_per_track(self):
        # Two short tracks that each accept once must not share one cooldown.
        probabilities = [np.ones(30), np.ones(30)]
        rates = ambient_eval.false_accepts_per_hour(probabilities, np.array([0.5]),
                                                    stride=1, step_s=0.01)
        self.assertAlmostEqual(rates[0], 2 / (60 * 0.01 / 3600), places=6)

    def test_roc_area_is_finite(self):
        # Cutoffs rise, so both rates fall; the curve is read left to right in FAPH.
        faph = np.array([2.0, 1.0, 0.5, 0.0])
        frr = np.array([0.1, 0.2, 0.4, 1.0])
        curve = ambient_eval.roc_curve(faph, frr, np.array([0.7, 0.8, 0.9, 1.0]))
        self.assertTrue(math.isfinite(curve["auc"]))
        self.assertGreaterEqual(curve["auc"], 0.0)

    @unittest.skipUnless(HAVE_TF, "needs tensorflow to import the training implementations")
    def test_matches_the_training_definitions(self):
        from microwakeword.test import compute_false_accepts_per_hour, generate_roc_curve

        rng = np.random.default_rng(0)
        probabilities = [rng.beta(0.5, 9.0, 400) for _ in range(3)]
        probabilities[1][200:205] = 0.99
        cutoffs = np.arange(0, 1.01, 0.01)
        np.testing.assert_allclose(
            ambient_eval.false_accepts_per_hour(probabilities, cutoffs, stride=1),
            compute_false_accepts_per_hour(probabilities, cutoffs, 25, stride=1, step_s=0.01))

        peaks = rng.beta(7.0, 1.5, 40)
        frr = ambient_eval.false_negative_rates(peaks, cutoffs)
        rates = ambient_eval.false_accepts_per_hour(probabilities, cutoffs, stride=1)
        ours = ambient_eval.roc_curve(rates, frr, cutoffs)
        theirs = generate_roc_curve(rates, frr, cutoffs, max_faph=2.0)
        np.testing.assert_allclose(ours["faph"], theirs[0])
        np.testing.assert_allclose(ours["frr"], theirs[1])
        np.testing.assert_allclose(ours["cutoffs"], theirs[2])


class EvaluateAmbientTests(unittest.TestCase):
    def test_report_splits_sources_and_pools_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            noise = write_mmap(root / "ambient/testing_ambient/noise_mmap", [100, 100])
            speech = write_mmap(root / "ambient/testing_ambient/speech_mmap", [100])
            positives = write_mmap(root / "positive/testing/wakeword_mmap", [60])
            report = ambient_eval.evaluate_ambient(
                "model.tflite", {"noise": noise, "speech": speech}, cutoffs=[0.5, 0.99],
                stride=1, positives=positives,
                player=stub_player(np.concatenate([np.zeros(25), np.full(35, 0.9)])),
            )
        self.assertEqual(sorted(report["sources"]), ["noise", "speech"])
        self.assertEqual(report["sources"]["noise"]["track_count"], 2)
        self.assertEqual(report["sources"]["speech"]["hours"],
                         report["sources"]["noise"]["hours"] / 2)
        self.assertAlmostEqual(report["pooled"]["hours"],
                               report["sources"]["noise"]["hours"]
                               + report["sources"]["speech"]["hours"])
        # pooling must not change the total accept count per source
        for index in range(2):
            self.assertAlmostEqual(
                report["pooled"]["false_accepts"][index],
                sum(source["false_accepts"][index]
                    for source in report["sources"].values()))
        self.assertEqual(report["positives"]["count"], 1)
        self.assertEqual(report["positives"]["false_negative_rate"], [0.0, 1.0])
        self.assertEqual([row["cutoff"] for row in report["summary"]], [0.5, 0.99])
        self.assertEqual(report["roc"]["faph"][0] <= report["roc"]["faph"][-1], True)
        json.dumps(report)  # the notebook writes this report out as JSON

    def test_sources_accept_audio_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [write_wav(Path(directory) / "room.wav", seconds=2.0)]
            tracks = ambient_eval.AudioTracks(paths)
            report = ambient_eval.evaluate_ambient(
                "model.tflite", {"room": tracks}, cutoffs=[0.5], stride=1,
                player=stub_player(length=10),
                progress=None)
        self.assertEqual(report["sources"]["room"]["track_count"], 1)
        self.assertIn("1 audio files", report["sources"]["room"]["inputs"][0])

    def test_progress_reports_every_track(self):
        with tempfile.TemporaryDirectory() as directory:
            tracks = write_mmap(Path(directory) / "ambient/testing_ambient/noise_mmap",
                                [20, 20])
            seen = []
            ambient_eval.evaluate_ambient(
                "model.tflite", {"noise": tracks}, cutoffs=[0.5], stride=1,
                player=stub_player(length=20),
                progress=lambda label, done, total: seen.append((label, done)))
        self.assertEqual(seen, [("noise", 1), ("noise", 2)])

    def test_roc_area_is_a_fraction(self):
        faph = np.array([2.0, 1.0, 0.5, 0.0])
        frr = np.array([0.1, 0.2, 0.4, 1.0])
        curve = ambient_eval.roc_curve(faph, frr, np.array([0.7, 0.8, 0.9, 1.0]))
        self.assertTrue(math.isfinite(curve["auc"]))
        self.assertGreaterEqual(curve["auc"], 0.0)
        self.assertLessEqual(curve["auc"], 2.0)


if __name__ == "__main__":
    unittest.main()
