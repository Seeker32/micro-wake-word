"""Verify the audio-only notebook's ZIP joins both training recipes correctly."""

import json
import math
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch
import wave
import zipfile

from microwakeword.audio.hard_negatives import (
    SPLITS, evaluate_hard_negative_model, import_audio_zip, prepare_hard_negative_features,
)


REPO = Path(__file__).resolve().parents[1]


def write_wav(path, frequency, *, rate=22050, duration=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (round(10000 * math.sin(i * frequency * 2 * math.pi / rate))
               for i in range(round(duration * rate)))
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<h", value) for value in samples))


def make_audio(root):
    for index, split in enumerate(SPLITS):
        write_wav(root / split / "0.wav", 300 + index * 100)


def notebook_source(name, cell_id):
    notebook = json.loads((REPO / "notebooks" / name).read_text())
    return next("".join(c["source"]) for c in notebook["cells"]
                if c.get("metadata", {}).get("id") == cell_id)


class HardNegativeTests(unittest.TestCase):
    def test_zip_preserves_splits_and_rejects_invalid_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            make_audio(original)
            archive_path = root / "audio.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in original.rglob("*.wav"):
                    archive.write(path, path.relative_to(original))
            extracted = import_audio_zip(root / "raw", archive_path)
            for split in SPLITS:
                self.assertEqual((extracted / split / "0.wav").read_bytes(),
                                 (original / split / "0.wav").read_bytes())
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.wav", "not audio")
            with self.assertRaisesRegex(ValueError, "ZIP"):
                import_audio_zip(root / "invalid", archive_path)
            self.assertFalse((root / "escape.wav").exists())
            self.assertFalse((root / "invalid").exists())

    def test_missing_audio_duplicate_and_long_phrase_stop_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                prepare_hard_negative_features(root / "raw", root / "features")
            make_audio(root / "raw")
            target = root / "raw/testing/0.wav"
            original = target.read_bytes()
            target.write_bytes((root / "raw/training/0.wav").read_bytes())
            with self.assertRaisesRegex(ValueError, "重复"):
                prepare_hard_negative_features(root / "raw", root / "features")
            target.write_bytes(original)
            write_wav(target, 500, duration=2)
            with self.assertRaisesRegex(ValueError, "完整发音"):
                prepare_hard_negative_features(root / "raw", root / "features")
            self.assertFalse((root / "features").exists())

    def test_real_feature_export_cache_and_both_notebook_configs(self):
        from mmap_ninja.ragged import RaggedMmap
        from microwakeword.data import FeatureHandler
        for name in ("basic_training_notebook.ipynb", "basic_training_notebook_wsl.ipynb"):
            with self.subTest(notebook=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_audio(root / "raw")
                namespace = {
                    "hey_hi_audio_dir": root / "raw",
                    "hey_hi_audio_zip": root / "absent.zip",
                    "hey_hi_features_dir": root / "features/negative/hey_hi",
                    "hey_hi_context_frames": 204,
                    "features_dir": root / "features",
                    "positive_dir": root / "features/positive_hey_kamana",
                    "train_dir": root / "trained_models/hey_kamana_hey_hi",
                    "config_path": root / "parameters.yaml",
                }
                exec(notebook_source(name, "hey-hi-features"), namespace)
                lineage = namespace["hey_hi_lineage"]
                self.assertEqual(lineage["counts"], dict.fromkeys(SPLITS, 1))
                self.assertFalse(lineage["truth"])
                for split in SPLITS:
                    features = RaggedMmap(namespace["hey_hi_features_dir"] / split / "hey_hi_mmap")
                    self.assertEqual(len(features), 1)
                    self.assertEqual(features[0].shape[1], 40)
                    self.assertGreaterEqual(features[0].shape[0], 204)
                exec(notebook_source(name, "Ii1A14GsGVQT"), namespace)
                config = namespace["config"]
                self.assertEqual(len(config["features"]), 5)
                entry = config["features"][3]
                self.assertFalse(entry["truth"])
                self.assertEqual(entry["sampling_weight"], 2)
                self.assertEqual(entry["truncation_strategy"], "truncate_start")
                self.assertEqual(sum(row["truth"] for row in config["features"]), 1)
                config.update(features=[entry], stride=3)
                handler = FeatureHandler(config)
                data, labels, _ = handler.get_data("training", batch_size=4, features_length=204)
                self.assertEqual(data.shape, (4, 204, 40))
                self.assertTrue((labels == 0).all())
                # Matching cache can be reused; changed source audio cannot silently reuse it.
                exec(notebook_source(name, "hey-hi-features"), namespace)
                write_wav(root / "raw/training/0.wav", 350)
                with self.assertRaisesRegex(FileExistsError, "不一致"):
                    exec(notebook_source(name, "hey-hi-features"), namespace)

    def test_held_out_trigger_rate_resets_stream_state_per_clip(self):
        import numpy as np
        from mmap_ninja.ragged import RaggedMmap

        class FakeModel:
            instances = 0

            def __init__(self, path, stride):
                self.index = FakeModel.instances
                FakeModel.instances += 1

            def predict_spectrogram(self, track):
                return [0.0] * 25 + [0.99 if self.index == 0 else 0.2] * 10

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mmap_path = root / "testing/hey_hi_mmap"
            mmap_path.parent.mkdir()
            RaggedMmap.from_generator(mmap_path, (np.zeros((300, 40)) for _ in range(2)), batch_size=2)
            with patch.dict("sys.modules", {"microwakeword.inference": types.SimpleNamespace(Model=FakeModel)}):
                report = evaluate_hard_negative_model("fixture.tflite", root, cutoffs=(0.95,))
            self.assertEqual(FakeModel.instances, 2)
            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(report["results"][0]["false_trigger_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
