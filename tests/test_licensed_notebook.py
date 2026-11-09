"""Exercise notebook dataset wiring without downloading audio or loading ML libraries."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/basic_training_notebook.ipynb"
POSITIVE_GENERATION_CELL = "-SvGtCCM9akR"
AUGMENTATION_CELL = "XW3bmbI5-JAz"
NEGATIVE_FEATURES_CELL = "1pGuJDPyp3ax"
TRAINING_CONFIG_CELL = "Ii1A14GsGVQT"


class LicensedNotebookTests(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text())["cells"]

    def cell_source(self, cell_id):
        return next("".join(cell["source"]) for cell in self.cells
                    if cell.get("metadata", {}).get("id") == cell_id)

    def test_path_cell_uses_existing_colab_checkout(self):
        source = next("".join(c["source"]) for c in self.cells
                      if "# 路径和唤醒词名称集中设置" in "".join(c["source"]))
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            content = Path(directory).resolve()
            checkout = content / "micro-wake-word"
            licensed_data = checkout / "microwakeword/audio/licensed_data.py"
            licensed_data.parent.mkdir(parents=True)
            licensed_data.touch()
            namespace = {}
            source = source.replace('Path("/content/micro-wake-word")', f"Path({str(checkout)!r})")
            try:
                with patch("subprocess.run") as run:
                    exec(source, namespace)
                run.assert_not_called()
                self.assertEqual(namespace["repo_path"], checkout)
                self.assertEqual(namespace["preview_dir"], checkout / "generated_samples/hey_kamana/preview")
                self.assertEqual(namespace["sample_dir"], checkout / "generated_samples/hey_kamana/three_speeds_v1")
                self.assertEqual(namespace["positive_dir"].name, "positive_hey_kamana_three_speeds_v1")
                self.assertTrue(namespace["use_hey_hi_negatives"])
                self.assertTrue(namespace["train_dir"].name.endswith("_hey_hi"))
            finally:
                os.chdir(previous_cwd)

    def test_positive_generation_publishes_three_complete_speeds_and_reuses_them(self):
        source = self.cell_source(POSITIVE_GENERATION_CELL)
        scales = {"fast": "0.65", "normal": "1.0", "slow": "1.25"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = root / "three_speeds_v1"
            preview = root / "preview"
            preview.mkdir()
            (preview / "0.wav").touch()
            namespace = {"repo_path": root, "sample_dir": sample_dir, "Path": Path,
                         "target_word": "hˈeɪ kɑmˈɑnɑ"}

            def fake_piper(command, *, cwd, check):
                output = Path(command[command.index("--output-dir") + 1])
                self.assertEqual(cwd, root / "piper-sample-generator")
                self.assertTrue(check)
                self.assertEqual(command[2:5], ["piper_sample_generator", namespace["target_word"], "--phoneme-input"])
                self.assertEqual(command[command.index("--length-scales") + 1], scales[output.name])
                self.assertEqual(command[command.index("--max-samples") + 1], "1000")
                self.assertEqual(command[command.index("--batch-size") + 1], "100")
                self.assertFalse(sample_dir.exists())
                output.mkdir(parents=True)
                for i in range(1000):
                    (output / f"{i}.wav").touch()

            with patch("subprocess.run", side_effect=fake_piper) as run, redirect_stdout(io.StringIO()) as log:
                exec(source, namespace)
                exec(source, namespace)
                self.assertEqual(run.call_count, 3)
                self.assertEqual({path.parent.name for path in sample_dir.glob("*/*.wav")}, set(scales))
                self.assertEqual(len(list(sample_dir.glob("*/*.wav"))), 3000)
                self.assertTrue((preview / "0.wav").is_file())
                self.assertIn("共 3000 条", log.getvalue())
                with self.assertRaisesRegex(FileExistsError, "positive_dataset_name"):
                    exec(source, {**namespace, "target_word": "different"})
                (sample_dir / "fast/0.wav").unlink()
                with self.assertRaisesRegex(FileExistsError, "positive_dataset_name"):
                    exec(source, namespace)
                self.assertEqual(run.call_count, 3)

    def test_positive_generation_failure_leaves_no_published_or_staged_data(self):
        source = self.cell_source(POSITIVE_GENERATION_CELL)
        for failure in (subprocess.CalledProcessError, RuntimeError):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sample_dir = root / "three_speeds_v1"

                def fake_piper(command, **kwargs):
                    output = Path(command[command.index("--output-dir") + 1])
                    output.mkdir(parents=True)
                    (output / "0.wav").touch()
                    if failure is subprocess.CalledProcessError and output.name == "slow":
                        raise subprocess.CalledProcessError(1, command)

                with patch("subprocess.run", side_effect=fake_piper) as run, self.assertRaises(failure):
                    exec(source, {"repo_path": root, "sample_dir": sample_dir, "Path": Path, "target_word": "hey"})
                self.assertEqual(run.call_count, 3)
                self.assertEqual(list(root.iterdir()), [])

    def test_ambient_filter_keeps_untagged_flac_with_valid_duration(self):
        class UntaggedFlac(dict):
            info = types.SimpleNamespace(length=7.0)

        class AudioPaths(list):
            def cast_column(self, name, feature):
                return self

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "untagged.flac"
            audio.touch()
            modules = {
                "datasets": types.SimpleNamespace(
                    Dataset=types.SimpleNamespace(from_dict=lambda columns: AudioPaths(columns["audio"])),
                    Audio=lambda **kwargs: None,
                ),
                "mutagen": types.SimpleNamespace(File=lambda path: UntaggedFlac()),
                "numpy": types.SimpleNamespace(array=list),
                "microwakeword.audio.audio_utils": types.SimpleNamespace(remove_silence_webrtc=None),
            }
            with patch.dict("sys.modules", modules):
                clips_path = NOTEBOOK.parents[1] / "microwakeword/audio/clips.py"
                clips_type = runpy.run_path(str(clips_path))["Clips"]
                ambient = clips_type(directory, "*.flac", min_clip_duration_s=6)
            self.assertEqual(ambient.clips, [str(audio)])

    def test_negative_generation_and_training_config_use_new_sources(self):
        generated = []
        loaded = []

        class Clips:
            def __init__(self, **kwargs):
                loaded.append(kwargs)
                self.clips = ["fixture"]

        class SpectrogramGeneration:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def spectrogram_generator(self):
                yield "feature"

        def from_generator(**kwargs):
            self.assertEqual(list(kwargs["sample_generator"]), ["feature"])
            generated.append(kwargs["out_dir"])

        ragged = types.ModuleType("mmap_ninja.ragged")
        ragged.RaggedMmap = types.SimpleNamespace(from_generator=from_generator)
        yaml = types.ModuleType("yaml")
        yaml.dump = lambda config, file: file.write(json.dumps(config))
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                root = Path("licensed_data_v1/prepared")
                root.mkdir(parents=True)
                (root / "manifest.json").write_text('{"files": []}')
                namespace = {
                    "data_root": root,
                    "sample_dir": Path("generated_samples/hey_kamana/three_speeds_v1"),
                    "positive_dir": Path("licensed_features_v1/positive_hey_kamana_three_speeds_v1"),
                    "features_dir": Path("licensed_features_v1"),
                    "train_dir": Path("trained_models/hey_kamana_licensed_inputs"),
                    "config_path": Path("training_parameters_hey_kamana.yaml"),
                    "use_hey_hi_negatives": False,
                }
                modules = {
                    "microwakeword.audio.augmentation": types.SimpleNamespace(
                        Augmentation=lambda **kwargs: kwargs
                    ),
                    "microwakeword.audio.clips": types.SimpleNamespace(Clips=Clips),
                    "microwakeword.audio.spectrograms": types.SimpleNamespace(
                        SpectrogramGeneration=SpectrogramGeneration
                    ),
                    "mmap_ninja.ragged": ragged,
                    "yaml": yaml,
                }
                with patch.dict("sys.modules", modules):
                    exec(self.cell_source(AUGMENTATION_CELL), namespace)
                    exec(self.cell_source(NEGATIVE_FEATURES_CELL), namespace)
                    exec(self.cell_source(TRAINING_CONFIG_CELL), namespace)
                self.assertEqual(loaded[0]["input_directory"], str(namespace["sample_dir"]))
                self.assertEqual(loaded[0]["file_pattern"], "*/*.wav")
                self.assertEqual(set(namespace["augmenters"]), {"training", "validation", "testing"})
                for split, augmenter in namespace["augmenters"].items():
                    self.assertEqual(augmenter["background_paths"], [str(root / "noise" / split)])
                    self.assertEqual(augmenter["impulse_paths"], [str(root / "rirs")])
                self.assertEqual(len(generated), 10)
                for corpus in ("speech", "noise"):
                    for split in ("training", "validation", "testing"):
                        self.assertIn(f"licensed_features_v1/negative/{corpus}/{split}/negative_mmap", generated)
                ambient = [row for row in loaded if "min_clip_duration_s" in row]
                self.assertEqual(len(ambient), 4)
                self.assertTrue(all("repeat_clip_min_duration_s" not in row for row in ambient))
                config = namespace["config"]
                self.assertEqual(config["features"][0]["features_dir"], str(namespace["positive_dir"]))
                self.assertTrue(all(row["features_dir"].startswith("licensed_features_v1/")
                                    for row in config["features"]))
                self.assertEqual(sum(row["truth"] for row in config["features"]), 1)
                self.assertNotEqual(config["train_dir"], "trained_models/wakeword")
                lineage = json.loads(Path("licensed_features_v1/feature_sources.json").read_text())
                self.assertEqual(lineage["positive_augmentation"]["testing"]["background_pool"],
                                 "licensed_data_v1/prepared/noise/testing")
                self.assertTrue(lineage["pending"])
            finally:
                os.chdir(old_cwd)

    def test_changed_python_cells_compile_and_legacy_sources_are_absent(self):
        for cell_id in ("YJRG4Qvo9nXG", AUGMENTATION_CELL, "D7BHcY1mEGbK",
                        NEGATIVE_FEATURES_CELL, TRAINING_CONFIG_CELL):
            compile(self.cell_source(cell_id), f"cell-{cell_id}", "exec")
        source = "\n".join("".join(cell["source"]) for cell in self.cells)
        for legacy in ("agkphysics/AudioSet", "mchl914/fma_xsmall",
                       "MIT_environmental_impulse_responses", "negative_datasets/",
                       "datasets/kahrendt/microwakeword", "background_paths = ['fma_16k'"):
            self.assertNotIn(legacy, source)
        self.assertIn("augmenter=augmenters[split]", source)


if __name__ == "__main__":
    unittest.main()
