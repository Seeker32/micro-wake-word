"""Exercise notebook dataset wiring without downloading audio or loading ML libraries."""

import json
import os
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/basic_training_notebook.ipynb"
AUGMENTATION_CELL = "XW3bmbI5-JAz"
NEGATIVE_FEATURES_CELL = "1pGuJDPyp3ax"
TRAINING_CONFIG_CELL = "Ii1A14GsGVQT"


class LicensedNotebookTests(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text())["cells"]

    def cell_source(self, cell_id):
        return next("".join(cell["source"]) for cell in self.cells
                    if cell.get("metadata", {}).get("id") == cell_id)

    def test_install_cell_finds_colab_checkout(self):
        source = self.cell_source("BFf6511E65ff")
        source = re.sub(r"(?m)^([ \t]*)[!%].*$", r"\1pass", source)

        with tempfile.TemporaryDirectory() as directory:
            content = Path(directory)
            checkout = content / "micro-wake-word"
            licensed_data = checkout / "microwakeword/audio/licensed_data.py"
            licensed_data.parent.mkdir(parents=True)
            licensed_data.touch()
            namespace = {}
            with patch.object(Path, "cwd", return_value=content):
                exec(source, namespace)
            self.assertEqual(namespace["repo_path"], checkout)

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
                namespace = {}
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
