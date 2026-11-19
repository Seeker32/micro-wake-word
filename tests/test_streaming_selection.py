"""Exercise the notebook's selection policy without loading TensorFlow."""

import ast
import json
import math
from pathlib import Path
import random
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np


class StreamingSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "notebooks/basic_training_notebook.ipynb"
        cell = next(c for c in json.loads(path.read_text())["cells"]
                    if c.get("metadata", {}).get("id") == "quantized-streaming-checkpoint-selection")
        names = {"count_accepts", "lowest_cutoff", "point", "validation_point", "best_validation_index",
                 "select_streaming_checkpoint"}
        tree = ast.parse("".join(cell["source"]))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        cls.code = compile(tree, str(path), "exec")
        cls.globals = {"np": np, "math": math, "SCORE_MAX": 10, "COOLDOWN": 1, "hours": lambda _: 9.82}
        cls.functions = dict(cls.globals)
        exec(cls.code, cls.functions)

    def test_does_not_miss_every_positive_to_force_zero_false_accepts(self):
        scored = (np.array([8, 10]), {"ambient": [np.array([10, 7, 4])]})
        point = self.functions["validation_point"](scored, 0.1)
        self.assertEqual((point["cutoff_bin"], point["false_accepts"], point["missed"]), (7, 1, 0))
        self.assertGreater(point["faph"], 0.1)

    def test_keeps_a_working_threshold_within_budget_when_available(self):
        scored = (np.array([8, 10]), {"ambient": [np.array([6, 5])]})
        point = self.functions["validation_point"](scored, 0.1)
        self.assertEqual((point["cutoff_bin"], point["false_accepts"], point["missed"]), (6, 0, 0))

    def test_winner_uses_validation_only_and_requires_a_hit(self):
        rank = self.functions["best_validation_index"]
        silent = {"faph": 0, "frr": 1}
        working = {"faph": 0.2, "frr": 0.3}
        self.assertIsNone(rank([silent], 0.1))
        self.assertEqual(rank([silent, working, {"faph": 0.4, "frr": 0}], 0.1), 1)
        self.assertEqual(rank([working, {"faph": 0.1, "frr": 0.4}], 0.1), 1)

    def test_failed_holdout_keeps_the_tested_tflite_separate(self):
        with TemporaryDirectory() as directory:
            train = Path(directory)
            weights = train / "candidate.weights.h5"
            weights.touch()
            deployed = train / "quant/model.tflite"
            deployed.parent.mkdir()
            deployed.write_bytes(b"previous")

            def load_split(split):
                if split == "testing":
                    locked = json.loads((train / "streaming_checkpoint_selection.json").read_text())
                    self.assertEqual(locked["winner"]["cutoff_bin"], 7)
                    self.assertIsNone(locked["held_out_test"])
                return (np.array([8, 10]), {"ambient": [np.array([10, 7, 4])]})

            def export(_weights, directory, _calibration):
                (directory / "stream_state_internal").mkdir(parents=True)
                path = directory / "quant/model.tflite"
                path.parent.mkdir()
                path.write_bytes(b"candidate")

            ns = dict(self.globals, train_dir=train, shortlist=lambda _: [(500, weights)],
                      json=json, random=random, shutil=shutil, TemporaryDirectory=TemporaryDirectory,
                      Path=Path, load_streaming_split=load_split, eval_config={"spectrogram_length": 1},
                      FeatureHandler=lambda _: SimpleNamespace(get_data=lambda *a, **k: ([1], None, None)),
                      STREAMING_FAPH_BUDGET=0.1, TFLITE_DIR="quant", TFLITE_FILE="model.tflite", STRIDE=3,
                      WINDOW=5, export_candidate=export, stream_scores=lambda path, scored: scored,
                      hey_hi_features_dir=train, score_bins=lambda scores: scores,
                      write_diagnostic_roc=lambda *_: None,
                      ae=SimpleNamespace(open_mmap=lambda _: None, StreamPlayer=lambda *a, **k: None,
                                         evaluate_tracks=lambda *a, **k: {"peaks": np.array([5])},
                                         POSITIVE_SKIP_FIRST=0))
            exec(self.code, ns)
            report = ns["select_streaming_checkpoint"]()
            self.assertFalse(report["held_out_test_passed"])
            self.assertEqual((train / report["evaluated_tflite"]).read_bytes(), b"candidate")
            self.assertEqual(deployed.read_bytes(), b"previous")
