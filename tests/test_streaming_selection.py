"""Exercise the notebook's record-only streaming evaluation without TensorFlow."""

import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np


CELL_ID = "quantized-streaming-checkpoint-selection"
FUNCTIONS = {"track_accepts", "accept_counts", "threshold_curve", "miss_area",
             "write_curve", "select_streaming_checkpoint"}


class StreamingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "notebooks/basic_training_notebook.ipynb"
        cell = next(c for c in json.loads(path.read_text())["cells"]
                    if c.get("metadata", {}).get("id") == CELL_ID)
        tree = ast.parse("".join(cell["source"]))
        tree.body = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
        cls.code = compile(tree, str(path), "exec")

    def namespace(self, **overrides):
        namespace = {
            "np": np, "csv": csv, "json": json, "Path": Path, "shutil": shutil,
            "random": random, "re": re, "TemporaryDirectory": TemporaryDirectory,
            "datetime": datetime, "timezone": timezone,
            "SCORE_MAX": 10, "CUTOFFS": np.arange(11, dtype=np.int32), "COOLDOWN": 1,
            "RANKING_FAPH_LIMIT": 2.0,
            "CURVE_COLUMNS": ["weights", "step", "cutoff_bin", "cutoff", "missed",
                              "positives", "false_accepts", "ambient_hours", "frr", "faph"],
            "TFLITE_DIR": "quant", "TFLITE_FILE": "model.tflite",
            "hours": lambda tracks: 10.0,
        }
        namespace.update(overrides)
        exec(self.code, namespace)
        return namespace

    def test_track_accepts_matches_greedy_counting(self):
        rng = np.random.default_rng(7)
        track = rng.integers(0, 6, size=200).astype(np.int16)
        cutoffs = np.arange(11, dtype=np.int32)
        ns = self.namespace(COOLDOWN=5)
        counts = ns["track_accepts"](track, cutoffs)
        expected = []
        for cutoff in cutoffs:
            above = np.flatnonzero(track[4:] > cutoff)
            total, index = 0, 0
            while index < len(above):
                total += 1
                index = int(np.searchsorted(above, above[index] + 5))
            expected.append(total)
        self.assertEqual(counts.tolist(), expected)

    def test_threshold_curve_records_every_cutoff(self):
        ns = self.namespace()
        scored = (np.array([5, 5]), {"ambient": [np.array([0, 0, 5], dtype=np.int16)]})
        curve = ns["threshold_curve"](scored)
        self.assertEqual(len(curve["cutoffs"]), 11)
        self.assertEqual(curve["missed"].tolist(), [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2])
        self.assertEqual(curve["false_accepts"].tolist(), [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(curve["positives"], 2)
        self.assertEqual(curve["ambient_hours"], 10.0)
        self.assertAlmostEqual(curve["faph"][0], 0.1)
        self.assertAlmostEqual(curve["frr"][0], 0.0)
        self.assertAlmostEqual(curve["frr"][10], 1.0)

    def test_miss_area_takes_the_achievable_envelope(self):
        ns = self.namespace()
        # 阈值 0.0 处零误唤醒但漏掉全部正样本，不能提前把面积压到 0。
        area = ns["miss_area"]([0.0, 0.5, 3.0], [1.0, 0.5, 0.0])
        self.assertAlmostEqual(area, 1.25)
        # 所有阈值都超出 0–2 次/小时时，整段按 100% 漏唤醒计。
        self.assertEqual(ns["miss_area"]([3.0], [0.0]), 2.0)

    def run_selection(self, directory, candidates, validation, testing):
        train = Path(directory)

        def export_candidate(weights, target, calibration):
            blob = weights.read_bytes()
            model = Path(target) / "quant/model.tflite"
            model.parent.mkdir(parents=True)
            model.write_bytes(blob)
            (Path(target) / "stream_state_internal").mkdir()
            (Path(target) / "stream_state_internal/saved_model.pb").write_bytes(blob)

        def stream_scores(model_path, split):
            return (validation if split == "validation" else testing)[Path(model_path).read_bytes()]

        ns = self.namespace(
            train_dir=train,
            shortlist=lambda _: candidates,
            load_streaming_split=lambda split: split,
            export_candidate=export_candidate,
            stream_scores=stream_scores,
            FeatureHandler=lambda _: SimpleNamespace(get_data=lambda *a, **k: ([1, 2], None, None)),
            eval_config={"spectrogram_length": 1},
        )
        return ns, ns["select_streaming_checkpoint"]()

    def test_winner_uses_validation_only_and_always_exports(self):
        with TemporaryDirectory() as directory:
            train = Path(directory)
            best = train / "best_weights.weights.h5"
            best.write_bytes(b"best")
            snapshot = train / "train/candidate.weights.h5"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"snapshot")
            deployed = train / "quant/model.tflite"
            deployed.parent.mkdir()
            deployed.write_bytes(b"previous")

            validation = {
                b"best": (np.array([10, 10]), {"ambient": [np.zeros(6, dtype=np.int16)]}),
                b"snapshot": (np.array([1, 1]),
                              {"ambient": [np.array([0, 0, 10, 0], dtype=np.int16)]}),
            }
            testing = {
                b"best": (np.array([0]), {"ambient": [np.full(10, 10, dtype=np.int16)]}),
                b"snapshot": (np.array([10, 10]), {"ambient": [np.zeros(6, dtype=np.int16)]}),
            }

            ns, report = self.run_selection(
                directory, [(None, best), (1000, snapshot)], validation, testing)

            self.assertEqual(report["candidates"][0]["validation_area"], 0.0)
            self.assertAlmostEqual(report["candidates"][1]["validation_area"], 0.1)
            self.assertEqual(report["winner"]["weights"], "best_weights.weights.h5")
            self.assertIsNone(report["winner"]["step"])

            # 入选模型的测试成绩更差，但排名和导出都不看测试集。
            other = ns["threshold_curve"](testing[b"snapshot"])
            self.assertEqual(ns["miss_area"](other["faph"], other["frr"]), 0.0)
            self.assertGreater(report["winner"]["testing"]["area"], 0.0)
            self.assertNotIn("held_out_test_passed", json.dumps(report))
            self.assertNotIn("streaming_diagnostic", json.dumps(report))
            self.assertFalse((train / "streaming_diagnostic").exists())

            # 旧部署文件不动，这次评估的产物写进独立目录。
            self.assertEqual(deployed.read_bytes(), b"previous")
            evaluation_dir = train / report["evaluation_dir"]
            self.assertTrue(evaluation_dir.is_dir())
            self.assertEqual((evaluation_dir / "quant/model.tflite").read_bytes(), b"best")
            self.assertEqual((evaluation_dir / "best_streaming_weights.weights.h5").read_bytes(),
                             b"best")
            self.assertTrue((evaluation_dir / "selection.json").is_file())
            with (evaluation_dir / "validation_curves.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.reader(handle))), 2 * 11 + 1)
            with (evaluation_dir / "testing_curve.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.reader(handle))), 11 + 1)

    def test_equal_areas_prefer_the_non_streaming_best_weights(self):
        with TemporaryDirectory() as directory:
            train = Path(directory)
            best = train / "best_weights.weights.h5"
            best.write_bytes(b"best")
            snapshot = train / "train/candidate.weights.h5"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"snapshot")
            scores = {
                b"best": (np.array([10]), {"ambient": [np.zeros(4, dtype=np.int16)]}),
                b"snapshot": (np.array([10]), {"ambient": [np.zeros(4, dtype=np.int16)]}),
            }
            _, report = self.run_selection(
                directory, [(1000, snapshot), (None, best)], scores, scores)
            self.assertEqual(report["candidates"][0]["validation_area"],
                             report["candidates"][1]["validation_area"])
            self.assertEqual(report["winner"]["weights"], "best_weights.weights.h5")

    def test_each_run_keeps_the_previous_evaluation(self):
        with TemporaryDirectory() as directory:
            train = Path(directory)
            best = train / "best_weights.weights.h5"
            best.write_bytes(b"best")
            scores = {
                b"best": (np.array([10]), {"ambient": [np.zeros(4, dtype=np.int16)]}),
            }
            _, first = self.run_selection(directory, [(None, best)], scores, scores)
            _, second = self.run_selection(directory, [(None, best)], scores, scores)
            self.assertNotEqual(first["evaluation_dir"], second["evaluation_dir"])
            self.assertTrue((train / first["evaluation_dir"] / "selection.json").is_file())
            self.assertTrue((train / second["evaluation_dir"] / "selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
