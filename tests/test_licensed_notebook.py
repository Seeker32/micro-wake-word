"""Exercise notebook dataset wiring without downloading audio or loading ML libraries."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

from microwakeword.audio.kokoro_v1 import (
    MODEL, REVISION, SPEEDS, generate_in_venv, load_training_manifest, speed_plan,
)
from microwakeword.audio.piper_clean100 import prepare_clean100_samples


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/basic_training_notebook.ipynb"
PREVIEW_CELL = "dEluu7nL7ywd"
POSITIVE_GENERATION_CELL = "-SvGtCCM9akR"
AUGMENTATION_CELL = "XW3bmbI5-JAz"
POSITIVE_FEATURES_CELL = "D7BHcY1mEGbK"
NEGATIVE_FEATURES_CELL = "1pGuJDPyp3ax"
TRAINING_CONFIG_CELL = "Ii1A14GsGVQT"
KAMANA_PHONEMES = "hˈeɪ kˌɑmˈa nə"


def make_kokoro_audio(root, speeds=(1.0,)):
    (root / "training").mkdir(parents=True)
    samples = []
    for index, speed in enumerate(speeds, 16):
        name = f"training/af_bella_{speed:g}.wav"
        with wave.open(str(root / name), "wb") as wav:
            wav.setparams((1, 2, 24000, 0, "NONE", ""))
            wav.writeframes(bytes((index, 0)) * 12000)
        samples.append({"file": name, "voice": "af_bella", "speed": speed})
    (root / "generation.json").write_text(json.dumps({
        "model": MODEL, "revision": REVISION, "text": "Hi Circle",
        "samples": samples,
    }))
    return root / "training/af_bella_1.wav"


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
            licensed_data.with_name("piper_clean100.py").touch()
            licensed_data.with_name("kokoro_v1.py").touch()
            namespace = {}
            source = source.replace('Path("/content/micro-wake-word")', f"Path({str(checkout)!r})")
            try:
                with patch("subprocess.run") as run:
                    exec(source, namespace)
                run.assert_not_called()
                self.assertEqual(namespace["repo_path"], checkout)
                self.assertEqual(namespace["wake_word_phonemes"], "hˈaɪ sˈɜːkəl")
                self.assertEqual(namespace["wake_word_text"], "Hi Circle")
                self.assertEqual(namespace["hey_hi_phonemes"], "hˈeɪ hˈaɪ")
                sample_root = checkout / "generated_samples/hi_circle"
                self.assertEqual(namespace["preview_dir"], sample_root / "clean100_v1_three_speeds_preview")
                self.assertEqual(namespace["sample_dir"], sample_root / "clean100_v1_three_speeds")
                self.assertEqual(namespace["positive_dir"].name, "positive_hi_circle_clean100_v1_three_speeds")
                self.assertFalse(namespace["USE_KOKORO"])
                self.assertEqual(namespace["kokoro_audio_dir"], sample_root / "kokoro_v1_wide_speeds_r2")
                self.assertTrue(namespace["train_dir"].name.endswith("_clean100_v1_hey_hi"))
                with patch("subprocess.run"):
                    exec(source.replace("USE_KOKORO = False", "USE_KOKORO = True"), namespace)
                self.assertTrue(namespace["train_dir"].name.endswith("_clean100_v1_hey_hi_kokoro_v1_wide_speeds_r2"))
                self.assertIn("_kokoro_v1_wide_speeds_r2", namespace["feature_sources_path"].name)
            finally:
                os.chdir(previous_cwd)

    def test_kokoro_generation_excludes_rejected_audio(self):
        plan = speed_plan("Hi Circle")
        self.assertNotIn("am_fenrir", plan)
        self.assertEqual(plan["am_puck"], (0.74, 0.86, 1.0, 1.25))
        self.assertEqual(sum(len(speeds) for speeds in plan.values()), 74)
        self.assertEqual(speed_plan("Other wake word")["am_puck"], SPEEDS)

    def test_cell_5_plays_piper_and_compares_kokoro_speeds_in_order(self):
        displayed = []
        modules = {
            "IPython.display": types.SimpleNamespace(Audio=lambda **kwargs: kwargs, display=displayed.append),
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules), \
                patch("microwakeword.audio.piper_clean100.load_clean100"), \
                patch("microwakeword.audio.piper_clean100.write_clean100_preview"), redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace = {
                "Path": Path, "sample_root": root, "preview_dir": root / "piper_preview",
                "clean100_model_dir": root / "model",
                "wake_word_text": "Hi Circle", "wake_word_phonemes": "hˈaɪ sˈɜːkəl",
                "kokoro_audio_dir": root / "kokoro_v1", "USE_KOKORO": False,
            }
            source = self.cell_source(PREVIEW_CELL)
            exec(source, namespace)
            piper = {"filename": str(root / "piper_preview/0.wav"), "autoplay": True}
            self.assertEqual(displayed, [piper])
            pilot = root / "kokoro_v1_pilot/listen_positive.wav"
            pilot.parent.mkdir()
            pilot.touch()
            displayed.clear()
            exec(source, namespace)
            self.assertEqual(displayed, [piper, {"filename": str(pilot)}])
            sample = make_kokoro_audio(namespace["kokoro_audio_dir"], speeds=(1.0, 0.74, 1.55))
            displayed.clear()
            exec(source, namespace)
            self.assertEqual(displayed[0], piper)
            self.assertEqual(len(displayed), 2)
            with wave.open(io.BytesIO(displayed[1]["data"]), "rb") as wav:
                self.assertEqual(wav.getframerate(), 24000)
                for index in (17, 16, 18):
                    self.assertEqual(wav.readframes(12000), bytes((index, 0)) * 12000)
                    if index != 18:
                        self.assertEqual(wav.readframes(9600), b"\0\0" * 9600)
                self.assertEqual(wav.readframes(1), b"")
            with self.assertRaisesRegex(ValueError, "唤醒词"):
                load_training_manifest(namespace["kokoro_audio_dir"], "Hi Other")
            sample.with_name("unlisted.wav").touch()
            with self.assertRaisesRegex(ValueError, "不一致"):
                load_training_manifest(namespace["kokoro_audio_dir"], "Hi Circle")

    def test_cell_5_mounts_drive_then_generates_only_a_missing_batch(self):
        modules = {
            "IPython.display": types.SimpleNamespace(Audio=lambda **kwargs: kwargs, display=lambda audio: None),
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules), \
                patch("microwakeword.audio.piper_clean100.load_clean100"), \
                patch("microwakeword.audio.piper_clean100.write_clean100_preview"), redirect_stdout(io.StringIO()):
            root = Path(directory)
            drive_root = root / "drive"
            output = drive_root / "MyDrive/kokoro_v1"
            data_dir = root / "data"
            namespace = {
                "Path": Path, "repo_path": root, "data_dir": data_dir, "sample_root": root,
                "preview_dir": root / "preview", "clean100_model_dir": root / "model",
                "wake_word_text": "Hi Circle", "wake_word_phonemes": "hˈaɪ sˈɜːkəl",
                "kokoro_audio_dir": output, "USE_KOKORO": False,
            }
            source = self.cell_source(PREVIEW_CELL).replace('"/content/drive"', repr(str(drive_root)))

            def generate_audio(repo, venv, text, dest):
                self.assertTrue((drive_root / "MyDrive").is_dir())
                return make_kokoro_audio(dest)

            drive = types.SimpleNamespace(mount=lambda path: (Path(path) / "MyDrive").mkdir(parents=True))
            with patch.dict("sys.modules", {"google.colab": types.SimpleNamespace(drive=drive)}), \
                    patch.object(drive, "mount", wraps=drive.mount) as mount, \
                    patch("microwakeword.audio.kokoro_v1.generate_in_venv", side_effect=generate_audio) as generate:
                exec(source, namespace)
                generate.assert_not_called()
                mount.assert_not_called()
                namespace["USE_KOKORO"] = True
                exec(source, namespace)
                exec(source, namespace)
                generate.assert_called_once_with(root, data_dir / "kokoro_venv", "Hi Circle", output)
                mount.assert_called_once_with(str(drive_root))

    def test_kokoro_bootstrap_and_generation_use_the_isolated_interpreter(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("microwakeword.audio.kokoro_v1.which", return_value=None), \
                patch("microwakeword.audio.kokoro_v1.sys.platform", "linux"), \
                patch("microwakeword.audio.kokoro_v1.subprocess.run") as run, redirect_stdout(io.StringIO()):
            repo = Path(directory).resolve()
            venv = repo / "venv"
            python = venv / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            audio = repo / "audio"
            generate_in_venv(repo, venv, "Hi Circle", audio)
            bootstrap, create, torch, packages, spacy, generate = [call.args[0] for call in run.call_args_list]
            self.assertEqual(bootstrap[:5], [sys.executable, "-m", "pip", "install", "--target"])
            self.assertEqual(create[:5], [f"{bootstrap[5]}/bin/uv", "venv", "--python", "3.12", "--seed"])
            self.assertEqual(torch[5:], ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
            self.assertTrue(all(cmd[0] == str(python) for cmd in (torch, packages, spacy, generate)))
            self.assertEqual(generate, [
                str(python), "-E", "-m", "microwakeword.audio.kokoro_v1", "Hi Circle", str(audio),
            ])
            run.reset_mock()
            generate_in_venv(repo, venv, "Hi Circle", repo / "other_audio")
            run.assert_called_once_with([*generate[:-1], str(repo / "other_audio")], cwd=repo, check=True)

    def fake_voice(self):
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                speaker_id_map={str(i): i for i in range(245)}, sample_rate=22050,
                phoneme_id_map=dict.fromkeys(KAMANA_PHONEMES, [1]),
            ),
            phonemes_to_ids=list,
        )

    def test_positive_generation_publishes_three_complete_speeds_and_reuses_them(self):
        source = self.cell_source(POSITIVE_GENERATION_CELL)
        with tempfile.TemporaryDirectory() as directory:
            sample_dir = Path(directory) / "clean100_v1_three_speeds"
            voice = self.fake_voice()
            namespace = {
                "sample_dir": sample_dir, "clean100_voice": voice,
                "wake_word_phonemes": KAMANA_PHONEMES,
                "prepare_clean100_samples": prepare_clean100_samples,
            }

            def fake_pcm(voice, phonemes, speaker, scale, noise):
                self.assertFalse(sample_dir.exists())
                return f"{speaker:04}{scale:.2f}{noise:.2f}".encode()

            with patch("microwakeword.audio.piper_clean100._synthesize_pcm", side_effect=fake_pcm) as synth, \
                    redirect_stdout(io.StringIO()) as log:
                exec(source, namespace)
                exec(source, namespace)
                self.assertEqual(synth.call_count, 3000)
                for speed in ("fast", "normal", "slow"):
                    self.assertEqual(len(list((sample_dir / speed).glob("*.wav"))), 1000)
                self.assertEqual(len({call.args[2] for call in synth.call_args_list}), 245)
                self.assertIn("共 3000 条", log.getvalue())
                marker = json.loads((sample_dir / "generation.json").read_text())
                self.assertEqual(marker["phonemes"], "hˈeɪ kˌɑmˈa nə")
                (sample_dir / "fast/0.wav").unlink()
                with self.assertRaises(FileExistsError):
                    exec(source, namespace)
                self.assertEqual(synth.call_count, 3000)

    def test_positive_generation_failure_leaves_no_published_or_staged_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("microwakeword.audio.piper_clean100._synthesize_pcm", return_value=b"same"):
                with self.assertRaisesRegex(RuntimeError, "不重复"):
                    prepare_clean100_samples(self.fake_voice(), root / "samples", KAMANA_PHONEMES)
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
        generation_settings = []

        class Clips:
            def __init__(self, **kwargs):
                loaded.append(kwargs)
                self.clips = ["fixture"]

        class SpectrogramGeneration:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def spectrogram_generator(self, **kwargs):
                generation_settings.append(kwargs)
                yield "feature"

        def from_generator(**kwargs):
            self.assertEqual(list(kwargs["sample_generator"]), ["feature"])
            generated.append(kwargs["out_dir"])
            Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=True)

        ragged = types.ModuleType("mmap_ninja.ragged")
        ragged.RaggedMmap = types.SimpleNamespace(from_generator=from_generator)
        yaml = types.ModuleType("yaml")
        yaml.dump = lambda config, file: file.write(json.dumps(config))
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                root = Path("data_v1/prepared")
                root.mkdir(parents=True)
                (root / "manifest.json").write_text('{"files": []}')
                namespace = {
                    "data_root": root,
                    "sample_dir": Path("generated_samples/hey_kamana/clean100_v1_three_speeds"),
                    "positive_dir": Path("features_v1/positive_hey_kamana_clean100_v1_three_speeds"),
                    "features_dir": Path("features_v1"),
                    "feature_sources_path": Path("features_v1/feature_sources.json"),
                    "chime6_features_dir": Path("features_v1/negative/ambient_chime6"),
                    "train_dir": Path("trained_models/hey_kamana_licensed_inputs"),
                    "config_path": Path("training_parameters_hey_kamana.yaml"),
                    "hey_hi_features_dir": Path("features_v1/negative/hey_hi_clean100_v1"),
                    "hey_hi_lineage": {"counts": {"training": 1}, "audio_manifest": "generated_datasets/hey_hi_negative_clean100_v1/raw/manifest.json"},
                    "hey_hi_context_frames": 204,
                    "USE_KOKORO": False,
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
                        self.assertIn(f"features_v1/negative/{corpus}/{split}/negative_mmap", generated)
                ambient = [row for row in loaded if "min_clip_duration_s" in row]
                self.assertEqual(len(ambient), 4)
                self.assertTrue(all("repeat_clip_min_duration_s" not in row for row in ambient))
                config = namespace["config"]
                self.assertEqual(config["features"][0]["features_dir"], str(namespace["positive_dir"]))
                self.assertTrue(all(row["features_dir"].startswith("features_v1/")
                                    for row in config["features"]))
                self.assertEqual(sum(row["truth"] for row in config["features"]), 1)
                self.assertNotEqual(config["train_dir"], "trained_models/wakeword")
                lineage = json.loads(Path("features_v1/feature_sources.json").read_text())
                self.assertEqual(lineage["positive_augmentation"]["testing"]["background_pool"],
                                 "data_v1/prepared/noise/testing")
                self.assertTrue(lineage["pending"])
                self.assertEqual(lineage["hard_negatives"], {"hey_hi": namespace["hey_hi_lineage"]})

                kokoro_audio_dir = Path("samples/kokoro_v1")
                sample = make_kokoro_audio(kokoro_audio_dir)
                kokoro_features_dir = Path("features_v1/positive_hi_circle_kokoro_v1")
                namespace.update(USE_KOKORO=True, wake_word_text="Hi Circle",
                                 kokoro_audio_dir=kokoro_audio_dir, kokoro_features_dir=kokoro_features_dir,
                                 feature_sources_path=Path("features_v1/feature_sources_mixed.json"))
                for split in ("training", "validation", "testing"):
                    (namespace["positive_dir"] / split / "wakeword_mmap").mkdir(parents=True)
                with patch.dict("sys.modules", modules):
                    exec(self.cell_source(POSITIVE_FEATURES_CELL), namespace)
                    self.assertEqual(generated[10:], [str(kokoro_features_dir / "training/wakeword_mmap")])
                    self.assertEqual(generation_settings[-1], {"repeat": 2})
                    self.assertEqual(namespace["kokoro_clips"].paths, [sample])
                    self.assertEqual(namespace["kokoro_spectrograms"].kwargs["augmenter"],
                                     namespace["augmenters"]["training"])
                    exec(self.cell_source(NEGATIVE_FEATURES_CELL), namespace)
                    exec(self.cell_source(TRAINING_CONFIG_CELL), namespace)
                mixed = namespace["config"]["features"]
                self.assertEqual([row for row in mixed if not row["truth"]],
                                 [row for row in config["features"] if not row["truth"]])
                self.assertEqual([row["sampling_weight"] for row in mixed if row["truth"]], [1.8, 0.2])
                self.assertFalse((kokoro_features_dir / "validation").exists())
                self.assertFalse((kokoro_features_dir / "testing").exists())
                mixed_lineage = json.loads(namespace["feature_sources_path"].read_text())
                self.assertEqual(mixed_lineage["kokoro_positive"]["training_wavs"], 1)
            finally:
                os.chdir(old_cwd)

    def test_changed_python_cells_compile_and_legacy_sources_are_absent(self):
        for cell_id in ("YJRG4Qvo9nXG", PREVIEW_CELL, AUGMENTATION_CELL, POSITIVE_FEATURES_CELL,
                        NEGATIVE_FEATURES_CELL, TRAINING_CONFIG_CELL):
            compile(self.cell_source(cell_id), f"cell-{cell_id}", "exec")
        source = "\n".join("".join(cell["source"]) for cell in self.cells)
        for legacy in ("agkphysics/AudioSet", "mchl914/fma_xsmall",
                       "MIT_environmental_impulse_responses", "negative_datasets/",
                       "datasets/kahrendt/microwakeword", "background_paths = ['fma_16k'",
                       "use_hey_hi_negatives"):
            self.assertNotIn(legacy, source)
        self.assertIn("augmenter=augmenters[split]", source)


if __name__ == "__main__":
    unittest.main()
