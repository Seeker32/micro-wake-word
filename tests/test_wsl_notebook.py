"""Check WSL dependency and synthesis failure handling without installing ML packages."""

import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/basic_training_notebook_wsl.ipynb"


class WslNotebookTests(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text())["cells"]

    def source(self, cell_id):
        return next("".join(cell["source"]) for cell in self.cells
                    if cell.get("metadata", {}).get("id") == cell_id)

    def test_install_bootstraps_pip_in_kernel_environment(self):
        with patch("importlib.util.find_spec", return_value=None), \
                patch("subprocess.run") as run:
            namespace = {"repo_path": Path("/fixture/checkout")}
            exec(self.source("BFf6511E65ff"), namespace)
        commands = [call.args[0] for call in run.call_args_list]
        python = namespace["sys"].executable
        self.assertEqual(commands[0], [python, "-m", "ensurepip", "--upgrade"])
        self.assertEqual(commands[-1][:5], [python, "-m", "pip", "install", "-e"])
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))

    def run_sample(self, *, fail_install=False, fail_synthesis=False):
        audio = Mock()
        display = types.ModuleType("IPython.display")
        display.Audio = audio
        synthesize = Mock(side_effect=ValueError("failed synthesis") if fail_synthesis else None)
        load = Mock(return_value=object())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = {
                "clean100_model_dir": root / "model",
                "preview_dir": root / "preview",
                "wake_word_phonemes": "hˈeɪ tˈɛst",
            }
            with patch.dict("sys.modules", {"IPython.display": display}), \
                    patch("importlib.util.find_spec", return_value=object()), \
                    patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip") if fail_install else None) as run, \
                    patch("microwakeword.audio.piper_clean100.load_clean100", load), \
                    patch("microwakeword.audio.piper_clean100.write_clean100_preview", synthesize):
                source = self.source("dEluu7nL7ywd")
                if fail_install or fail_synthesis:
                    error = subprocess.CalledProcessError if fail_install else ValueError
                    with self.assertRaises(error):
                        exec(source, namespace)
                else:
                    exec(source, namespace)

            run.assert_called_once_with(
                [namespace["sys"].executable, "-m", "pip", "install", "piper-tts==1.3.0"],
                check=True,
            )
            return audio, load, synthesize, root

    def test_dependency_failure_stops_before_synthesis_and_playback(self):
        audio, load, synthesize, _ = self.run_sample(fail_install=True)
        load.assert_not_called()
        synthesize.assert_not_called()
        audio.assert_not_called()

    def test_synthesis_failure_stops_before_playback(self):
        audio, _, synthesize, _ = self.run_sample(fail_synthesis=True)
        synthesize.assert_called_once()
        audio.assert_not_called()

    def test_success_plays_the_clean100_preview(self):
        audio, load, synthesize, root = self.run_sample()
        load.assert_called_once_with(root / "model")
        synthesize.assert_called_once_with(load.return_value, root / "preview/0.wav", "hˈeɪ tˈɛst")
        audio.assert_called_once_with(filename=str(root / "preview/0.wav"), autoplay=True)


if __name__ == "__main__":
    unittest.main()
