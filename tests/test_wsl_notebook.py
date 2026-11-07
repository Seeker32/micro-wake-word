"""Check WSL dependency and synthesis failure handling without installing ML packages."""

import json
import os
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

    def run_sample(self, fail_at=None):
        audio = Mock()
        commands = []
        display = types.ModuleType("IPython.display")
        display.Audio = audio

        def run(command, **kwargs):
            commands.append(command)
            self.assertTrue(kwargs["check"])
            if fail_at is not None and fail_at in command:
                raise subprocess.CalledProcessError(1, command)
            if "piper_sample_generator" in command:
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "0.wav").touch()

        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                model = Path("piper-sample-generator/models/en_US-libritts_r-medium.pt")
                model.parent.mkdir(parents=True)
                model.touch()
                with patch.dict("sys.modules", {"IPython.display": display}), \
                        patch("importlib.util.find_spec", return_value=object()), \
                        patch("subprocess.run", side_effect=run):
                    if fail_at is None:
                        exec(self.source("dEluu7nL7ywd"), {
                            "repo_path": Path(directory), "sample_dir": Path(directory) / "samples",
                        })
                    else:
                        with self.assertRaises(subprocess.CalledProcessError):
                            exec(self.source("dEluu7nL7ywd"), {
                                "repo_path": Path(directory), "sample_dir": Path(directory) / "samples",
                            })
            finally:
                os.chdir(previous_cwd)
        return audio, commands

    def test_dependency_failure_stops_before_synthesis_and_playback(self):
        audio, commands = self.run_sample(fail_at="pip")
        audio.assert_not_called()
        self.assertEqual(len(commands), 1)

    def test_synthesis_failure_stops_before_playback(self):
        audio, commands = self.run_sample(fail_at="piper_sample_generator")
        audio.assert_not_called()
        self.assertEqual(len(commands), 3)

    def test_success_plays_explicit_filename_and_reuses_existing_model(self):
        audio, commands = self.run_sample()
        audio.assert_called_once()
        self.assertTrue(Path(audio.call_args.kwargs["filename"]).is_absolute())
        self.assertTrue(audio.call_args.kwargs["autoplay"])
        self.assertEqual(len(commands), 3)


if __name__ == "__main__":
    unittest.main()
