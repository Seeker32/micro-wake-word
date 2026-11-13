import json
from pathlib import Path
import unittest


class ColabOnnxSetupTests(unittest.TestCase):
    def test_piper_is_installed_before_the_cuda_runtime(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/basic_training_notebook.ipynb"
        cells = json.loads(path.read_text())["cells"]
        source = "\n".join("".join(cell.get("source", [])) for cell in cells)
        piper = source.index('%pip install "piper-tts==1.3.0"')
        remove = source.index('%pip uninstall -y onnxruntime onnxruntime-gpu')
        cuda = source.index('%pip install "onnxruntime-gpu[cuda,cudnn]"')
        preload = source.index('ort.preload_dlls()')
        load = source.index('clean100_voice = load_clean100(')
        self.assertTrue(piper < remove < cuda < preload < load)
        self.assertNotIn('pip", "install", "piper-tts', source[cuda:load])


if __name__ == "__main__":
    unittest.main()
