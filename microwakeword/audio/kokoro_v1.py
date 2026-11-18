"""Generate a training-only WAV dataset in a separate Kokoro environment."""

import argparse
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
from shutil import which
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave


MODEL = "hexgrad/Kokoro-82M"
REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
WEIGHTS = "kokoro-v1_0.pth"
RATE = 24000
MAX_DURATION_S = 1.8
VOICES = (
    "af_bella", "af_heart", "af_nicole", "af_sarah",
    "am_michael", "am_puck", "am_onyx",
    "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
    "bm_fable", "bm_daniel", "bm_george", "bm_lewis",
)
SPEEDS = (0.74, 0.86, 1.0, 1.25, 1.55)


def speed_plan(text):
    # At 1.55× Puck drops "Hi" in "Hi Circle".
    return {
        voice: tuple(speed for speed in SPEEDS if speed != 1.55)
        if voice == "am_puck" and text.casefold() == "hi circle" else SPEEDS
        for voice in VOICES
    }


def generate_in_venv(repo_dir, venv_dir, text, output_dir):
    """Prepare an isolated CPU runtime on demand and run the existing generator."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} 已存在；请复用或用新的数据版本目录。")
    venv_dir = Path(venv_dir).resolve()
    python = venv_dir / "bin/python"
    ready = venv_dir / ".kokoro-ready"
    if not (python.is_file() and ready.is_file()):
        print("首次准备 Kokoro 独立环境（Python 3.12）…", flush=True)
        with TemporaryDirectory() as bootstrap:
            uv = which("uv")
            if uv is None:
                subprocess.run([sys.executable, "-m", "pip", "install", "--target", bootstrap, "uv"], check=True)
                uv = str(Path(bootstrap) / "bin/uv")
            subprocess.run([uv, "venv", "--python", "3.12", "--seed", "--allow-existing", str(venv_dir)],
                           check=True)
        pip = [str(python), "-E", "-m", "pip", "install"]
        if sys.platform.startswith("linux"):
            subprocess.run([*pip, "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
        subprocess.run([*pip, "kokoro==0.9.4", "misaki[en]==0.9.4"], check=True)
        subprocess.run([str(python), "-E", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        ready.touch()
    subprocess.run([str(python), "-E", "-m", "microwakeword.audio.kokoro_v1", text, str(output_dir)],
                   cwd=repo_dir, check=True)


def load_training_manifest(audio_dir, text, *, max_duration_s=MAX_DURATION_S):
    """Check that the training directory contains exactly the reviewed WAVs."""
    audio_dir = Path(audio_dir)
    manifest_path = audio_dir / "generation.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 {manifest_path}，请先用 microwakeword.audio.kokoro_v1 生成音频。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("model"), manifest.get("revision"), manifest.get("text")) != (MODEL, REVISION, text):
        raise ValueError(f"Kokoro 模型版本或唤醒词与当前配置不符：{manifest_path}")
    expected = [row["file"] for row in manifest["samples"]]
    paths = sorted((audio_dir / "training").glob("*.wav"))
    if not paths or sorted(expected) != [path.relative_to(audio_dir).as_posix() for path in paths]:
        raise ValueError(f"Kokoro training/*.wav 与 generation.json 不一致：{audio_dir}")
    seen = set()
    for path in paths:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
                raise ValueError(f"Kokoro WAV 必须是 24 kHz 单声道 PCM16：{path}")
            if frames / RATE > max_duration_s:
                raise ValueError(f"Kokoro WAV 超出训练截窗：{path}（{frames / RATE:.2f} 秒）")
            pcm = wav.readframes(frames)
        if len(pcm) != frames * 2 or not any(pcm) or pcm in seen:
            raise ValueError(f"Kokoro WAV 为空、不完整或重复：{path}")
        seen.add(pcm)
    return manifest


def speed_preview_wav(audio_dir, samples):
    """Join validated WAVs in playback order without changing their speed."""
    result = BytesIO()
    pause = b"\0\0" * round(0.4 * RATE)
    with wave.open(result, "wb") as out:
        out.setparams((1, 2, RATE, 0, "NONE", ""))
        for index, sample in enumerate(samples):
            if index:
                out.writeframesraw(pause)
            with wave.open(str(Path(audio_dir) / sample["file"]), "rb") as wav:
                out.writeframesraw(wav.readframes(wav.getnframes()))
    return result.getvalue()


class KokoroClips:
    """Read validated PCM16 WAVs for SpectrogramGeneration without TorchCodec."""

    def __init__(self, audio_dir, manifest):
        self.paths = [Path(audio_dir) / sample["file"] for sample in manifest["samples"]]

    def audio_generator(self, repeat=1):
        from scipy.io import wavfile
        from scipy.signal import resample_poly

        for _ in range(repeat):
            for path in self.paths:
                _, pcm = wavfile.read(path)
                yield resample_poly(pcm.astype("float32") / 32768.0, 2, 3).clip(-1, 1)


def generate(text, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} 已存在；请复用或用新的数据版本目录。")
    text = text.strip()
    if not text:
        raise ValueError("请提供唤醒词文本。")

    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline
    import numpy as np
    import torch

    def asset(filename):
        return hf_hub_download(MODEL, filename, revision=REVISION, token=False)

    torch.set_num_threads(4)
    model = KModel(repo_id=MODEL, config=asset("config.json"), model=asset(WEIGHTS)).eval()
    pipes = {lang: KPipeline(lang_code=lang, repo_id=MODEL, model=model, device="cpu") for lang in ("a", "b")}
    card = f"https://huggingface.co/{MODEL}/blob/{REVISION}/README.md"
    manifest = {
        "model": MODEL, "version": "v1.0", "revision": REVISION, "model_weights": WEIGHTS,
        "license": "Apache-2.0", "model_card": card,
        "training_data_attribution": {"Koniwa (tnc)": "CC BY 3.0", "SIWIS": "CC BY 4.0",
                                      "details": f"{card}#creative-commons-attribution"},
        "software": {name: version(name) for name in ("kokoro", "misaki", "torch")},
        "text": text, "sample_rate": RATE, "channels": 1, "encoding": "PCM_16",
        "samples": [],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "audio"
        (staged / "training").mkdir(parents=True)
        seen = set()
        for voice, speeds in speed_plan(text).items():
            pack = asset(f"voices/{voice}.pt")
            for speed in speeds:
                segments = list(pipes[voice[0]](text, voice=pack, speed=speed))
                output = segments[0].audio if len(segments) == 1 else None
                audio = output.detach().cpu().numpy() if output is not None else np.array([])
                peak = np.max(np.abs(audio), initial=0)
                if audio.ndim != 1 or not 0.02 < peak < 1:
                    print(f"跳过 {voice} ×{speed:g}：空音频、多段输出或削波")
                    continue
                active = np.flatnonzero(np.abs(audio) >= peak * 0.01)
                # Leave a margin so quiet consonants survive the silence trim.
                pad = round(RATE * 0.06)
                start, stop = max(int(active[0]) - pad, 0), min(int(active[-1]) + pad + 1, len(audio))
                pcm = np.rint(audio[start:stop] * 32767).astype("<i2").tobytes()
                if stop - start > RATE * MAX_DURATION_S or pcm in seen:
                    print(f"跳过 {voice} ×{speed:g}：裁剪后超过 1.8 秒或重复")
                    continue
                seen.add(pcm)
                name = f"training/{voice}_{speed:g}.wav"
                with wave.open(str(staged / name), "wb") as wav:
                    wav.setparams((1, 2, RATE, 0, "NONE", ""))
                    wav.writeframes(pcm)
                manifest["samples"].append({
                    "file": name, "voice": voice, "speed": speed, "phonemes": segments[0].phonemes,
                    "crop_samples": [start, stop],
                })

        (staged / "generation.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        load_training_manifest(staged, text)
        staged.rename(output_dir)
    print(f"Kokoro · {len(manifest['samples'])} 条候选音频：{output_dir}，请试听后再用于训练。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Wake phrase, e.g. 'Hi Circle'")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    generate(args.text, args.output_dir)
