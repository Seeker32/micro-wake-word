"""Generate versioned wake-word WAV datasets with Bryce Beattie's Clean 100."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unicodedata import normalize
import wave

from microwakeword.audio.licensed_data import digest, download


LENGTH_SCALES = {"fast": 0.65, "normal": 1.0, "slow": 1.25}
NOISE_SCALES = (0.60, 0.667, 0.72, 0.78, 0.83)
NOISE_W_SCALE = 0.8
SAMPLES_PER_SPEED = 1000

MODEL_BASE_URL = "https://sfo3.digitaloceanspaces.com/bkmdls"
MODEL_HASHES = {
    "clean100.onnx": "a48b44084f5465a9d1c5e74c46c6ce6e1907af85e34f15292b9f44b13263f6e4",
    "clean100.onnx.json": "575a90e0f03154d30c4e8d9b1a86dcb590f7df2fc65e24466dde3c91491984b8",
}


def load_clean100(model_dir):
    """Load the hash-verified author release on CPU."""
    from piper import PiperVoice

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in MODEL_HASHES.items():
        path = model_dir / filename
        if path.is_file():
            if digest(path) != expected_hash:
                raise ValueError(f"{path} 的 SHA-256 不匹配，请先移走该文件。")
        else:
            with TemporaryDirectory(dir=model_dir) as staging:
                candidate = download(f"{MODEL_BASE_URL}/{filename}", Path(staging) / filename)
                if digest(candidate) != expected_hash:
                    raise ValueError(f"{filename} 下载后的 SHA-256 不匹配。")
                candidate.replace(path)
    return PiperVoice.load(model_dir / "clean100.onnx")


def _phoneme_ids(voice, ipa):
    phonemes = list(normalize("NFD", ipa))
    missing = set(phonemes) - voice.config.phoneme_id_map.keys()
    if missing:
        raise ValueError(f"Clean 100 模型未定义的音素：{sorted(missing)}")
    return voice.phonemes_to_ids(phonemes)


def _synthesize_pcm(voice, phoneme_ids, speaker_id, length_scale, noise_scale):
    import numpy as np
    from piper import SynthesisConfig

    audio = voice.phoneme_ids_to_audio(phoneme_ids, SynthesisConfig(
        speaker_id=speaker_id, length_scale=length_scale,
        noise_scale=noise_scale, noise_w_scale=NOISE_W_SCALE,
    ))
    peak = np.max(np.abs(audio), initial=0)
    if not np.isfinite(peak) or peak == 0:
        raise ValueError(f"Clean 100 ID {speaker_id} 输出为空或包含无效样本")
    return np.rint(audio / peak * 32767).astype("<i2").tobytes()


def _write_wav(path, pcm, sample_rate):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, sample_rate, 0, "NONE", ""))
        wav.writeframes(pcm)


def write_clean100_preview(voice, output_path, phonemes):
    """Render the reference speaker and speed used in the C comparison."""
    pcm = _synthesize_pcm(voice, _phoneme_ids(voice, phonemes), 0, 1.0, 0.667)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(output_path, pcm, voice.config.sample_rate)


def prepare_clean100_samples(voice, output_dir, phonemes, *, hard_negative_max_duration_s=None):
    """Generate or reuse 1000 WAVs per speed, with held-out splits for hard negatives."""
    output_dir = Path(output_dir)
    speaker_ids = sorted(set(voice.config.speaker_id_map.values()))
    if len(speaker_ids) * len(NOISE_SCALES) < SAMPLES_PER_SPEED:
        raise ValueError("Clean 100 说话人 ID 数量不足，无法生成预定的变体")
    split_negatives = hard_negative_max_duration_s is not None
    split_counts = {"training": 800, "validation": 100, "testing": 100}
    expected_counts = (
        {split: count * len(LENGTH_SCALES) for split, count in split_counts.items()}
        if split_negatives else dict.fromkeys(LENGTH_SCALES, SAMPLES_PER_SPEED)
    )
    split_seed = 42
    settings = {
        "model": "Bryce Beattie's Piper Clean 100",
        "model_release": "https://brycebeattie.com/files/tts/",
        "model_download_base": MODEL_BASE_URL,
        "model_sha256": MODEL_HASHES,
        "training_data": "LibriTTS-R train-clean-100",
        "training_data_source": "https://www.openslr.org/141/",
        "training_data_license": "https://creativecommons.org/licenses/by/4.0/",
        "phonemes": phonemes, "speaker_ids": speaker_ids,
        "sample_rate": voice.config.sample_rate, "length_scales": LENGTH_SCALES,
        "samples_per_speed": SAMPLES_PER_SPEED, "noise_scales": list(NOISE_SCALES),
        "noise_w_scale": NOISE_W_SCALE, "counts": expected_counts,
        "split_seed": split_seed if split_negatives else None,
        "max_duration_s": hard_negative_max_duration_s,
    }
    marker = output_dir / "generation.json"
    if output_dir.exists():
        counts = Counter(path.parent.name for path in output_dir.glob("*/*.wav") if path.is_file())
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != settings or counts != expected_counts:
            raise FileExistsError(f"{output_dir} 的生成参数或文件数量不匹配，请换用新的数据目录。")
        return expected_counts

    phoneme_ids = _phoneme_ids(voice, phonemes)
    seen = set()
    rng = Random(split_seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as staging:
        staged = Path(staging) / "samples"
        for folder in expected_counts:
            (staged / folder).mkdir(parents=True)
        for speed, scale in LENGTH_SCALES.items():
            if split_negatives:
                folders = [split for split, count in split_counts.items() for _ in range(count)]
                rng.shuffle(folders)
            else:
                folders = [speed] * SAMPLES_PER_SPEED
            for index, folder in enumerate(folders):
                variant, speaker = divmod(index, len(speaker_ids))
                for _ in range(10):
                    pcm = _synthesize_pcm(voice, phoneme_ids, speaker_ids[speaker], scale, NOISE_SCALES[variant])
                    fingerprint = sha256(pcm).digest()
                    if split_negatives and len(pcm) / (2 * voice.config.sample_rate) > hard_negative_max_duration_s:
                        continue
                    if fingerprint not in seen:
                        break
                else:
                    raise RuntimeError(f"Clean 100 无法为 {speed}/{index} 生成不重复且时长合规的音频")
                seen.add(fingerprint)
                filename = f"{speed}_{index}.wav" if split_negatives else f"{index}.wav"
                _write_wav(staged / folder / filename, pcm, voice.config.sample_rate)
            print(f"{output_dir.name} · {speed} 已生成 {SAMPLES_PER_SPEED} 条 WAV", flush=True)
        (staged / "generation.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged.rename(output_dir)
    return expected_counts
