"""Import split WAV hard negatives without changing their source split or phrase."""

import hashlib
import json
from math import gcd
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import wave
import zipfile

from microwakeword.audio.licensed_data import digest


SPLITS = ("training", "validation", "testing")


def import_audio_zip(audio_dir, audio_zip):
    """Use an existing split directory, or unpack the audio notebook's WAV ZIP."""
    audio_dir = Path(audio_dir)
    if audio_dir.is_dir():
        return audio_dir
    if audio_zip is None or not Path(audio_zip).is_file():
        raise FileNotFoundError(
            f"找不到 hey hi 音频目录 {audio_dir}。请先运行 hey_hi_negative_dataset.ipynb，"
            f"或将其导出的音频 ZIP 放到 {audio_zip}。"
        )
    audio_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=audio_dir.parent) as temporary:
        staged = Path(temporary) / "raw"
        staged.mkdir()
        with zipfile.ZipFile(audio_zip) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                path = PurePosixPath(member.filename)
                if (len(path.parts) != 2 or path.parts[0] not in SPLITS
                        or path.suffix.lower() != ".wav" or "\\" in member.filename):
                    raise ValueError(f"ZIP 只应包含 <split>/*.wav: {member.filename}")
                target = staged.joinpath(*path.parts)
                target.parent.mkdir(exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        if any(not list((staged / split).glob("*.wav")) for split in SPLITS):
            raise ValueError("hey hi ZIP 的 training/validation/testing 均需包含 WAV 音频")
        staged.rename(audio_dir)
    return audio_dir


def _read_pcm(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"hey hi 音频必须是单声道 PCM16: {path}")
        rate = wav.getframerate()
        frames = wav.getnframes()
        pcm = wav.readframes(frames)
    if not frames or len(pcm) != frames * 2:
        raise ValueError(f"空白或不完整的 hey hi WAV: {path}")
    return rate, frames, pcm


def prepare_hard_negative_features(
    audio_dir, features_dir, *, audio_zip=None, context_frames=204,
    step_ms=10, duration_s=3.2, tail_s=0.2,
):
    """Validate all audio before export; keep the complete phrase in the tail crop.

    The default 204-frame context matches both notebooks' current mixednet flags.
    Long phrases fail explicitly instead of silently becoming partial negatives.
    Each source yields one feature track, with no repetition or resplitting.
    """
    if context_frames < 1 or step_ms != 10 or tail_s < 0:
        raise ValueError("需要正数 context_frames、10 ms 特征步长和非负 tail_s")
    context_s = (context_frames - 1) * step_ms / 1000
    if context_s <= tail_s or duration_s < context_s + 0.03:
        raise ValueError("音频预处理长度和尾部留白与训练上下文不兼容")
    audio_dir = import_audio_zip(audio_dir, audio_zip)
    features_dir = Path(features_dir)
    records, seen = [], set()
    for split in SPLITS:
        paths = sorted((audio_dir / split).glob("*.wav"))
        if not paths:
            raise ValueError(f"hey hi 缺少 {split} WAV 音频: {audio_dir}")
        for path in paths:
            rate, frames, pcm = _read_pcm(path)
            fingerprint = hashlib.sha256(str(rate).encode() + b":" + pcm).hexdigest()
            if fingerprint in seen:
                raise ValueError(f"重复的 hey hi 原始音频，请先去重且保持集合隔离: {path}")
            seen.add(fingerprint)
            if frames / rate + tail_s > context_s:
                raise ValueError(
                    f"hey hi 音频过长，当前 {context_frames} 帧窗口无法保留完整发音: {path} "
                    f"({frames / rate:.2f}s)。请检查音频，或同步调整模型上下文和 hey_hi_context_frames。"
                )
            records.append({
                "path": path.relative_to(audio_dir).as_posix(), "split": split,
                "sha256": digest(path), "pcm_sha256": fingerprint,
                "sample_rate": rate, "frames": frames,
            })
    counts = {split: sum(r["split"] == split for r in records) for split in SPLITS}
    manifest = {
        "truth": False, "audio_dir": str(audio_dir.resolve()),
        "context_frames": context_frames, "step_ms": step_ms,
        "duration_s": duration_s, "tail_s": tail_s, "sample_rate": 16000,
        "processing": "resample; left-pad; retain complete phrase with trailing silence; truncate_start",
        "counts": counts, "files": records,
    }
    marker = features_dir / "feature_sources.json"
    if features_dir.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != manifest:
            raise FileExistsError(f"hey hi 特征与当前音频/参数不一致，请更换 hey_hi_features_dir: {features_dir}")

    import numpy as np
    from mmap_ninja.ragged import RaggedMmap
    from scipy.signal import resample_poly
    from microwakeword.audio.audio_utils import generate_features_for_clip

    if features_dir.exists():
        for split in SPLITS:
            if len(RaggedMmap(features_dir / split / "hey_hi_mmap")) != counts[split]:
                raise ValueError(f"hey hi 已缓存特征不完整: {split}")
        return manifest

    def generate(split):
        for record in records:
            if record["split"] != split:
                continue
            rate, _, pcm = _read_pcm(audio_dir / record["path"])
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            if rate != 16000:
                divisor = gcd(rate, 16000)
                audio = resample_poly(audio, 16000 // divisor, rate // divisor)
            audio = np.clip(audio, -1, 1).astype(np.float32)
            tail = round(tail_s * 16000)
            left = round(duration_s * 16000) - len(audio) - tail
            padded = np.pad(audio, (left, tail))
            feature = generate_features_for_clip(padded, step_ms=step_ms)
            if (feature.ndim != 2 or feature.shape[1] != 40
                    or len(feature) < context_frames or not np.isfinite(feature).all()):
                raise ValueError(f"无效的 hey hi 特征: {record['path']}")
            yield feature

    features_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=features_dir.parent) as temporary:
        staged = Path(temporary) / "features"
        for split in SPLITS:
            output = staged / split / "hey_hi_mmap"
            output.parent.mkdir(parents=True)
            RaggedMmap.from_generator(str(output), generate(split), batch_size=100, verbose=True)
            if len(RaggedMmap(output)) != counts[split]:
                raise RuntimeError(f"hey hi 特征数量异常: {split}")
        (staged / "feature_sources.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staged.rename(features_dir)
    return manifest


def evaluate_hard_negative_model(model_path, features_dir, *, cutoffs=(0.95, 0.98, 0.99)):
    """Report per-clip false triggers, separately from ambient false accepts/hour."""
    import numpy as np
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.inference import Model

    tracks = RaggedMmap(Path(features_dir) / "testing" / "hey_hi_mmap")
    if not len(tracks):
        raise ValueError("hey hi 测试集为空")
    peaks = []
    for track in tracks:
        # A fresh interpreter prevents streaming state leaking between recordings.
        model = Model(str(model_path), stride=3)
        probabilities = np.asarray(model.predict_spectrogram(track))[25:]
        if len(probabilities) < 5:
            raise ValueError("hey hi 测试特征长度不足")
        means = np.lib.stride_tricks.sliding_window_view(probabilities, 5).mean(axis=-1)
        peaks.append(float(means.max()))
    return {
        "sample_count": len(peaks), "sliding_window_length": 5,
        "stride": 3, "step_ms": 10,
        "metric": "fraction of held-out hey hi clips that trigger; not false accepts per hour",
        "results": [{
            "cutoff": float(cutoff), "false_triggers": sum(p > cutoff for p in peaks),
            "false_trigger_rate": sum(p > cutoff for p in peaks) / len(peaks),
        } for cutoff in cutoffs],
    }
