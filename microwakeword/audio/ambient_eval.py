"""Score a trained streaming model on ambient audio: false accepts per hour.

Two ways to feed it, neither of which needs the training notebook at runtime:

* The ambient/positive feature mmaps the training notebook left behind
  (`<features_dir>/<pool>/<mode>/*_mmap`), found with `find_ambient_sets`.
* Audio folders, via `AudioTracks`: features are generated on the fly with the same
  C micro-frontend the training data used, so no TensorFlow and no feature cache
  are required.

`microwakeword.audio.ambient_features` builds the first kind of input from the
prepared datasets; it is deterministic and reuses a verified set instead of
regenerating identical features.

False accepts, cooldown and the ROC coordinates are the definitions from
`microwakeword.test`, copied here so evaluation works without TensorFlow;
`tests/test_ambient_eval.py` checks both implementations agree.

Usage:

    from microwakeword.audio.ambient_eval import (
        AudioTracks, evaluate_ambient, find_ambient_sets, find_audio_files,
    )

    sources = find_ambient_sets("features_v1/negative/ambient",
                                "features_v1/negative/ambient_chime6")
    # ...or, without any prepared features:
    sources = {"room": AudioTracks(find_audio_files("my_room_recordings"))}
    report = evaluate_ambient(
        "model/stream_state_internal_quant.tflite", sources,
        positives=AudioTracks(find_audio_files("wake_word_recordings")),
        positives_skip_first=0,
    )
    report["summary"]                       # one row per cutoff
    report["sources"]["room"]["false_accepts_per_hour"]

`evaluate_tracks` takes any callable that maps one spectrogram to streaming
probabilities, so the whole pipeline can be tested without a TFLite model.
"""

from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
from mmap_ninja.ragged import RaggedMmap


AMBIENT_MODES = ("testing_ambient", "validation_ambient")
STEP_MS = 10
WINDOW = 5  # moving average over 5 slices, as in the training notebook's ROC
IGNORE_SLICES_AFTER_ACCEPT = 25
POSITIVE_SKIP_FIRST = 25  # prepared positive clips are left-padded before the word
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg")
SAMPLE_RATE = 16000


def find_ambient_sets(*roots, mode="testing_ambient"):
    """Map a source name to the ragged mmaps that hold it.

    Every `<root>/**/<mode>/*_mmap` folder below the given roots is collected. The
    folder name without `_mmap` is the label (`noise`, `speech`, `chime6`); if two
    pools use the same label, the pool directory is prefixed to keep them apart.
    """
    if mode not in AMBIENT_MODES:
        raise ValueError(f"mode must be one of {AMBIENT_MODES}, got {mode!r}")
    found = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        for folder in sorted(root.rglob("*_mmap")):
            if folder.is_dir() and folder.parent.name == mode:
                found.append((folder, folder.relative_to(root)))
    if not found:
        raise FileNotFoundError(
            f"No <pool>/{mode}/*_mmap under {[str(Path(r)) for r in roots]}; "
            "run the training notebook's ambient feature cells, or pass audio folders"
        )
    counts = Counter(folder.name[: -len("_mmap")] for folder, _ in found)
    sources = {}
    for folder, relative in found:
        label = folder.name[: -len("_mmap")]
        if counts[label] > 1:
            pool = relative.parts[list(relative.parts).index(mode) - 1]
            label = f"{pool}_{label}"
        sources.setdefault(label, []).append(folder)
    return sources


def find_audio_files(*roots, suffixes=AUDIO_SUFFIXES):
    """Sorted audio files below the given directories (files are taken as given)."""
    files = []
    for root in roots:
        root = Path(root).expanduser()
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*")
                         if path.is_file() and path.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(
            f"No {', '.join(suffixes)} files under {[str(Path(r)) for r in roots]}"
        )
    return sorted(set(files))


def read_audio(path, sample_rate=SAMPLE_RATE):
    """Mono float32 samples, resampled to `sample_rate`, from WAV/FLAC/OGG."""
    samples, rate = _load_audio(path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)  # mix channels instead of picking one
    if rate != sample_rate:
        from scipy.signal import resample_poly

        divisor = gcd(int(rate), int(sample_rate))
        samples = resample_poly(samples, sample_rate // divisor, rate // divisor)
    return np.clip(samples, -1.0, 1.0).astype(np.float32)


def _load_audio(path):
    """Read via soundfile when available (FLAC/OGG), else scipy for plain WAV."""
    try:
        import soundfile
    except ImportError:
        soundfile = None
    if soundfile is not None:
        samples, rate = soundfile.read(str(path), dtype="float32", always_2d=False)
        return np.asarray(samples, dtype=np.float32), int(rate)
    from scipy.io import wavfile

    rate, samples = wavfile.read(str(path))
    if samples.dtype.kind == "i":
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    return np.asarray(samples, dtype=np.float32), int(rate)


def features_for_audio(samples, step_ms=STEP_MS):
    """Feature track for one clip, using the C micro-frontend on 10 ms steps.

    This is the same call the training data went through (`use_c=True` in
    `microwakeword.audio.audio_utils.generate_features_for_clip`), so features built
    here are interchangeable with the prepared `*_mmap` tracks.
    """
    from pymicro_features import MicroFrontend

    if step_ms != STEP_MS:
        raise ValueError(f"The C micro-frontend is fixed at {STEP_MS} ms, got {step_ms!r}")
    samples = np.asarray(samples)
    if samples.dtype in (np.float32, np.float64):
        samples = np.clip(samples * 32768, -32768, 32767).astype(np.int16)
    payload = samples.tobytes()
    frontend, features, offset = MicroFrontend(), [], 0
    while offset + 160 * 2 < len(payload):
        result = frontend.process_samples(payload[offset: offset + 160 * 2])
        offset += result.samples_read * 2
        if result.features:
            features.append(result.features)
    if not features:
        raise ValueError("Clip is too short to yield features")
    return np.array(features, dtype=np.float32)


class AudioTracks:
    """Indexable feature tracks from audio files, with durations read cheaply.

    `evaluate_tracks` only needs `len()` and `[index] -> spectrogram`, so this stands
    in for a `RaggedMmap` and lets the notebook evaluate plain recordings.
    """

    def __init__(self, paths, step_ms=STEP_MS):
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("AudioTracks needs at least one audio file")
        self.step_ms = step_ms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return features_for_audio(read_audio(self.paths[index]), step_ms=self.step_ms)

    @property
    def hours(self):
        return float(sum(audio_duration(path) for path in self.paths) / 3600.0)

    def describe(self):
        folders = sorted({str(path.parent) for path in self.paths})
        return f"{len(self.paths)} audio files in {', '.join(folders)}"


def audio_duration(path):
    try:
        import soundfile
    except ImportError:
        soundfile = None
    if soundfile is not None:
        info = soundfile.info(str(path))
        return float(info.frames) / float(info.samplerate)
    from scipy.io import wavfile

    rate, samples = wavfile.read(str(path))
    return float(len(samples)) / float(rate)


def open_tracks(source, max_tracks=None):
    """Return something indexable by track: a mmap path, AudioTracks, or a sequence."""
    tracks = RaggedMmap(str(source)) if isinstance(source, (str, Path)) else source
    if not hasattr(tracks, "__len__") or not hasattr(tracks, "__getitem__"):
        raise TypeError(f"Expected a mmap path, AudioTracks or tracks, got {type(source)}")
    if max_tracks is not None:
        if max_tracks < 1:
            raise ValueError(f"max_tracks must be positive, got {max_tracks!r}")
        tracks = tracks[:max_tracks]
    if not len(tracks):
        raise ValueError(f"No tracks in {source}")
    return tracks


def describe_source(tracks):
    """Short label for inventory tables and JSON reports."""
    if hasattr(tracks, "describe"):
        return tracks.describe()
    return str(Path(tracks))


def summarize_tracks(source, *, step_ms=STEP_MS, max_tracks=None):
    """Track count and audio hours, without decoding anything."""
    tracks = open_tracks(source, max_tracks)
    if hasattr(tracks, "hours"):
        hours = tracks.hours
    else:
        hours = sum(len(tracks[index]) for index in range(len(tracks))) * step_ms / 1000 / 3600
    return {"tracks": len(tracks), "hours": float(hours)}


def summarize_source(source, *, step_ms=STEP_MS, max_tracks=None):
    """Track count, hours and a short label for one source or a list of them."""
    inputs = list(source) if isinstance(source, (list, tuple)) else [source]
    rows = [summarize_tracks(item, step_ms=step_ms, max_tracks=max_tracks) for item in inputs]
    return {
        "tracks": sum(row["tracks"] for row in rows),
        "hours": float(sum(row["hours"] for row in rows)),
        "inputs": [describe_source(item) for item in inputs],
    }


def moving_average(probabilities, window=WINDOW):
    """Average each `window` of consecutive slices, as the ROC code does."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if window < 1:
        raise ValueError(f"window must be positive, got {window!r}")
    if len(probabilities) < window:
        raise ValueError(f"Need at least {window} slices, got {len(probabilities)}")
    return np.lib.stride_tricks.sliding_window_view(probabilities, window).mean(axis=-1)


class StreamPlayer:
    """Run one spectrogram through a streaming model, from a clean interpreter.

    A fresh interpreter per track stops streaming state leaking from one recording
    into the next, which would otherwise inflate or hide false accepts.
    """

    def __init__(self, model_path, stride=3):
        self.model_path = str(model_path)
        self.stride = int(stride)

    def __call__(self, spectrogram):
        from microwakeword.inference import Model

        model = Model(self.model_path, stride=self.stride)
        return np.asarray(model.predict_spectrogram(spectrogram), dtype=np.float64)


def evaluate_tracks(tracks, player, *, window=WINDOW, skip_first=0, progress=None):
    """Moving-average probabilities and per-track peaks for one set of tracks.

    `player` maps a spectrogram to streaming probabilities; `skip_first` drops the
    leading slices, which for prepared positive clips are left-padded context
    (`POSITIVE_SKIP_FIRST` matches `tflite_streaming_model_roc`). `progress(done, total)`
    is called after every track, and a fresh call per track is what keeps streaming
    state from one recording out of the next.
    """
    if isinstance(tracks, (str, Path)):
        tracks = open_tracks(tracks)
    probabilities, peaks = [], []
    for index in range(len(tracks)):
        stream = np.asarray(player(tracks[index]), dtype=np.float64)
        if len(stream) - skip_first < window:
            raise ValueError(f"Track {index} has {len(stream)} slices; "
                             f"need {window + skip_first}")
        averaged = moving_average(stream[skip_first:], window)
        probabilities.append(averaged)
        peaks.append(float(averaged.max()))
        if progress is not None:
            progress(index + 1, len(tracks))
    return {"probabilities": probabilities, "peaks": peaks, "track_count": len(peaks)}


def track_hours(probabilities, *, stride=3, step_s=STEP_MS / 1000):
    """Audio hours covered by moving-average tracks, counted as the ROC code does."""
    return float(sum(len(track) * stride * step_s for track in probabilities) / 3600.0)


def false_accepts_per_hour(probabilities, cutoffs, *, stride=3,
                           ignore_slices_after_accept=IGNORE_SLICES_AFTER_ACCEPT,
                           step_s=STEP_MS / 1000):
    """False accepts per hour at each cutoff, using the training notebook's convention.

    Same arithmetic as `microwakeword.test.compute_false_accepts_per_hour`: every
    track starts with a `ignore_slices_after_accept` cooldown, an accept resets it,
    and the rate is accepts over the audio duration of all tracks.
    """
    cutoffs = np.asarray(cutoffs, dtype=float)
    accepts = np.zeros(cutoffs.shape[0])
    duration_h = 0.0
    for track in probabilities:
        track = np.asarray(track, dtype=float)
        duration_h += len(track) * stride * step_s / 3600.0
        cooldown = np.full(cutoffs.shape[0], float(ignore_slices_after_accept))
        for probability in track:
            cooldown = np.maximum(cooldown - 1, 0)
            for index in np.nonzero(cooldown == 0)[0]:
                if probability > cutoffs[index]:
                    accepts[index] += 1
                    cooldown[index] = ignore_slices_after_accept
    return accepts / duration_h


def false_negative_rates(peaks, cutoffs):
    """Share of positive clips whose peak probability is not above each cutoff."""
    peaks = np.asarray(peaks, dtype=np.float64)
    if not len(peaks):
        raise ValueError("No positive clips")
    return np.array([1.0 - float((peaks > cutoff).mean()) for cutoff in cutoffs])


def roc_curve(false_accepts, false_rejections, cutoffs, max_faph=2.0):
    """FAPH/FRR coordinates plus the curve's area, as computed during training.

    Same coordinates as `microwakeword.test.generate_roc_curve`, trimmed at
    `max_faph` and walked from no false accepts towards that rate. The area is the
    integral over 0–`max_faph` false accepts per hour, not a normalized ROC AUC: a
    perfect model scores 0 and a useless one approaches `max_faph`.
    """
    false_accepts = np.asarray(false_accepts, dtype=float)
    false_rejections = np.asarray(false_rejections, dtype=float)
    cutoffs = np.asarray(cutoffs, dtype=float)
    if false_accepts[0] > max_faph:
        first = 1
        while false_accepts[first] > max_faph:
            first += 1
        x0, y0 = false_accepts[first - 1], false_rejections[first - 1]
        x1, y1 = false_accepts[first], false_rejections[first - 1]
        fnr = (y0 * (x1 - max_faph) + y1 * (max_faph - x0)) / (x1 - x0)
        start_cutoff = (cutoffs[first] + cutoffs[first - 1]) / 2.0
    else:
        first = 0
        fnr = false_rejections[first]
        start_cutoff = cutoffs[first]

    x, y, points = [max_faph], [fnr], [start_cutoff]
    for index in range(first, len(false_rejections)):
        if false_accepts[index] != x[-1]:  # one point per distinct rate
            x.append(false_accepts[index])
            y.append(false_rejections[index])
            points.append(cutoffs[index])
    if x[-1] > 0:
        x.append(0.0)
        y.append(1.0)
        points.append(0.0)
    x, y, points = np.flip(x), np.flip(y), np.flip(points)
    return {"faph": x, "frr": y, "cutoffs": points,
            "auc": float(np.trapezoid(y, x)) if len(x) > 1 else float("nan")}


def evaluate_ambient(model_path, sources, *, cutoffs=None, stride=3, window=WINDOW,
                     step_ms=STEP_MS, max_tracks=None, positives=None,
                     positives_skip_first=POSITIVE_SKIP_FIRST,
                     ignore_slices_after_accept=IGNORE_SLICES_AFTER_ACCEPT,
                     max_faph=2.0, progress=None, player=None):
    """Report false accepts per hour per source and pooled, plus the ROC if positives.

    `sources` maps a label to one track source or a list of them: a mmap path, an
    `AudioTracks`, as returned by `find_ambient_sets`. `max_tracks` keeps a run short
    while exploring; the hours reported then cover only the tracks that were read.
    `progress(label, done, total)` is called after every track so a long CHiME-6 pass
    can show where it is. `player` overrides the model callable, which is how the
    tests avoid TFLite.
    """
    cutoffs = np.arange(0, 1.01, 0.01) if cutoffs is None else np.asarray(cutoffs, dtype=float)
    if cutoffs.ndim != 1 or not len(cutoffs):
        raise ValueError("cutoffs must be a non-empty 1-D sequence")
    step_s = step_ms / 1000
    if player is None:
        player = StreamPlayer(model_path, stride=stride)
    report = {
        "model": str(model_path), "stride": stride, "step_ms": step_ms,
        "sliding_window_length": window, "cutoffs": cutoffs.tolist(),
        "ignore_slices_after_accept": ignore_slices_after_accept,
        "metric": "false accepts per hour on long ambient audio with no wake word; "
                  "same convention as tflite_streaming_roc.txt",
        "sources": {},
    }
    pooled, pooled_peaks = [], []
    for label, given in sources.items():
        inputs = list(given) if isinstance(given, (list, tuple)) else [given]
        probabilities, peaks, hours, tracks_read, described = [], [], 0.0, 0, []
        for source in inputs:
            open_mmap = open_tracks(source, max_tracks)
            result = evaluate_tracks(
                open_mmap, player, window=window,
                progress=None if progress is None else
                (lambda done, total, label=label, offset=tracks_read:
                 progress(label, offset + done, None)))
            probabilities += result["probabilities"]
            peaks += result["peaks"]
            hours += track_hours(result["probabilities"], stride=stride, step_s=step_s)
            tracks_read += result["track_count"]
            described.append(describe_source(source))
        rates = false_accepts_per_hour(
            probabilities, cutoffs, stride=stride,
            ignore_slices_after_accept=ignore_slices_after_accept, step_s=step_s)
        report["sources"][label] = {
            "inputs": described, "hours": hours,
            "track_count": tracks_read, "peaks": peaks,
            "false_accepts_per_hour": rates.tolist(),
            "false_accepts": [float(rate * hours) for rate in rates],
            "tracks_with_any_trigger": [int(sum(peak > cutoff for peak in peaks))
                                        for cutoff in cutoffs],
        }
        pooled += probabilities
        pooled_peaks += peaks

    pooled_rates = false_accepts_per_hour(
        pooled, cutoffs, stride=stride,
        ignore_slices_after_accept=ignore_slices_after_accept, step_s=step_s)
    pooled_hours = track_hours(pooled, stride=stride, step_s=step_s)
    report["pooled"] = {
        "hours": pooled_hours, "peaks": pooled_peaks,
        "false_accepts_per_hour": pooled_rates.tolist(),
        "false_accepts": [float(rate * pooled_hours) for rate in pooled_rates],
    }

    if positives is not None:
        result = evaluate_tracks(open_tracks(positives), player, window=window,
                                 skip_first=positives_skip_first)
        rates = false_negative_rates(result["peaks"], cutoffs)
        report["positives"] = {
            "input": describe_source(positives), "count": result["track_count"],
            "skip_first": positives_skip_first,
            "peaks": result["peaks"], "false_negative_rate": rates.tolist(),
        }
        report["roc"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                         for key, value in roc_curve(pooled_rates, rates, cutoffs,
                                                     max_faph=max_faph).items()}

    report["summary"] = [
        {
            "cutoff": float(cutoff),
            "false_negative_rate": None if "positives" not in report
            else report["positives"]["false_negative_rate"][index],
            "false_accepts_per_hour": float(pooled_rates[index]),
            **{f"{label}_faph": float(source["false_accepts_per_hour"][index])
               for label, source in report["sources"].items()},
        }
        for index, cutoff in enumerate(cutoffs)
    ]
    return report
