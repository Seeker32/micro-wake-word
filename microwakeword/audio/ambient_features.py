"""Build the ambient-only feature sets from the prepared datasets, reproducibly.

Only the ambient evaluation sets are built (`<pool>/<mode>/*_mmap`); no training
features are touched. Builds are deterministic and idempotent:

* clips are processed in sorted path order, with no augmentation, shuffling,
  repetition or random chunking, so the same dataset and parameters always produce
  the same tracks;
* every build is staged in a temporary directory and renamed into place, so an
  interrupted run never leaves a half-written set;
* the manifest records the parameter signature, the prepared-dataset digest and a
  SHA-256 per track; a re-run verifies those and keeps the existing bytes, while a
  dataset, parameter or content change rebuilds instead of silently evaluating stale
  features.

Source integrity stays the job of `licensed_data.prepare_licensed_data` and
`chime6.ensure_chime6`; this module only records the manifest digest it built from.

Usage:

    from microwakeword.audio.ambient_features import build_all

    manifests = build_all("features_v1", licensed_prepared="data_v1/prepared",
                          chime6_prepared="chime6_v1/prepared")
    manifests["ambient"]["hours"]      # {"validation_ambient": 0.5, ...}
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.ambient_eval import (
    SAMPLE_RATE,
    STEP_MS,
    audio_duration,
    features_for_audio,
    read_audio,
)
from microwakeword.audio.licensed_data import digest


WINDOW_MS = 30
BINS = 40
MIN_DURATION_S = 6.0
AMBIENT_SPLITS = (("validation", "validation_ambient"), ("testing", "testing_ambient"))
# corpus -> pool directory below features_v1/negative, source pattern and the
# minimum clip duration its evaluation set uses (CHiME-6 segments are kept whole).
CORPORA = {
    "noise": {"pool": "ambient", "pattern": "*.wav", "min_duration_s": MIN_DURATION_S},
    "speech": {"pool": "ambient", "pattern": "*.flac", "min_duration_s": MIN_DURATION_S},
    "chime6": {"pool": "ambient_chime6", "pattern": "*.wav", "min_duration_s": 0.0},
}
MANIFEST_NAME = "feature_sources.json"


def build_all(features_dir, *, licensed_prepared=None, chime6_prepared=None,
              corpora=tuple(CORPORA), splits=AMBIENT_SPLITS, **options):
    """Build the licensed pool (noise + speech) and the CHiME-6 pool.

    Returns `{pool: manifest}`. A missing prepared directory is an error rather than a
    silently smaller evaluation set.
    """
    corpora = tuple(corpora)
    unknown = set(corpora) - set(CORPORA)
    if unknown:
        raise ValueError(f"Unknown corpora {sorted(unknown)}; known: {sorted(CORPORA)}")
    manifests = {}
    licensed = [corpus for corpus in corpora if corpus != "chime6"]
    if licensed:
        if licensed_prepared is None:
            raise ValueError("licensed_prepared is required for noise/speech ambient sets")
        manifests["ambient"] = build_ambient_set(
            licensed_prepared, Path(features_dir) / "negative" / "ambient",
            corpora=licensed, splits=splits, **options)
    if "chime6" in corpora:
        if chime6_prepared is None:
            raise ValueError("chime6_prepared is required for the CHiME-6 ambient set")
        manifests["ambient_chime6"] = build_ambient_set(
            chime6_prepared, Path(features_dir) / "negative" / "ambient_chime6",
            corpora=("chime6",), splits=splits, **options)
    return manifests


def build_ambient_set(prepared, output_dir, *, corpora=tuple(CORPORA),
                      splits=AMBIENT_SPLITS, min_duration_s=None,
                      step_ms=STEP_MS, rebuild=False, verify=True, progress=None):
    """Build (or verify and reuse) the ambient feature sets of one pool.

    `prepared` is a prepared-audio directory with a `manifest.json`, as written by
    `licensed_data.prepare_licensed_data` or `chime6.prepare_chime6`. `splits` maps
    official split names to feature mode folders. `verify=False` skips the per-track
    hash check and trusts the signature; `rebuild=True` regenerates unconditionally.
    """
    prepared, output_dir = Path(prepared), Path(output_dir)
    splits = tuple((str(split), str(mode)) for split, mode in splits)
    if not splits:
        raise ValueError("splits must not be empty")
    if step_ms != STEP_MS:
        raise ValueError(f"The C micro-frontend is fixed at {STEP_MS} ms, got {step_ms!r}")
    if min_duration_s is not None and min_duration_s < 0:
        raise ValueError(f"min_duration_s must be non-negative, got {min_duration_s!r}")
    corpora = tuple(corpora)
    if not corpora:
        raise ValueError("corpora must not be empty")
    unknown = set(corpora) - set(CORPORA)
    if unknown:
        raise ValueError(f"Unknown corpora {sorted(unknown)}; known: {sorted(CORPORA)}")

    records, manifest_path = _prepared_records(prepared)
    prepared_sha = digest(manifest_path)
    durations = _durations(corpora, min_duration_s)
    parameters = _parameters(corpora, splits, durations, step_ms)

    existing = _load_manifest(output_dir / MANIFEST_NAME)
    if not rebuild and existing and _same_inputs(existing, prepared_sha, parameters):
        if not verify or _tracks_match(existing, output_dir):
            print(f"Reusing verified ambient features in {output_dir}", flush=True)
            return existing
        print(f"Tracks in {output_dir} no longer match their manifest; rebuilding",
              flush=True)
    elif not rebuild and existing:
        print(f"Ambient features in {output_dir} came from other data or parameters; "
              "rebuilding", flush=True)

    request = _select(request_prepared=prepared, prepared_sha=prepared_sha,
                      records=records, corpora=corpora, splits=splits,
                      durations=durations, step_ms=step_ms,
                      parameters=parameters, dataset=_dataset(manifest_path))
    manifest = _build(request, output_dir, progress=progress)
    print(f"Built ambient features in {output_dir}", flush=True)
    return manifest


def _prepared_records(prepared):
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json in {prepared}; run the matching data preparation step")
    files = json.loads(manifest_path.read_text())["files"]
    return files, manifest_path


def _belongs(record, corpus):
    path = record["path"]
    return path.startswith("audio/") if corpus == "chime6" else path.split("/")[0] == corpus


def clip_duration(record, prepared):
    """Duration from the source manifest when it has it, else from the header."""
    if "duration_s" in record:
        return float(record["duration_s"])
    return audio_duration(Path(prepared) / record["path"])


def _durations(corpora, min_duration_s):
    return {corpus: (CORPORA[corpus]["min_duration_s"] if min_duration_s is None
                     else float(min_duration_s)) for corpus in corpora}


def _parameters(corpora, splits, durations, step_ms):
    return {
        "min_duration_s": durations, "step_ms": step_ms,
        "window_ms": WINDOW_MS, "bins": BINS, "sample_rate": SAMPLE_RATE,
        "corpora": sorted(corpora), "splits": [list(pair) for pair in splits],
        "pattern": {corpus: CORPORA[corpus]["pattern"] for corpus in sorted(corpora)},
    }


def _signature(prepared_sha, corpus, parameters, splits):
    payload = {"corpus": corpus, "prepared_manifest_sha256": prepared_sha,
               "splits": [list(pair) for pair in splits],
               "pattern": parameters["pattern"][corpus],
               "min_duration_s": parameters["min_duration_s"][corpus],
               **{key: parameters[key] for key in
                  ("step_ms", "window_ms", "bins", "sample_rate")}}
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_manifest(path):
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return manifest if manifest.get("ambient_only") is True else None


def _same_inputs(existing, prepared_sha, parameters):
    return (existing.get("prepared", {}).get("manifest_sha256") == prepared_sha
            and existing.get("parameters") == parameters)


def _tracks_match(manifest, output_dir):
    """Re-hash every recorded track; a few seconds, and it catches silent corruption."""
    for mode, sets in manifest["sets"].items():
        for corpus, set_info in sets.items():
            tracks = RaggedMmap(str(output_dir / mode / f"{corpus}_mmap"))
            if len(tracks) != len(set_info["clip_hashes"]):
                return False
            for index, expected in enumerate(set_info["clip_hashes"]):
                if track_sha256(tracks[index]) != expected:
                    return False
    return True


def track_sha256(features):
    """Digest of one feature track, used to prove a rebuild produced the same bytes."""
    return sha256(np.ascontiguousarray(features, dtype=np.float32).tobytes()).hexdigest()


def _select(*, request_prepared, prepared_sha, records, corpora, splits,
            durations, step_ms, parameters, dataset):
    """Resolve the clip list of every set: sorted paths, exact durations, no gaps."""
    request = {"prepared": Path(request_prepared).resolve(), "prepared_sha": prepared_sha,
               "splits": splits, "step_ms": step_ms, "parameters": parameters,
               "dataset": dataset, "corpora": {}}
    for corpus in corpora:
        pattern = CORPORA[corpus]["pattern"]
        rows = sorted((row for row in records if _belongs(row, corpus)),
                      key=lambda row: row["path"])
        clips = {}
        for split, mode in splits:
            minimum = durations[corpus]
            selected = [row for row in rows if row.get("split") == split
                        and clip_duration(row, request_prepared) > minimum]
            if not selected:
                raise ValueError(f"No {corpus} audio longer than {minimum}s for "
                                 f"split {split!r} in {request_prepared}")
            clips[mode] = selected
        request["corpora"][corpus] = {
            "pattern": pattern, "clips": clips,
            "signature": _signature(prepared_sha, corpus, parameters, splits),
        }
    return request


def _build(request, output_dir, *, progress):
    manifest = {
        "ambient_only": True,
        "dataset": request["dataset"],
        "prepared": {"root": str(request["prepared"]),
                     "manifest": str(request["prepared"] / "manifest.json"),
                     "manifest_sha256": request["prepared_sha"]},
        "parameters": request["parameters"],
        "processing": "16 kHz mono; whole clips; no augmentation, repetition, "
                      "shuffling or chunking; clips in sorted path order",
        "hours": {}, "sets": {}, "clips": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ambient-build-", dir=output_dir.parent) as stage:
        staged = Path(stage) / "features"
        for split, mode in request["splits"]:
            manifest["sets"][mode] = {}
            hours = 0.0
            for corpus, data in request["corpora"].items():
                hashes, frames = [], []

                def tracks(clips=data["clips"][mode]):
                    for row in clips:
                        features = _clip_features(request["prepared"] / row["path"],
                                                  request["step_ms"])
                        hashes.append(track_sha256(features))
                        frames.append(len(features))
                        yield features

                folder = staged / mode / f"{corpus}_mmap"
                folder.parent.mkdir(parents=True, exist_ok=True)
                RaggedMmap.from_generator(str(folder), tracks(), batch_size=1,
                                          verbose=False)
                if len(hashes) != len(data["clips"][mode]):
                    raise RuntimeError(f"{corpus} {mode}: generated {len(hashes)} tracks "
                                       f"for {len(data['clips'][mode])} clips")
                track_hours = sum(frames) * request["step_ms"] / 1000 / 3600
                manifest["sets"][mode][corpus] = {
                    "signature": data["signature"], "mmap": f"{mode}/{corpus}_mmap",
                    "tracks": len(hashes), "hours": track_hours,
                    "clip_hashes": hashes,
                }
                hours += track_hours
                manifest["clips"] += [
                    _clip_record(row, corpus, split, mode, frames[index], hashes[index])
                    for index, row in enumerate(data["clips"][mode])]
                if progress is not None:
                    progress(corpus, mode, len(hashes))
            manifest["hours"][mode] = hours
        (staged / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        _replace(staged, output_dir)
    return manifest


def _dataset(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    names = sorted({row.get("dataset", "") for row in manifest.get("files", [])
                    if row.get("dataset")})
    return " + ".join(names)


def _clip_record(row, corpus, split, mode, frames, feature_sha):
    record = {"corpus": corpus, "split": split, "mode": mode, "path": row["path"],
              "sha256": row.get("sha256"), "feature_frames": frames,
              "feature_sha256": feature_sha}
    for key in ("dataset", "version", "session", "room", "array", "channel",
                "start_s", "duration_s", "title", "license_url"):
        if key in row:
            record[key] = row[key]
    return record


def _clip_features(path, step_ms):
    """Decode with librosa (what the training stack used) and generate features."""
    try:
        import librosa
    except ImportError:
        samples = read_audio(path)
    else:
        samples = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)[0]
    features = features_for_audio(samples, step_ms=step_ms)
    if features.ndim != 2 or features.shape[1] != BINS:
        raise ValueError(f"{path}: unexpected feature shape {features.shape}")
    return features


def _replace(staged, output_dir):
    """Swap the staged pool in without ever leaving a partial set behind."""
    stale = output_dir.with_name(output_dir.name + ".stale")
    if output_dir.exists():
        if stale.exists():
            shutil.rmtree(stale)
        output_dir.rename(stale)
    staged.rename(output_dir)
    if stale.exists():
        shutil.rmtree(stale)


def main(argv=None):
    """Build the ambient feature sets from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--features-dir", default="features_v1")
    parser.add_argument("--licensed-prepared", default="data_v1/prepared")
    parser.add_argument("--chime6-prepared", default="chime6_v1/prepared")
    parser.add_argument("--corpora", default=",".join(CORPORA),
                        help="comma separated subset of noise,speech,chime6")
    parser.add_argument("--mode", default="both", choices=("both", "validation", "testing"))
    parser.add_argument("--min-duration-s", type=float, default=None,
                        help="override the per-corpus minimum clip duration")
    parser.add_argument("--rebuild", action="store_true",
                        help="regenerate even when the manifest verifies")
    parser.add_argument("--no-verify", action="store_true",
                        help="trust the manifest signature instead of re-hashing tracks")
    args = parser.parse_args(argv)

    splits = tuple(pair for pair in AMBIENT_SPLITS if args.mode in ("both", pair[0]))
    corpora = tuple(part.strip() for part in args.corpora.split(",") if part.strip())
    roots = {"noise": Path(args.licensed_prepared),
             "speech": Path(args.licensed_prepared),
             "chime6": Path(args.chime6_prepared)}
    available = tuple(corpus for corpus in corpora if roots[corpus].is_dir())
    for corpus in corpora:
        if corpus not in available:
            print(f"Skipping {corpus}: no prepared data in {roots[corpus]}", flush=True)
    if not available:
        raise SystemExit("No prepared datasets found; nothing to build")
    manifests = build_all(args.features_dir,
                          licensed_prepared=args.licensed_prepared,
                          chime6_prepared=args.chime6_prepared,
                          corpora=available, splits=splits,
                          min_duration_s=args.min_duration_s,
                          rebuild=args.rebuild, verify=not args.no_verify)
    for pool, manifest in manifests.items():
        hours = ", ".join(f"{mode}={value:.2f}h"
                          for mode, value in manifest["hours"].items())
        print(f"{pool}: {len(manifest['clips'])} clips, {hours}")


if __name__ == "__main__":
    main()
