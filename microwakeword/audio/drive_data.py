"""Copy the datasets from a persistent folder (Google Drive) into the run directory.

Colab sessions start empty, the archives are ~11 GiB and the prepared audio ~15 GiB,
so the notebooks default to reusing whatever the user already keeps in Drive instead
of downloading it again. Copying is safe to repeat:

* a file already present with the same size is skipped, so a second run costs a walk;
* bytes land in a `.part` file and are renamed only when the copy finished, so an
  interrupted run never leaves a truncated archive behind;
* published archive MD5s (FSD50K, LibriSpeech) are checked after copying and a bad
  copy is deleted rather than trusted.

Deep verification of the copied `prepared/` audio stays with the code that owns those
manifests: `licensed_data.prepare_licensed_data` and `chime6.ensure_chime6` hash every
listed file when they load it.

Usage:

    from microwakeword.audio.drive_data import mount_drive, sync_datasets

    mount_drive()                      # no-op outside Colab
    report = sync_datasets("/content/drive/MyDrive/wake-word", ".")
    report["data_v1/prepared"]["copied"]
"""

from hashlib import md5
import os
from pathlib import Path
import shutil

from microwakeword.audio.licensed_data import FSD_FILES, SPEECH_FILES


# Drive entry -> destination relative to the run directory. The CHiME-6 archive goes
# where `ensure_chime6` already looks for it.
ENTRIES = {
    "trained_models": "trained_models",
    "data_v1/prepared": "data_v1/prepared",
    "data_v1/downloads": "data_v1/downloads",
    "chime6_v1/prepared": "chime6_v1/prepared",
    "chime6_v1.zip": "data_v1/downloads/chime6_v1.zip",
    "features_v1/negative/ambient": "features_v1/negative/ambient",
    "features_v1/negative/ambient_chime6": "features_v1/negative/ambient_chime6",
}
ARCHIVE_MD5 = dict(FSD_FILES)
ARCHIVE_MD5.update({f"{subset}.tar.gz": digest for _, (subset, digest) in SPEECH_FILES.items()})
OS_JUNK = (".DS_Store", "Thumbs.db", "desktop.ini")


def mount_drive(mountpoint="/content/drive"):
    """Mount Google Drive in Colab; return False where Drive is not available."""
    root = Path(mountpoint) / "MyDrive"
    if root.is_dir():
        return True
    try:
        from google.colab import drive
    except ModuleNotFoundError:
        return False
    drive.mount(str(mountpoint))
    return root.is_dir()


def sync_datasets(drive_root, run_root=".", *, entries=ENTRIES, verify=True,
                  progress=None):
    """Copy every known entry that exists in `drive_root`; skip what is already there.

    Returns `{drive entry: stats}` with `copied`/`skipped` file counts and the copied
    bytes. Missing entries are reported as absent instead of raising, so a Drive folder
    with only some of the datasets still works.
    """
    drive_root, run_root = Path(drive_root).expanduser(), Path(run_root).expanduser()
    report = {}
    for entry, destination in entries.items():
        source = drive_root / entry
        target = run_root / destination
        if not source.exists():
            report[entry] = {"source": str(source), "present": False}
            continue
        stats = copy_entry(source, target, progress=progress)
        stats["present"] = True
        if verify:
            stats["verified"] = verify_archives(target)
        report[entry] = stats
    return report


def copy_entry(source, target, *, progress=None):
    """Copy a file or directory tree, skipping identical files and never partially."""
    source, target = Path(source), Path(target)
    if not source.exists():
        raise FileNotFoundError(f"Nothing to copy from {source}")
    files = ([source] if source.is_file()
             else sorted(path for path in source.rglob("*")
                         if path.is_file() and path.name not in OS_JUNK
                         and not path.name.startswith("._")))
    stats = {"source": str(source), "target": str(target), "files": len(files),
             "copied": 0, "skipped": 0, "bytes": 0}
    for path in files:
        destination = target if source.is_file() else target / path.relative_to(source)
        if destination.is_file() and destination.stat().st_size == path.stat().st_size:
            stats["skipped"] += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        try:
            shutil.copyfile(path, partial)
            os.replace(partial, destination)  # only complete files get the real name
        finally:
            if partial.exists():
                partial.unlink()
        stats["copied"] += 1
        stats["bytes"] += destination.stat().st_size
        if progress is not None:
            progress(str(destination.relative_to(target)), stats["copied"], stats["files"])
    return stats


def verify_archives(target):
    """Check published MD5s of the archives under `target`; drop and report bad copies."""
    target = Path(target)
    checked, bad = {}, []
    for name, published in ARCHIVE_MD5.items():
        path = target / name
        if not path.is_file():
            continue
        actual = _md5(path)
        checked[name] = actual == published
        if actual != published:
            path.unlink(missing_ok=True)
            bad.append(name)
    if bad:
        raise ValueError(f"Copied archives do not match the published MD5 and were "
                         f"removed: {bad}")
    return {"checked": checked} if checked else None


def _md5(path):
    result = md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def archives_present(directory):
    """True when every archive `prepare_licensed_data` needs is already downloaded."""
    directory = Path(directory)
    names = [*ARCHIVE_MD5, "rirs_noises.zip"]
    return all((directory / name).is_file() for name in names)
