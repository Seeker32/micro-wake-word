"""Prepare the notebook's replacement datasets and record their provenance.

This audits these data inputs only, not TTS weights or product dependencies.
"""

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile


FSD_SOURCE = "https://zenodo.org/records/4060432"
RIR_SOURCE = "https://www.openslr.org/28/"
SPEECH_SOURCE = "https://www.openslr.org/12/"
BY = "https://creativecommons.org/licenses/by/4.0/"
CC_TEXT_SOURCE = (
    "https://raw.githubusercontent.com/creativecommons/cc-legal-tools-data/"
    "a0dac18d1a773b4cdb7d8dd5d9bf4efab7901030/docs"
)
FSD_FILES = {
    "FSD50K.metadata.zip": "b9ea0c829a411c1d42adb9da539ed237",
    "FSD50K.doc.zip": "3516162b82dc2945d3e7feba0904e800",
    "FSD50K.eval_audio.z01": "3090670eaeecc013ca1ff84fe4442aeb",
    "FSD50K.eval_audio.zip": "6fa47636c3a3ad5c7dfeba99f2637982",
}
SPEECH_FILES = {
    "training": ("train-clean-100", "2a93770f6d5c6c964bc36631d331a522"),
    "validation": ("dev-clean", "42e2234ba48799c1f50f24a7926300a1"),
    "testing": ("test-clean", "32fa31d27d2e1cad72775fee3f4849a9"),
}


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url, path, md5=None):
    """Keep interrupted downloads out of the cache; check published MD5s."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            print(f"Downloading {url}", flush=True)
            with urlopen(url, timeout=60) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            if md5 and digest(temporary, "md5") != md5:
                raise ValueError(f"Checksum mismatch: {url}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if md5 and digest(path, "md5") != md5:
        raise ValueError(f"Corrupt cached download; remove and retry: {path}")
    return path


def allowed_fsd_license(value):
    """Exact URL allowlist: missing, NC, ND, SA, Sampling+ and unknown fail closed."""
    url = urlsplit(value or "")
    return (
        url.scheme in ("http", "https")
        and url.netloc == "creativecommons.org"
        and not url.query
        and not url.fragment
        and url.path in (
            "/publicdomain/zero/1.0/",
            "/licenses/by/3.0/",
            "/licenses/by/4.0/",
        )
    )


def fsd_split(uploader):
    # Keep every clip from an uploader in one split, including augmentation audio.
    bucket = int(hashlib.sha256(uploader.encode()).hexdigest(), 16) % 10
    return "validation" if bucket == 0 else "testing" if bucket == 1 else "training"


def copy_audio(source, destination, *, dataset, version, upstream_path,
               source_url, license_url, author, title, split, records, root):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"Duplicate audio destination: {destination}")
    with destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    records.append({
        "path": destination.relative_to(root).as_posix(),
        "sha256": digest(destination), "dataset": dataset, "version": version,
        "upstream_path": upstream_path, "source_url": source_url,
        "license_url": license_url, "author": author, "title": title,
        "split": split, "modifications": "none (original audio bytes)",
    })


def select_fsd(metadata, archive, root, records, excluded):
    """Select only licensed clips, without extracting arbitrary archive paths."""
    for clip_id, info in sorted(metadata.items()):
        if not re.fullmatch(r"[0-9]+", clip_id):
            raise ValueError(f"Invalid FSD50K clip ID: {clip_id!r}")
        if not allowed_fsd_license(info.get("license")):
            excluded.append({"id": clip_id, "license": info.get("license", ""),
                             "reason": "license not on CC0/CC-BY allowlist"})
            continue
        if not info.get("uploader") or not info.get("title"):
            raise ValueError(f"Missing attribution for FSD50K clip {clip_id}")
        split = fsd_split(info["uploader"])
        member = f"FSD50K.eval_audio/{clip_id}.wav"
        with archive.open(member) as source:
            copy_audio(source, root / "noise" / split / f"{clip_id}.wav",
                       dataset="FSD50K", version="1.0, record 4060432 (eval subset)",
                       upstream_path=member,
                       source_url=f"https://freesound.org/s/{clip_id}/",
                       license_url=info["license"], author=info["uploader"],
                       title=info["title"], split=split, records=records, root=root)


def select_rirs(archive, root, records):
    for member in archive.namelist():
        path = PurePosixPath(member)
        if (path.parts[:2] != ("RIRS_NOISES", "simulated_rirs")
                or path.suffix != ".wav" or ".." in path.parts):
            continue
        with archive.open(member) as source:
            copy_audio(source, root / "rirs" / Path(*path.parts[2:]),
                       dataset="OpenSLR SLR28", version="RIRS_NOISES (2017)",
                       upstream_path=member, source_url=RIR_SOURCE,
                       license_url="https://www.apache.org/licenses/LICENSE-2.0",
                       author="RIRS_NOISES contributors (see sources/SLR28-README)",
                       title=path.name, split="shared augmentation RIR",
                       records=records, root=root)


def select_speech(archive, subset, split, root, records):
    for member in archive:
        path = PurePosixPath(member.name)
        if (not member.isfile() or path.parts[:2] != ("LibriSpeech", subset)
                or path.suffix != ".flac" or ".." in path.parts):
            continue
        if not re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+\.flac", path.name):
            raise ValueError(f"Invalid LibriSpeech clip name: {member.name}")
        with archive.extractfile(member) as source:
            copy_audio(source, root / "speech" / split / path.name,
                       dataset="LibriSpeech", version=f"SLR12 (2015), {subset}",
                       upstream_path=member.name, source_url=SPEECH_SOURCE,
                       license_url=BY,
                       author=f"V. Panayotov, G. Chen, D. Povey, S. Khudanpur; reader {path.parts[2]}",
                       title=path.stem, split=split, records=records, root=root)


def prepare_licensed_data(directory="licensed_data_v1"):
    """Download replacements (~15 GB compressed), select audio, write audit files.

    FSD50K's eval subset is re-split by uploader for this custom wake-word task;
    these splits must not be presented as an official FSD50K benchmark.
    """
    directory = Path(directory)
    raw = directory / "downloads"
    prepared = directory / "prepared"
    if prepared.exists():
        records = json.loads((prepared / "manifest.json").read_text())["files"]
        expected = {row["path"] for row in records}
        actual = {p.relative_to(prepared).as_posix() for folder in ("noise", "speech", "rirs")
                  for p in (prepared / folder).rglob("*") if p.is_file()}
        if actual != expected or any(digest(prepared / r["path"]) != r["sha256"] for r in records):
            raise ValueError("Prepared data changed; use a fresh data directory")
        return prepared
    raw.mkdir(parents=True, exist_ok=True)
    archives = []

    def fetch(url, filename, md5=None):
        path = download(url, raw / filename, md5)
        archives.append({"url": url, "filename": filename, "sha256": digest(path),
                         "published_md5": md5})
        return path

    for filename, md5 in FSD_FILES.items():
        fetch(f"{FSD_SOURCE}/files/{filename}?download=1", filename, md5)
    joined = raw / "FSD50K.eval_audio.joined.zip"
    if not joined.exists():
        # shortcut: joins only the two-part eval archive; revisit for the dev archive.
        temporary = raw / "FSD50K.eval_audio.joining.zip"
        try:
            subprocess.run(["zip", "-s", "0", str(raw / "FSD50K.eval_audio.zip"),
                            "--out", str(temporary)], check=True)
            temporary.replace(joined)
        finally:
            temporary.unlink(missing_ok=True)
    rirs = fetch("https://www.openslr.org/resources/28/rirs_noises.zip", "rirs_noises.zip")
    speech = {split: fetch(f"https://www.openslr.org/resources/12/{subset}.tar.gz",
                           f"{subset}.tar.gz", md5)
              for split, (subset, md5) in SPEECH_FILES.items()}

    with tempfile.TemporaryDirectory(prefix="preparing-", dir=directory) as staging:
        root = Path(staging)
        sources = root / "sources"
        sources.mkdir()
        for label, url in (("FSD50K", FSD_SOURCE), ("SLR28", RIR_SOURCE),
                           ("SLR12", SPEECH_SOURCE)):
            download(url, sources / f"{label}.html")
        with zipfile.ZipFile(raw / "FSD50K.doc.zip") as archive:
            for filename in ("LICENSE-DATASET", "README.md"):
                (sources / f"FSD50K-{filename}").write_bytes(archive.read(f"FSD50K.doc/{filename}"))
        with zipfile.ZipFile(raw / "FSD50K.metadata.zip") as archive:
            metadata_bytes = archive.read("FSD50K.metadata/eval_clips_info_FSD50K.json")
            (sources / "eval_clips_info_FSD50K.json").write_bytes(metadata_bytes)
        records, excluded = [], []
        with zipfile.ZipFile(joined) as archive:
            select_fsd(json.loads(metadata_bytes), archive, root, records, excluded)
        with zipfile.ZipFile(rirs) as archive:
            for label, member in (
                ("SLR28-README", "RIRS_NOISES/README"),
                ("SLR28-simulated-README", "RIRS_NOISES/simulated_rirs/README"),
            ):
                (sources / label).write_bytes(archive.read(member))
            select_rirs(archive, root, records)
        for split, path in speech.items():
            with tarfile.open(path, "r|gz") as archive:
                select_speech(archive, SPEECH_FILES[split][0], split, root, records)
        for folder in ("noise", "speech"):
            for split in SPEECH_FILES:
                if not list((root / folder / split).glob("*")):
                    raise ValueError(f"No eligible audio in {folder}/{split}")
        if not list((root / "rirs").rglob("*.wav")):
            raise ValueError("No simulated RIRs found")
        license_documents = []
        for license_url in sorted({r["license_url"] for r in records} | {BY}):
            name = hashlib.sha256(license_url.encode()).hexdigest()[:16]
            source_url = license_url.replace("http://", "https://", 1)
            if urlsplit(source_url).netloc == "creativecommons.org":
                source_url = CC_TEXT_SOURCE + urlsplit(source_url).path + "legalcode.txt"
            path = download(source_url, sources / f"license-{name}.txt")
            license_documents.append({"license_url": license_url, "source_url": source_url,
                                      "path": path.relative_to(root).as_posix(),
                                      "sha256": digest(path)})
        manifest = {
            "scope": "replacement augmentation and negative audio only",
            "pending": ["legacy TTS model provenance", "product dependency/distribution audit",
                        "fresh training and real-device evaluation"],
            "policy": "CC0/CC-BY allowlist; simulated SLR28 RIRs; official LibriSpeech splits",
            "archives": archives, "license_documents": license_documents, "files": records,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for filename, rows, fields in (
            ("manifest.csv", records, list(records[0])),
            ("excluded_fsd.csv", excluded, ["id", "license", "reason"]),
        ):
            with (root / filename).open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        notices = ["# Audio data attribution\n",
                   f"FSD50K v1.0 — Eduardo Fonseca et al.; dataset CC BY 4.0: {FSD_SOURCE}\n",
                   "Original audio bytes are retained. Feature extraction, resampling, mixing, "
                   "reverberation and segmentation happen downstream in the notebook.\n",
                   "FSD50K asks commercial users to contact its maintainers; see archived source page.\n",
                   "This list does not clear the TTS model or the product software.\n"]
        for row in records:
            notices.append(f"- {row['path']}: {row['title']} — {row['author']}; "
                           f"{row['source_url']}; {row['license_url']}\n")
        (root / "ATTRIBUTION.md").write_text("\n".join(notices), encoding="utf-8")
        root.rename(prepared)
    return prepared
