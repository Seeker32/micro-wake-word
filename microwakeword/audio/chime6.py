"""Keep a small far-field subset of CHiME-6 (OpenSLR SLR150) for ambient evaluation.

The dev (11 GB) and eval (12 GB) archives are decompressed as a stream and never
kept for good. Only channel CH1 of the reference Kinect array chosen for each room
is written, cut into fixed-length PCM segments; reading stops once every selected
array has been seen. The official session split is kept, so no session crosses
splits: dev -> validation (validation_ambient), eval -> testing (testing_ambient).

    python -m microwakeword.audio.chime6 --output-dir chime6_v1

An interrupted run is not thrown away: the archive prefix read so far stays in
`<output-dir>/downloads/*.tar.gz.part` (removed once its subset is complete), and
the arrays already cut, with their manifest rows, stay in `<output-dir>/work`,
which is renamed to `prepared/` only when everything is done. Rerun the same
command to continue; the download resumes with a Range request at the cached byte.

Defaults keep the busiest room of each dev session (about 4.5 h, sized for the
in-memory validation_ambient set) and all three rooms of each eval session.
"""

import argparse
from collections import Counter
import csv
import hashlib
from http.client import HTTPException
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from time import monotonic, sleep
from urllib.request import Request, urlopen
import wave
import zipfile
import zlib

from microwakeword.audio.licensed_data import digest, download


SOURCE = "https://www.openslr.org/150/"
MIRRORS = (
    "https://openslr.trmal.net/resources/150",
    "https://openslr.elda.org/resources/150",
)
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
CITATION = (
    "Barker, J., Watanabe, S., Vincent, E., Trmal, J. (2018) The Fifth 'CHiME' Speech "
    "Separation and Recognition Challenge: Dataset, Task and Baselines. Proc. Interspeech "
    "2018, 1561-1565, doi: 10.21437/Interspeech.2018-1768"
)
SUBSETS = {"dev": "validation", "eval": "testing"}  # official subset -> local split
ROOM_CHOICES = ("busiest", "all", "kitchen", "dining", "living")
SAMPLE_RATE = 16000
CHUNK = 1024 * 1024
AUDIO_NAME = re.compile(r"(S[0-9]{2})_(U0[1-6])\.(CH[1-4])\.wav")
# File manager droppings are not data: a Finder window must not fail verification.
OS_JUNK = re.compile(r"(?:\.DS_Store|Thumbs\.db|desktop\.ini|\._.*)")
# Damage in transit or in the archive: try the next mirror. Selection errors are fatal.
TRANSIENT = (OSError, EOFError, HTTPException, tarfile.TarError, zlib.error)
WORK = "work"  # resumable extraction; renamed to prepared/ when complete
STATE_NAME = "state.json"
FOLDER = "chime6_v1"  # default output directory, and the name of a saved archive


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_state(work, options):
    """Resume an interrupted run; discard work built with different options."""
    try:
        state = json.loads((work / STATE_NAME).read_text())
    except (OSError, ValueError):
        state = None
    if state is not None and state.get("options") == options:
        return state
    if work.exists():
        print(f"Discarding interrupted work with different options: {work}", flush=True)
        shutil.rmtree(work)
    work.mkdir(parents=True)
    state = {"options": options, "subsets": {}}
    save_state(work, state)
    return state


def save_state(work, state):
    _write_json(work / STATE_NAME, state)


class HTTPStream:
    """Sequential reader for one large HTTP object; resumes with Range after a drop."""

    def __init__(self, url, retries=8, label=None, offset=0, length=None, validator=None):
        self.url, self.retries, self.label = url, retries, label or url
        self.offset, self.response = offset, None
        self.length, self.validator = length, validator
        self.next_report = monotonic() + 10
        self.response = self._open()

    def _open(self):
        headers = {"Range": f"bytes={self.offset}-"} if self.offset else {}
        response = urlopen(Request(self.url, headers=headers), timeout=60)
        validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
        if self.offset:
            if self.length is None or not self.validator:
                response.close()
                raise OSError(f"Cannot resume {self.url} at byte {self.offset}")
            expected = f"bytes {self.offset}-{self.length - 1}/{self.length}"
            if (response.status != 206 or response.headers.get("Content-Range") != expected
                    or validator != self.validator):
                response.close()
                raise OSError(f"Cannot resume {self.url} at byte {self.offset}")
        else:
            length = response.headers.get("Content-Length", "").strip()
            self.length = int(length) if length.isdecimal() else None
            self.validator = validator
        return response

    def read(self, size=-1):
        failures = 0
        while True:
            try:
                if self.response is None:
                    self.response = self._open()
                chunk = self.response.read(size)
                if not chunk and self.length is not None and self.offset < self.length:
                    raise OSError("connection closed early")
            except (OSError, HTTPException) as error:
                self.close()
                failures += 1
                # Resuming needs a known length and a validator to detect a changed file.
                if failures > self.retries or self.length is None or not self.validator:
                    raise
                print(f"  {self.label}: {error}; resuming at byte {self.offset}", flush=True)
                sleep(min(60, 2 ** failures))
                continue
            self.offset += len(chunk)
            if (now := monotonic()) >= self.next_report:
                progress = f"{self.offset / 2**30:.2f} GiB"
                if self.length:
                    progress += f" / {self.length / 2**30:.2f} GiB ({self.offset / self.length:.1%})"
                print(f"  {self.label}: {progress}", flush=True)
                self.next_report = now + 10
            return chunk

    def close(self):
        if self.response is not None:
            self.response.close()
            self.response = None


class ResumableArchive:
    """Tar.gz bytes cached in a local file and extended from the network on demand.

    `tarfile` re-reads the cached prefix from disk on every run (cheap), while the
    download resumes with a Range request at the cached size (expensive to repeat).
    """

    def __init__(self, url, path, opener=None, label=None):
        self.url, self.path = url, Path(path)
        self.opener, self.label = opener or open_http, label or url
        self.state_path = self.path.with_name(self.path.name + ".json")
        self.length = self.validator = None
        self.cached = self.position = self.downloaded = 0
        self.next_save = monotonic() + 10
        self.file = self.stream = None
        self._restore()

    def _restore(self):
        """Trust a cache only when its sidecar proves it is a prefix of this URL."""
        try:
            saved = json.loads(self.state_path.read_text())
            size = self.path.stat().st_size
        except (OSError, ValueError):
            saved, size = None, 0
        if (saved and saved.get("url") == self.url and saved.get("length")
                and saved.get("validator") and 0 < int(saved.get("bytes") or 0) <= size):
            self.cached = int(saved["bytes"])
            self.length, self.validator = saved["length"], saved["validator"]
            print(f"  {self.label}: resuming at byte {self.cached:,}", flush=True)
        else:
            self.path.unlink(missing_ok=True)
            self.state_path.unlink(missing_ok=True)

    def _save(self):
        _write_json(self.state_path, {"url": self.url, "bytes": self.cached,
                                      "length": self.length, "validator": self.validator})

    def _start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.file = self.path.open("r+b")
        self.file.truncate(self.cached)  # drop bytes the sidecar does not cover
        self.position = 0

    def read(self, size=-1):
        if size is None or size < 0:
            return b"".join(iter(lambda: self.read(CHUNK), b""))
        if size == 0:
            return b""
        if self.file is None:
            self._start()
        if self.position < self.cached:
            self.file.seek(self.position)
            chunk = self.file.read(min(size, self.cached - self.position))
        else:
            chunk = self._fetch(size)
        self.position += len(chunk)
        return chunk

    def _fetch(self, size):
        if self.stream is None:
            if self.length is not None and self.cached >= self.length:
                return b""  # the cached file already holds the whole archive
            self.stream = self.opener(self.url, self.cached, self.length, self.validator)
            self.length = self.stream.length or self.length
            self.validator = self.stream.validator or self.validator
            self._save()
        chunk = self.stream.read(size)
        if chunk:
            self.file.seek(self.cached)
            self.file.write(chunk)
            self.file.flush()
            self.cached += len(chunk)
            self.downloaded += len(chunk)
            if (now := monotonic()) >= self.next_save:
                self._save()
                self.next_save = now + 10
        return chunk

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.cached:
            self._save()

    def discard(self):
        """Forget the cache: the bytes are no longer a prefix of this archive."""
        self.close()
        self.path.unlink(missing_ok=True)
        self.state_path.unlink(missing_ok=True)
        self.cached = self.position = self.downloaded = 0


class _HashingReader:
    """Non-seekable reader, so `wave` reads every byte through the hash."""

    def __init__(self, source):
        self.source, self.hash = source, hashlib.sha256()

    def read(self, size=-1):
        chunk = self.source.read(size)
        self.hash.update(chunk)
        return chunk


def open_http(url, offset, length, validator):
    return HTTPStream(url, label=PurePosixPath(url).name, offset=offset,
                      length=length, validator=validator)


def open_archive(subset, mirror, cache):
    url = f"{mirror}/CHiME6_{subset}.tar.gz"
    return url, ResumableArchive(url, Path(cache) / f"CHiME6_{subset}.tar.gz.part",
                                 label=f"CHiME6_{subset}.tar.gz")


def fetch(filename, path):
    """Small files: try each mirror; `download` keeps partial files out of the cache."""
    errors = []
    for mirror in MIRRORS:
        try:
            return download(f"{mirror}/{filename}", path)
        except TRANSIENT as error:
            errors.append(f"{mirror}: {error}")
    raise OSError(f"Cannot download {filename}: " + "; ".join(errors))


def snapshot(url, path, attempts=3):
    """Keep a copy of the dataset page: provenance only, so failures only warn."""
    for attempt in range(1, attempts + 1):
        try:
            return download(url, path)
        except TRANSIENT as error:
            if attempt == attempts:
                print(f"  cannot snapshot {url}: {error}", flush=True)
                return None
            print(f"  snapshot of {url} failed ({error}); retrying", flush=True)
            sleep(2 ** attempt)


def _seconds(value):
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_transcripts(archive_path, subset):
    """Return {session: utterances} for one official subset."""
    prefix = ("transcriptions", "transcriptions", subset)
    transcripts = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (member.isfile() and path.parts[:3] == prefix and len(path.parts) == 4
                    and re.fullmatch(r"S[0-9]{2}\.json", path.name)):
                transcripts[path.stem] = json.load(archive.extractfile(member))
    if not transcripts:
        raise ValueError(f"No {subset} transcriptions in {archive_path}")
    return transcripts


def select_arrays(transcripts, rooms, channel):
    """Pick each room's reference array (the `ref` field), as named by the organisers."""
    selected = []
    for session, utterances in sorted(transcripts.items()):
        arrays, speech = {}, Counter()
        for utterance in utterances:
            arrays.setdefault(utterance["location"], set()).add(utterance["ref"])
            speech[utterance["location"]] += (
                _seconds(utterance["end_time"]) - _seconds(utterance["start_time"]))
        for room, refs in arrays.items():
            if len(refs) != 1:
                raise ValueError(f"{session} {room} has several reference arrays: {sorted(refs)}")
        if rooms == "busiest":
            chosen = [max(sorted(speech), key=speech.get)]  # most transcribed speech
        elif rooms == "all":
            chosen = sorted(arrays)
        elif rooms in arrays:
            chosen = [rooms]
        else:
            raise ValueError(f"{session} has no {rooms!r} room; found {sorted(arrays)}")
        for room in chosen:
            array = next(iter(arrays[room]))
            if not AUDIO_NAME.fullmatch(f"{session}_{array}.{channel}.wav"):
                raise ValueError(f"Unexpected array name {array!r} in {session}")
            selected.append({"session": session, "room": room, "array": array,
                             "channel": channel})
    return selected


def write_segments(source, member_name, target_dir, *, subset, split, choice,
                   segment_seconds, min_seconds):
    """Copy one upstream WAV into fixed-length WAVs; PCM samples are not altered."""
    reader = _HashingReader(source)
    records = []
    with wave.open(reader, "rb") as wav:
        params = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        if params != (1, 2, SAMPLE_RATE):
            raise ValueError(f"{member_name}: expected mono 16-bit 16 kHz, got {params}")
        total = wav.getnframes()
        step, minimum = segment_seconds * SAMPLE_RATE, min_seconds * SAMPLE_RATE
        for index, start in enumerate(range(0, total, step)):
            count = min(step, total - start)
            pcm = wav.readframes(count)
            if len(pcm) != count * 2:
                raise EOFError(f"{member_name}: audio ended after {start} frames")
            if count < minimum:
                continue  # too short for the >=6 s ambient clips
            name = (f"chime6_{choice['session']}_{choice['array']}_{choice['channel']}"
                    f"_{index:03d}.wav")
            path = target_dir / name
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(SAMPLE_RATE)
                output.writeframes(pcm)
            end = start + count
            records.append({
                "path": f"audio/{split}/{name}", "sha256": digest(path),
                "dataset": "CHiME-6 (OpenSLR SLR150)",
                "version": f"SLR150 (CC BY-SA 4.0 re-issue, 2024), {subset}",
                "upstream_path": member_name, "upstream_sha256": None,
                "source_url": SOURCE, "license_url": LICENSE_URL,
                "author": "J. Barker, S. Watanabe, E. Vincent, J. Trmal (CHiME-5/6)",
                "title": (f"{choice['session']} {choice['room']} {choice['array']} "
                          f"{choice['channel']}, {start / SAMPLE_RATE:.0f}-{end / SAMPLE_RATE:.0f} s"),
                "split": split,
                "modifications": f"samples [{start}, {end}) of the upstream WAV; "
                                 "PCM bytes unchanged, new WAV header",
                "session": choice["session"], "room": choice["room"],
                "array": choice["array"], "channel": choice["channel"],
                "start_s": start / SAMPLE_RATE, "duration_s": count / SAMPLE_RATE,
            })
    while reader.read(CHUNK):  # hash any chunks after `data`
        pass
    for record in records:
        record["upstream_sha256"] = reader.hash.hexdigest()
    if not records:
        raise ValueError(f"{member_name} is shorter than {min_seconds} s")
    return records


def stream_subset(subset, choices, work, cache, state, *, segment_seconds, min_seconds):
    """Read one archive until every chosen array is copied; try mirrors in order.

    Each finished array is recorded in `state` right away, so an interrupted run
    keeps both its WAVs and their manifest rows, and the archive prefix left in
    `cache` keeps the bytes already downloaded.
    """
    split = SUBSETS[subset]
    entry = state["subsets"].setdefault(subset, {"split": split, "arrays": {}})
    done = entry["arrays"]
    wanted = {f"{c['session']}_{c['array']}.{c['channel']}.wav": c for c in choices}
    target_dir = work / "audio" / split
    target_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for mirror in MIRRORS:
        url, stream = open_archive(subset, mirror, cache)
        try:
            print(f"Streaming {url} for {sorted(set(wanted) - set(done))}", flush=True)
            found = set()
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    if path.name not in wanted or not member.isfile():
                        continue
                    if (".." in path.parts or path.parts[-3:-1] != ("audio", subset)
                            or path.name in found):
                        raise ValueError(f"Unexpected archive member: {member.name}")
                    if path.name not in done:
                        print(f"  keeping {member.name} ({member.size / 2**20:.0f} MiB)",
                              flush=True)
                        done[path.name] = write_segments(
                            archive.extractfile(member), member.name, target_dir,
                            subset=subset, split=split, choice=wanted[path.name],
                            segment_seconds=segment_seconds, min_seconds=min_seconds)
                        entry["archive"] = None  # not finished yet
                        save_state(work, state)
                    found.add(path.name)
                    if found == set(wanted):
                        break  # stop downloading; the rest of the archive is not needed
            if missing := sorted(set(wanted) - found):
                raise ValueError(f"{url} does not contain {missing}")
            entry["archive"] = {
                "url": url, "filename": f"CHiME6_{subset}.tar.gz",
                "content_length": stream.length, "validator": stream.validator,
                "compressed_bytes_downloaded": stream.downloaded,
                "compressed_bytes_read": stream.position,
                "note": "streamed through a resume cache, not kept; SLR150 publishes "
                        "no checksum, upstream_sha256 covers each kept WAV",
            }
            stream.discard()  # the rest of this subset is already on disk
            save_state(work, state)
            return entry
        except TRANSIENT as error:
            # The cache may hold bytes that no longer parse, so start this subset over.
            errors.append(f"{url}: {error}")
            print(f"  failed: {error}", flush=True)
            stream.discard()
        finally:
            stream.close()
    raise OSError(f"All mirrors failed for CHiME6_{subset}.tar.gz: " + "; ".join(errors))


def prepared_files(root):
    """Every file the manifest is expected to list, relative to `root`."""
    return {p.relative_to(root).as_posix() for folder in ("audio", "transcriptions")
            for p in (root / folder).rglob("*") if p.is_file() and not OS_JUNK.match(p.name)}


def verify_prepared(prepared):
    records = json.loads((prepared / "manifest.json").read_text())["files"]
    expected = {row["path"] for row in records}
    if (prepared_files(prepared) != expected
            or any(digest(prepared / r["path"]) != r["sha256"] for r in records)):
        raise ValueError(f"Prepared CHiME-6 data changed; use a fresh directory: {prepared}")
    return prepared


def prepare_chime6(output_dir=FOLDER, *, dev_rooms="busiest", eval_rooms="all",
                   channel="CH1", segment_seconds=600, min_seconds=6):
    """Write `<output_dir>/prepared/audio/{validation,testing}/*.wav` plus audit files.

    A finished directory is verified against its manifest and reused offline; an
    interrupted one resumes from `<output_dir>/work` and the archive cache.
    """
    if segment_seconds < min_seconds or min_seconds <= 0:
        raise ValueError("Need 0 < min_seconds <= segment_seconds")
    for rooms in (dev_rooms, eval_rooms):
        if rooms not in ROOM_CHOICES:
            raise ValueError(f"rooms must be one of {ROOM_CHOICES}, got {rooms!r}")
    if not re.fullmatch(r"CH[1-4]", channel):
        raise ValueError(f"channel must be CH1-CH4, got {channel!r}")
    output_dir = Path(output_dir)
    prepared = output_dir / "prepared"
    if prepared.exists():
        return verify_prepared(prepared)

    options = {"dev_rooms": dev_rooms, "eval_rooms": eval_rooms, "channel": channel,
               "segment_seconds": segment_seconds, "min_seconds": min_seconds}
    work = output_dir / WORK
    state = load_state(work, options)
    downloads = output_dir / "downloads"
    transcripts_archive = fetch("CHiME6_transcriptions.tar.gz",
                                downloads / "CHiME6_transcriptions.tar.gz")
    selections = {subset: select_arrays(read_transcripts(transcripts_archive, subset),
                                        options[f"{subset}_rooms"], channel)
                  for subset in SUBSETS}

    sources = work / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    fetch("LICENSE.txt", sources / "SLR150-LICENSE.txt")  # required, so fail early
    records = []
    archives = [{"url": f"{MIRRORS[0]}/CHiME6_transcriptions.tar.gz",
                 "filename": "CHiME6_transcriptions.tar.gz",
                 "sha256": digest(transcripts_archive)}]

    # Transcripts of kept sessions, for checking what was said at a false accept.
    with tarfile.open(transcripts_archive, "r:gz") as archive:
        for subset, choices in selections.items():
            split = SUBSETS[subset]
            for session in sorted({c["session"] for c in choices}):
                member = f"transcriptions/transcriptions/{subset}/{session}.json"
                path = work / "transcriptions" / split / f"{session}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
                records.append({
                    "path": path.relative_to(work).as_posix(), "sha256": digest(path),
                    "dataset": "CHiME-6 (OpenSLR SLR150)", "upstream_path": member,
                    "source_url": SOURCE, "license_url": LICENSE_URL, "split": split,
                    "modifications": "none (original bytes)",
                })

    for subset, choices in selections.items():
        wanted = [f"{c['session']}_{c['array']}.{c['channel']}.wav" for c in choices]
        entry = state["subsets"].get(subset, {})
        if not (entry.get("archive") and all(n in entry.get("arrays", {}) for n in wanted)):
            entry = stream_subset(subset, choices, work, downloads, state,
                                  segment_seconds=segment_seconds, min_seconds=min_seconds)
        records += [row for name in sorted(entry["arrays"]) for row in entry["arrays"][name]]
        archives.append(entry["archive"])
        save_state(work, state)

    page = snapshot(SOURCE, sources / "SLR150.html")
    manifest = {
        "scope": "CHiME-6 far-field ambient evaluation audio (no wake word present)",
        "policy": "official sessions: dev -> validation, eval -> testing; one channel "
                  "of each room's reference array; CC BY-SA 4.0 (ShareAlike applies "
                  "if derived data is distributed)",
        "selection": {"dev_rooms": dev_rooms, "eval_rooms": eval_rooms,
                      "channel": channel, "segment_seconds": segment_seconds,
                      "min_seconds": min_seconds,
                      "arrays": {SUBSETS[s]: c for s, c in selections.items()}},
        "citation": CITATION, "archives": archives,
        "source_page": None if page is None else {"url": SOURCE,
                                                  "path": page.relative_to(work).as_posix(),
                                                  "sha256": digest(page)},
        "license_documents": [{"license_url": LICENSE_URL,
                               "path": "sources/SLR150-LICENSE.txt",
                               "sha256": digest(sources / "SLR150-LICENSE.txt")}],
        "files": records,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    audio = [r for r in records if r["path"].startswith("audio/")]
    with (work / "manifest.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(audio[0]))
        writer.writeheader()
        writer.writerows(audio)
    hours = Counter()
    for row in audio:
        hours[row["split"]] += row["duration_s"] / 3600
    (work / "ATTRIBUTION.md").write_text(
        "# CHiME-6 attribution\n\n"
        f"Audio from CHiME-6 (OpenSLR SLR150, {SOURCE}), licensed {LICENSE_URL}.\n\n"
        f"Cite: {CITATION}\n\n"
        "Changes: one channel per selected array, cut into consecutive segments; "
        "sample values unchanged. Any distributed derivative must also be CC BY-SA 4.0.\n\n"
        + "".join(f"- {split}: {h:.2f} h\n" for split, h in sorted(hours.items())),
        encoding="utf-8")
    if (stray := prepared_files(work) ^ {row["path"] for row in records}):
        raise ValueError(f"{work} holds files the manifest does not list, remove it and "
                         f"retry: {sorted(stray)}")
    (work / STATE_NAME).unlink(missing_ok=True)  # resume notes are not part of the data
    work.rename(prepared)
    for split, h in sorted(hours.items()):
        print(f"CHiME-6 {split}: {h:.2f} h in {prepared / 'audio' / split}")
    return prepared


def _zip_members(archive):
    """Yield `(member, path, is_directory)`, rejecting anything outside prepared/."""
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("prepared",):
            raise ValueError(f"Archive holds something other than prepared/: {name}")
        yield member, path, member.is_dir()


def restore_prepared(source, output_dir):
    """Unpack a saved `chime6_v1.zip`, or use an already unpacked `prepared/` directory.

    Everything is verified against the manifest before it is published, so a truncated
    or altered upload fails instead of silently poisoning the evaluation set.
    """
    source, output_dir = Path(source).expanduser(), Path(output_dir)
    if source.is_dir():
        for directory in (source, source / "prepared"):
            if (directory / "manifest.json").is_file():
                return verify_prepared(directory)
        raise ValueError(f"No manifest.json in {source} or {source / 'prepared'}")
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip archive or an unpacked directory: {source}")
    target = output_dir / "prepared"
    if (target / "manifest.json").is_file():
        return verify_prepared(target)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Unpacking {source.name} into {output_dir}", flush=True)
    with tempfile.TemporaryDirectory(prefix="unpacking-", dir=output_dir) as staging:
        staged = Path(staging) / "prepared"
        with zipfile.ZipFile(source) as archive:
            for member, path, directory in _zip_members(archive):
                destination = Path(staging).joinpath(*path.parts)
                if directory:
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as handle, destination.open("xb") as output:
                    shutil.copyfileobj(handle, output)
        verify_prepared(staged)  # publish only a complete, unaltered copy
        staged.rename(target)
    return target


def find_chime6_source(*directories):
    """First `chime6_v1` directory or `chime6_v1.zip` found in the given places."""
    for directory in directories:
        if directory is None:
            continue
        directory = Path(directory).expanduser()
        for candidate in (directory, directory / FOLDER, directory / f"{FOLDER}.zip"):
            if candidate.is_file():
                return candidate
            if candidate.is_dir() and (
                    (candidate / "manifest.json").is_file()
                    or (candidate / "prepared").is_dir()):
                return candidate
    return None


def ensure_chime6(output_dir=FOLDER, source=None, *, source_dirs=()):
    """Reuse `prepared/`, unpack a saved `chime6_v1.zip`, or stream from OpenSLR.

    `source` may point at an unpacked directory or the archive itself; `source_dirs`
    are searched for `chime6_v1`/`chime6_v1.zip`, so the saved copy can sit next to
    the other dataset archives.
    """
    prepared = Path(output_dir) / "prepared"
    if (prepared / "manifest.json").is_file():
        return verify_prepared(prepared)
    found = Path(source).expanduser() if source else find_chime6_source(*source_dirs)
    if found is None:
        print(f"No CHiME-6 copy in {[str(Path(d)) for d in source_dirs if d]}; "
              f"streaming {SOURCE} (resumable)", flush=True)
        return prepare_chime6(output_dir)
    print(f"Restoring CHiME-6 from {found}", flush=True)
    return restore_prepared(found, output_dir)


AMBIENT_SPLITS = ("validation", "testing")  # local split -> <split>_ambient features


def _read_clip(path):
    """Return 16 kHz mono PCM16 samples as float32, rejecting anything else."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        params = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        frames = wav.getnframes()
        pcm = wav.readframes(frames)
    if params != (1, 2, SAMPLE_RATE) or len(pcm) != frames * 2:
        raise ValueError(f"{path}: expected complete mono 16-bit 16 kHz, got {params}")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def prepare_ambient_features(prepared, features_dir, *, step_ms=10, splits=AMBIENT_SPLITS):
    """Write `<features_dir>/<split>_ambient/chime6_mmap` tracks for evaluation.

    Tracks stay whole (no repetition, no 5 s chunking), so the training code counts
    false accepts per hour of real room audio. The manifest is re-verified first and
    the resulting clips are indexed in `feature_sources.json` for attribution.
    `splits` limits the build, e.g. to ("testing",) when `validation_ambient` would
    not fit in the training machine's memory.
    """
    splits = tuple(splits)
    if not splits or set(splits) - set(AMBIENT_SPLITS):
        raise ValueError(f"splits must be a subset of {AMBIENT_SPLITS}, got {splits!r}")
    import numpy as np
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.audio_utils import generate_features_for_clip

    prepared, features_dir = verify_prepared(Path(prepared)), Path(features_dir)
    clips = sorted((row for row in json.loads((prepared / "manifest.json").read_text())["files"]
                    if row["path"].startswith("audio/")), key=lambda row: row["path"])
    if not clips:
        raise ValueError(f"No audio in {prepared}")
    signature = {"prepared_manifest_sha256": digest(prepared / "manifest.json"),
                 "step_ms": step_ms, "window_ms": 30, "splits": list(splits),
                 "clips": [row["path"] for row in clips]}
    lineage_path = features_dir / "feature_sources.json"
    if lineage_path.is_file():
        saved = json.loads(lineage_path.read_text())
        counts = Counter(row["split"] for row in clips)
        if saved.get("signature") == signature and all(
                len(RaggedMmap(features_dir / f"{split}_ambient" / "chime6_mmap")) == counts[split]
                for split in splits):
            print(f"Reusing CHiME-6 ambient features in {features_dir}", flush=True)
            return saved

    frames = {}

    def tracks(indices):
        for index in indices:
            row = clips[index]
            feature = generate_features_for_clip(_read_clip(prepared / row["path"]), step_ms=step_ms)
            if feature.ndim != 2 or feature.shape[1] != 40 or not len(feature):
                raise ValueError(f"Invalid CHiME-6 features for {row['path']}: {feature.shape}")
            frames[row["path"]] = int(feature.shape[0])
            yield feature.astype(np.float32)

    features_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ambient-", dir=features_dir.parent) as staging:
        staged = Path(staging) / "features"
        for split in splits:
            indices = [i for i, row in enumerate(clips) if row["split"] == split]
            if not indices:
                raise ValueError(f"No {split} audio for ambient evaluation in {prepared}")
            output = staged / f"{split}_ambient" / "chime6_mmap"
            output.parent.mkdir(parents=True)
            RaggedMmap.from_generator(str(output), tracks(indices), batch_size=1, verbose=False)
        hours = Counter()
        for row in clips:
            hours[row["split"]] += row["duration_s"] / 3600
        lineage = {
            "dataset": "CHiME-6 (OpenSLR SLR150), real far-field home recordings",
            "prepared": str(prepared.resolve()),
            "prepared_manifest": str((prepared / "manifest.json").resolve()),
            "signature": signature,
            "processing": f"{step_ms} ms step, 30 ms window, 40 mel bins; whole clips, "
                          "no repetition and no chunking; ambient evaluation only",
            "hours": {split: hours[split] for split in splits},
            "clips": [{key: row[key] for key in ("path", "split", "session", "room", "array",
                                                 "channel", "duration_s", "start_s", "sha256")}
                      | {"feature_frames": frames.get(row["path"])} for row in clips
                      if row["split"] in splits],
            "license_url": LICENSE_URL,
            "attribution": str((prepared / "ATTRIBUTION.md").resolve()),
        }
        (staged / "feature_sources.json").write_text(
            json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
        if features_dir.exists():
            print(f"Replacing stale features in {features_dir}", flush=True)
            shutil.rmtree(features_dir)  # derived, and the signature no longer matches
        staged.rename(features_dir)
    return lineage


def evaluate_ambient_model(model_path, features_dir, *, split="testing", cutoffs=None,
                           sliding_window_length=5, ignore_slices_after_accept=25, stride=3):
    """False accepts per hour on one CHiME-6 split, with the clips that triggered.

    Uses the same sliding window, cooldown and per-hour convention as
    `microwakeword.test.tflite_streaming_model_roc`, so numbers are comparable to the
    pooled `testing_ambient` report while staying attributable to CHiME-6 alone.
    """
    import numpy as np
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.inference import Model
    from microwakeword.test import compute_false_accepts_per_hour

    features_dir = Path(features_dir)
    lineage = json.loads((features_dir / "feature_sources.json").read_text())
    tracks = RaggedMmap(features_dir / f"{split}_ambient" / "chime6_mmap")
    clips = [clip for clip in lineage["clips"] if clip["split"] == split]
    if len(tracks) != len(clips):
        raise ValueError(f"Cached {split} features do not match the manifest; rebuild them")
    cutoffs = np.arange(0, 1.01, 0.01) if cutoffs is None else np.asarray(cutoffs, dtype=float)
    window = np.lib.stride_tricks.sliding_window_view
    probabilities, peaks = [], []
    for clip, track in zip(clips, tracks):
        # A fresh interpreter per recording: no streaming state may leak between rooms.
        model = Model(str(model_path), stride=stride)
        streaming = np.asarray(model.predict_spectrogram(track))
        if len(streaming) < sliding_window_length:
            raise ValueError(f"{clip['path']} is too short to evaluate")
        moving = window(streaming, sliding_window_length).mean(axis=-1)
        probabilities.append(moving)
        peaks.append(float(moving.max()))
    hours = sum(len(moving) * stride * 0.01 for moving in probabilities) / 3600
    faph = compute_false_accepts_per_hour(
        probabilities, cutoffs, ignore_slices_after_accept, stride=stride, step_s=0.01)
    triggered = [sum(peak > cutoff for peak in peaks) for cutoff in cutoffs]
    order = sorted(range(len(clips)), key=lambda index: -peaks[index])
    return {
        "model": str(model_path), "features_dir": str(features_dir), "split": split,
        "hours": hours, "clip_count": len(clips), "stride": stride, "step_ms": 10,
        "sliding_window_length": sliding_window_length,
        "ignore_slices_after_accept": ignore_slices_after_accept,
        "metric": "false accepts per hour on CHiME-6 real far-field home audio; "
                  "same convention as testing_ambient",
        "results": [{"cutoff": float(cutoff), "false_accepts_per_hour": float(rate),
                     "false_accepts": float(rate) * hours,
                     "clips_with_any_trigger": int(count)}
                    for cutoff, rate, count in zip(cutoffs, faph, triggered)],
        "loudest_clips": [dict(clips[index], peak_probability=peaks[index])
                          for index in order[:10]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default=FOLDER)
    parser.add_argument("--dev-rooms", default="busiest", choices=ROOM_CHOICES,
                        help="rooms kept per dev session (validation_ambient is held in RAM)")
    parser.add_argument("--eval-rooms", default="all", choices=ROOM_CHOICES)
    parser.add_argument("--channel", default="CH1")
    parser.add_argument("--segment-seconds", type=int, default=600)
    parser.add_argument("--min-seconds", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        prepare_chime6(args.output_dir, dev_rooms=args.dev_rooms,
                       eval_rooms=args.eval_rooms, channel=args.channel,
                       segment_seconds=args.segment_seconds,
                       min_seconds=args.min_seconds)
    except KeyboardInterrupt:
        print("\nInterrupted; rerun the same command to resume where it stopped.",
              flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
