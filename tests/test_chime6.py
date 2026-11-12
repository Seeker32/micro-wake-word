from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

from microwakeword.audio import chime6

HAVE_TF = importlib.util.find_spec("tensorflow") is not None


def wav_bytes(frames, rate=16000, channels=1, payload=None):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(payload or bytes(range(256)) * (frames * 2 * channels // 256)
                           + b"\0" * (frames * 2 * channels % 256))
    return buffer.getvalue()


def noisy_wav(frames):
    """Incompressible PCM, so a truncated stream really is truncated."""
    return wav_bytes(frames, payload=random.Random(0).randbytes(frames * 2))


def utterance(session, room, ref, start, end):
    return {"session_id": session, "location": room, "ref": ref, "speaker": "P01",
            "words": "hello", "start_time": f"00:00:{start:05.2f}",
            "end_time": f"00:00:{end:05.2f}"}


TRANSCRIPTS = {
    "dev": {"S02": [utterance("S02", "kitchen", "U02", 0, 1),
                    utterance("S02", "living", "U05", 1, 9)]},
    "eval": {"S01": [utterance("S01", "kitchen", "U02", 0, 2),
                     utterance("S01", "dining", "U06", 2, 3)]},
}


def tar_gz(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def fake_prepared(root):
    """A tiny verified `prepared/` tree, built the way prepare_chime6 writes it."""
    root = Path(root)
    rows = []
    for subset, split, session in (("dev", "validation", "S02"), ("eval", "testing", "S01")):
        target = root / "audio" / split
        target.mkdir(parents=True)
        rows += chime6.write_segments(
            io.BytesIO(noisy_wav(16000 * 20)), f"CHiME6_{subset}/audio/{split}/{session}_U05.CH1.wav",
            target, subset=subset, split=split,
            choice={"session": session, "room": "living", "array": "U05", "channel": "CH1"},
            segment_seconds=20, min_seconds=6)
        transcript = root / "transcriptions" / split / f"{session}.json"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text("[]")
        rows.append({"path": transcript.relative_to(root).as_posix(),
                     "sha256": chime6.digest(transcript), "split": split})
    (root / "manifest.json").write_text(json.dumps({"files": rows}))
    (root / "ATTRIBUTION.md").write_text("# CHiME-6 attribution\n")
    return root


class FakeStream(io.BytesIO):
    """Serves `payload[offset:]` like a Range request would."""

    def __init__(self, payload, offset=0):
        super().__init__(payload[offset:])
        self.length = len(payload)
        self.validator = '"fixture"'

    @property
    def offset(self):
        return self.tell()


class Chime6Tests(unittest.TestCase):
    def test_reference_array_selection(self):
        dev = chime6.select_arrays(TRANSCRIPTS["dev"], "busiest", "CH1")
        self.assertEqual([(c["room"], c["array"]) for c in dev], [("living", "U05")])
        every = chime6.select_arrays(TRANSCRIPTS["eval"], "all", "CH1")
        self.assertEqual([(c["room"], c["array"]) for c in every],
                         [("dining", "U06"), ("kitchen", "U02")])
        with self.assertRaisesRegex(ValueError, "no 'living' room"):
            chime6.select_arrays(TRANSCRIPTS["eval"], "living", "CH1")
        clash = {"S01": [utterance("S01", "kitchen", "U01", 0, 1),
                         utterance("S01", "kitchen", "U02", 1, 2)]}
        with self.assertRaisesRegex(ValueError, "several reference arrays"):
            chime6.select_arrays(clash, "all", "CH1")

    def test_segments_keep_pcm_and_drop_short_tail(self):
        payload = wav_bytes(16000 * 25)
        with tempfile.TemporaryDirectory() as directory:
            records = chime6.write_segments(
                io.BytesIO(payload), "CHiME6_dev/CHiME6/audio/dev/S02_U05.CH1.wav",
                Path(directory), subset="dev", split="validation",
                choice={"session": "S02", "room": "living", "array": "U05", "channel": "CH1"},
                segment_seconds=10, min_seconds=6)
            # 25 s -> 10 s + 10 s; the 5 s tail is shorter than 6 s and is dropped.
            self.assertEqual([r["duration_s"] for r in records], [10.0, 10.0])
            with wave.open(io.BytesIO(payload)) as source:
                original = source.readframes(16000 * 20)
            joined = b""
            for record in records:
                with wave.open(str(Path(directory) / Path(record["path"]).name)) as part:
                    joined += part.readframes(part.getnframes())
            self.assertEqual(joined, original)
            self.assertEqual({r["upstream_sha256"] for r in records},
                             {chime6.hashlib.sha256(payload).hexdigest()})
        with tempfile.TemporaryDirectory() as directory, \
                self.assertRaisesRegex(ValueError, "expected mono 16-bit 16 kHz"):
            chime6.write_segments(io.BytesIO(wav_bytes(16000 * 8, rate=48000)), "x.wav",
                                  Path(directory), subset="dev", split="validation",
                                  choice={"session": "S02", "room": "living",
                                          "array": "U05", "channel": "CH1"},
                                  segment_seconds=10, min_seconds=6)

    def test_page_snapshot_failure_only_warns(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(chime6, "download", side_effect=OSError("ssl eof")) as attempt, \
                patch.object(chime6, "sleep"):
            self.assertIsNone(chime6.snapshot("https://www.openslr.org/150/",
                                              Path(directory) / "SLR150.html"))
            self.assertEqual(attempt.call_count, 3)

    def test_archive_cache_resumes_the_download_after_an_interrupt(self):
        payload = bytes(range(256)) * 4000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CHiME6_dev.tar.gz.part"
            offsets = []

            def opener(url, offset, length, validator):
                offsets.append((offset, length, validator))
                return FakeStream(payload, offset)

            first = chime6.ResumableArchive("https://mirror/dev", path, opener=opener)
            self.assertEqual(first.read(CHUNK := 100_000), payload[:CHUNK])
            first.close()  # a KeyboardInterrupt keeps the cache for the next run

            second = chime6.ResumableArchive("https://mirror/dev", path, opener=opener)
            self.assertEqual(second.cached, CHUNK)
            self.assertEqual(second.read(), payload)
            self.assertEqual(second.downloaded, len(payload) - CHUNK)
            self.assertEqual([o[0] for o in offsets], [0, CHUNK])
            self.assertEqual(offsets[1][1:], (len(payload), '"fixture"'))
            second.close()
            self.assertTrue(path.is_file())

            second.discard()  # the archive is deleted once its subset is complete
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(path.name + ".json").exists())

    def test_prepare_resumes_after_ctrl_c_without_downloading_again(self):
        members = [(f"CHiME6_dev/CHiME6/audio/dev/S02_U05.CH1.wav", noisy_wav(16000 * 8))]
        payload = tar_gz(members)
        transcripts = tar_gz(
            [("transcriptions/transcriptions/dev/S02.json",
              json.dumps(TRANSCRIPTS["dev"]["S02"]).encode()),
              ("transcriptions/transcriptions/eval/S01.json",
              json.dumps(TRANSCRIPTS["eval"]["S01"]).encode())])
        eval_payload = tar_gz(
            [(f"CHiME6_eval/CHiME6/audio/eval/{name}", noisy_wav(16000 * seconds))
             for name, seconds in [("S01_U02.CH1.wav", 7), ("S01_U06.CH1.wav", 9)]])
        served, offsets = [], []

        class Interrupt(FakeStream):
            """Stops reading after a couple of chunks, like Ctrl+C mid-download."""

            def __init__(self, payload, offset=0):
                super().__init__(payload, offset)
                self.left = 2

            def read(self, size=-1):
                if not self.left:
                    raise KeyboardInterrupt
                self.left -= 1
                return super().read(min(size, 4096))

        def fake_open(subset, mirror, cache):
            url = f"{mirror}/CHiME6_{subset}.tar.gz"
            body = {"dev": payload, "eval": eval_payload}[subset]

            def opener(_, offset, length, validator):
                offsets.append(offset)
                if len(served) == 0 and subset == "dev":  # first run is cut short
                    served.append(True)
                    return Interrupt(body, offset)
                return FakeStream(body, offset)

            return url, chime6.ResumableArchive(
                url, Path(cache) / f"CHiME6_{subset}.tar.gz.part", opener=opener)

        def fake_download(url, path, md5=None):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(transcripts if path.name.endswith("transcriptions.tar.gz")
                             else f"snapshot of {url}".encode())
            return path

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(chime6, "open_archive", side_effect=fake_open), \
                patch.object(chime6, "download", side_effect=fake_download):
            with self.assertRaises(KeyboardInterrupt):
                chime6.prepare_chime6(Path(directory), dev_rooms="living",
                                      eval_rooms="all", segment_seconds=6, min_seconds=6)
            self.assertTrue((Path(directory) / "work" / "state.json").is_file())
            self.assertEqual(offsets, [0])  # the first attempt started from zero

            offsets.clear()
            prepared = chime6.prepare_chime6(Path(directory), dev_rooms="living",
                                             eval_rooms="all", segment_seconds=6, min_seconds=6)
            self.assertGreater(offsets[0], 0)  # continued, so nothing was downloaded twice
            self.assertEqual(sorted(p.relative_to(prepared).as_posix()
                                    for p in (prepared / "audio").rglob("*.wav")),
                             ["audio/testing/chime6_S01_U02_CH1_000.wav",
                              "audio/testing/chime6_S01_U06_CH1_000.wav",
                              "audio/validation/chime6_S02_U05_CH1_000.wav"])
            self.assertFalse((prepared / "state.json").exists())
            self.assertFalse((Path(directory) / "work").exists())
            self.assertFalse(list((Path(directory) / "downloads").glob("*.part*")))

    def test_prepare_streams_only_selected_arrays_and_reuses_offline(self):
        audio = {"dev": [("S02_U02.CH1.wav", 7), ("S02_U05.CH2.wav", 7),
                         ("S02_U05.CH1.wav", 8), ("S02_U06.CH1.wav", 7)],
                 "eval": [("S01_U02.CH1.wav", 7), ("S01_U06.CH1.wav", 9)]}
        archives = {subset: tar_gz(
            [(f"CHiME6_{subset}/CHiME6/audio/{subset}/{name}", wav_bytes(16000 * seconds))
             for name, seconds in members]) for subset, members in audio.items()}
        transcripts = tar_gz(
            [(f"transcriptions/transcriptions/{subset}/{session}.json",
              json.dumps(utterances).encode())
             for subset, sessions in TRANSCRIPTS.items()
             for session, utterances in sessions.items()])
        opened = []

        def fake_open(subset, mirror, cache):
            opened.append((subset, mirror))
            url = f"{mirror}/CHiME6_{subset}.tar.gz"
            payload = archives[subset]
            if mirror == chime6.MIRRORS[0] and subset == "eval":
                payload = payload[:200]  # truncated mirror
            return url, chime6.ResumableArchive(
                url, Path(cache) / f"CHiME6_{subset}.tar.gz.part",
                opener=lambda _, offset, length, validator: FakeStream(payload, offset))

        def fake_download(url, path, md5=None):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(transcripts if path.name.endswith("transcriptions.tar.gz")
                             else f"snapshot of {url}".encode())
            return path

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(chime6, "open_archive", side_effect=fake_open), \
                patch.object(chime6, "download", side_effect=fake_download):
            prepared = chime6.prepare_chime6(Path(directory), segment_seconds=6, min_seconds=6)
            names = sorted(p.relative_to(prepared).as_posix()
                           for p in (prepared / "audio").rglob("*.wav"))
            self.assertEqual(names, [
                "audio/testing/chime6_S01_U02_CH1_000.wav",
                "audio/testing/chime6_S01_U06_CH1_000.wav",
                "audio/validation/chime6_S02_U05_CH1_000.wav",
            ])
            self.assertEqual(opened, [("dev", chime6.MIRRORS[0]), ("eval", chime6.MIRRORS[0]),
                                      ("eval", chime6.MIRRORS[1])])
            self.assertEqual(sorted(p.name for p in prepared.iterdir()),
                             ["ATTRIBUTION.md", "audio", "manifest.csv", "manifest.json",
                              "sources", "transcriptions"])
            manifest = json.loads((prepared / "manifest.json").read_text())
            self.assertEqual(len(manifest["files"]), 5)  # 3 WAVs + 2 transcripts
            self.assertTrue(all(f["license_url"] == chime6.LICENSE_URL
                                for f in manifest["files"]))
            self.assertIn("Barker", (prepared / "ATTRIBUTION.md").read_text())

            with patch.object(chime6, "download", side_effect=AssertionError("offline")):
                self.assertEqual(chime6.prepare_chime6(Path(directory)), prepared)
            next((prepared / "audio").rglob("*.wav")).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                chime6.prepare_chime6(Path(directory))

    def test_restore_prepared_from_a_saved_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_prepared(root / "build" / "prepared")
            archive_dir = root / "data_v1" / "downloads"  # shared with the other datasets
            archive_dir.mkdir(parents=True)
            with zipfile.ZipFile(archive_dir / "chime6_v1.zip", "w") as archive:
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(source.parent).as_posix())

            self.assertEqual(chime6.find_chime6_source(archive_dir),
                             archive_dir / "chime6_v1.zip")
            restored = chime6.ensure_chime6(root / "out", source_dirs=[archive_dir])
            self.assertEqual(restored, root / "out" / "prepared")
            self.assertEqual(len(list((restored / "audio").rglob("*.wav"))), 2)
            self.assertFalse(list((root / "out").glob("unpacking-*")))
            # A verified copy is reused without touching the archive again.
            with patch.object(chime6, "restore_prepared",
                              side_effect=AssertionError("must not unpack")):
                self.assertEqual(chime6.ensure_chime6(root / "out", source_dirs=[archive_dir]),
                                 restored)
            # The unpacked directory itself is also a valid source.
            self.assertEqual(chime6.find_chime6_source(restored.parent), restored.parent)
            self.assertEqual(chime6.restore_prepared(restored.parent, root / "copy"),
                             restored)  # used in place, not copied again
            with self.assertRaisesRegex(ValueError, r"\.zip archive"):
                chime6.restore_prepared(root / "build" / "prepared" / "manifest.json",
                                        root / "bad")

    def test_restore_rejects_archives_outside_prepared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evil = root / "chime6_v1.zip"
            with zipfile.ZipFile(evil, "w") as archive:
                archive.writestr("prepared/../escaped.json", "{}")
                archive.writestr("prepared/manifest.json", "{}")
            with self.assertRaisesRegex(ValueError, "other than prepared"):
                chime6.restore_prepared(evil, root / "out")
            self.assertFalse((root / "escaped.json").exists())

    @unittest.skipUnless(HAVE_TF, "needs tensorflow for the micro frontend")
    def test_ambient_features_and_false_accepts_report(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = fake_prepared(root / "chime6_v1" / "prepared")
            features = chime6.prepare_ambient_features(prepared, root / "features" / "ambient")
            self.assertEqual(sorted(features["hours"]), sorted(chime6.AMBIENT_SPLITS))
            self.assertEqual({clip["split"] for clip in features["clips"]},
                             set(chime6.AMBIENT_SPLITS))
            self.assertEqual(len(features["clips"]), 2)
            self.assertGreater(features["clips"][0]["feature_frames"], 1900)
            self.assertAlmostEqual(sum(features["hours"].values()), 40 / 3600, places=4)
            again = chime6.prepare_ambient_features(prepared, root / "features" / "ambient")
            self.assertEqual(again, features)  # cached, not regenerated

            class FakeModel:
                def __init__(self, path, stride=None):
                    self.stride = stride

                def predict_spectrogram(self, track):
                    # One 0.99 burst a third of the way in; silence elsewhere.
                    probabilities = np.zeros(len(track), dtype=np.float32)
                    start = len(track) // 3
                    probabilities[start:start + 5] = 0.99
                    return probabilities

            with patch("microwakeword.inference.Model", FakeModel):
                report = chime6.evaluate_ambient_model(
                    "model.tflite", root / "features" / "ambient",
                    cutoffs=np.array([0.5, 0.995]))
            quiet, burst = report["results"]
            self.assertEqual(report["clip_count"], 1)  # one testing clip per fixture split
            self.assertEqual(quiet["clips_with_any_trigger"], 1)
            self.assertEqual(quiet["false_accepts"], 1.0)
            self.assertAlmostEqual(burst["false_accepts_per_hour"], 0.0)
            self.assertAlmostEqual(quiet["false_accepts_per_hour"], 1 / report["hours"], places=6)
            self.assertAlmostEqual(report["loudest_clips"][0]["peak_probability"], 0.99, places=6)
            self.assertEqual(report["loudest_clips"][0]["room"], "living")
            self.assertEqual(report["loudest_clips"][0]["split"], "testing")


if __name__ == "__main__":
    unittest.main()
