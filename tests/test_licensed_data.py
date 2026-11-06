from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from microwakeword.audio import licensed_data as data


class LicensedDataTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("zip"), "Info-ZIP is required by data preparation")
    def test_system_zip_joins_the_two_part_fsd_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "audio.wav"
            payload.write_bytes(bytes(range(256)) * 400)
            split = root / "audio.zip"
            joined = root / "joined.zip"
            subprocess.run(["zip", "-q", "-j", "-0", "-s", "64k", str(split), str(payload)], check=True)
            self.assertTrue(split.with_suffix(".z01").exists())
            self.assertFalse(split.with_suffix(".z02").exists())
            subprocess.run(["zip", "-q", "-s", "0", str(split), "--out", str(joined)], check=True)
            with zipfile.ZipFile(joined) as archive:
                self.assertEqual(archive.read("audio.wav"), payload.read_bytes())

    def test_license_allowlist_fails_closed(self):
        for url in ("http://creativecommons.org/publicdomain/zero/1.0/",
                    "https://creativecommons.org/licenses/by/3.0/",
                    "https://creativecommons.org/licenses/by/4.0/"):
            self.assertTrue(data.allowed_fsd_license(url), url)
        for url in (None, "", "CC BY", "https://creativecommons.org/licenses/by-nc/3.0/",
                    "https://creativecommons.org/licenses/by-nd/4.0/",
                    "https://creativecommons.org/licenses/by-sa/4.0/",
                    "https://creativecommons.org/licenses/sampling+/1.0/",
                    "https://creativecommons.org.evil/licenses/by/4.0/",
                    "https://creativecommons.org/licenses/by/4.0/?license=NC"):
            self.assertFalse(data.allowed_fsd_license(url), url)

    def test_preparation_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = Path(directory) / "fixtures"
            fixtures.mkdir()
            metadata = {}
            for index, split in enumerate(data.SPEECH_FILES, 1):
                uploader = next(f"reader-{n}" for n in range(1000)
                                if data.fsd_split(f"reader-{n}") == split)
                metadata[str(index)] = {
                    "uploader": uploader, "title": f"Sound {index}",
                    "license": "http://creativecommons.org/licenses/by/3.0/",
                }
            metadata["4"] = {"license": "http://creativecommons.org/licenses/by-nc/3.0/"}
            with zipfile.ZipFile(fixtures / "FSD50K.metadata.zip", "w") as archive:
                archive.writestr("FSD50K.metadata/eval_clips_info_FSD50K.json", json.dumps(metadata))
            with zipfile.ZipFile(fixtures / "FSD50K.doc.zip", "w") as archive:
                for filename in ("LICENSE-DATASET", "README.md"):
                    archive.writestr(f"FSD50K.doc/{filename}", "dataset attribution")
            with zipfile.ZipFile(fixtures / "FSD50K.eval_audio.zip", "w") as archive:
                for clip_id in metadata:
                    archive.writestr(f"FSD50K.eval_audio/{clip_id}.wav", f"audio-{clip_id}")
            (fixtures / "FSD50K.eval_audio.z01").write_bytes(b"multipart fixture")
            with zipfile.ZipFile(fixtures / "rirs_noises.zip", "w") as archive:
                archive.writestr("RIRS_NOISES/README", "RIRS_NOISES authors")
                archive.writestr("RIRS_NOISES/simulated_rirs/README", "Simulated RIR authors")
                archive.writestr("RIRS_NOISES/simulated_rirs/room/1.wav", b"simulated")
                archive.writestr("RIRS_NOISES/real_rirs/1.wav", b"exclude real RIR")
                archive.writestr("RIRS_NOISES/simulated_rirs/../../escape.wav", b"exclude traversal")
            for subset, _ in data.SPEECH_FILES.values():
                with tarfile.open(fixtures / f"{subset}.tar.gz", "w:gz") as archive:
                    for name in (f"LibriSpeech/{subset}/1/2/1-2-3.flac",
                                 "LibriSpeech/../../escape.flac"):
                        member = tarfile.TarInfo(name)
                        member.size = 5
                        archive.addfile(member, io.BytesIO(b"audio"))

            def fake_download(url, path, md5=None):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fixture = fixtures / path.name
                if fixture.exists():
                    shutil.copyfile(fixture, path)
                else:
                    path.write_text(f"source snapshot: {url}")
                return path

            def join_archive(command, check):
                shutil.copyfile(fixtures / "FSD50K.eval_audio.zip", command[-1])

            with patch.object(data, "download", side_effect=fake_download), \
                    patch.object(data.subprocess, "run", side_effect=join_archive):
                root = data.prepare_licensed_data(Path(directory) / "data")
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(manifest["files"]), 7)
            self.assertTrue(manifest["pending"])
            self.assertEqual(len(list(csv.DictReader(io.StringIO((root / "manifest.csv").read_text())))), 7)
            rejected = list(csv.DictReader(io.StringIO((root / "excluded_fsd.csv").read_text())))
            self.assertEqual([row["id"] for row in rejected], ["4"])
            self.assertFalse(list(root.rglob("escape*")))
            self.assertNotIn("real_rirs", (root / "manifest.json").read_text())
            self.assertIn("Sound 1", (root / "ATTRIBUTION.md").read_text())
            for row in manifest["files"]:
                self.assertEqual(row["sha256"], data.digest(root / row["path"]))
                self.assertIn(row["license_url"], (root / "ATTRIBUTION.md").read_text())
            with patch.object(data, "download", side_effect=AssertionError("cache must be offline")):
                self.assertEqual(data.prepare_licensed_data(root.parent), root)
            audio = next((root / "noise").rglob("*.wav"))
            audio.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Prepared data changed"):
                data.prepare_licensed_data(root.parent)

    def test_missing_approved_clip_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.zip"
            with zipfile.ZipFile(path, "w"):
                pass
            with zipfile.ZipFile(path) as archive:
                with self.assertRaises(KeyError):
                    data.select_fsd({"1": {"license": data.BY, "uploader": "author",
                                           "title": "missing"}}, archive, Path(directory), [], [])

    def test_bad_download_never_enters_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "audio.zip"
            source = io.BytesIO(b"truncated")
            source.headers = {}
            with patch.object(data, "urlopen", return_value=source):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    data.download("https://example.org/audio.zip", target, "0" * 32)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".zip.part").exists())

    def test_download_reports_progress_with_or_without_content_length(self):
        payload = b"x" * (2 * 1024 * 1024)
        for headers, progress in (
            ({"Content-Length": str(len(payload))}, "1.00 MiB / 2.00 MiB (50.0%)"),
            ({}, "1.00 MiB"),
        ):
            with self.subTest(headers=headers), tempfile.TemporaryDirectory() as directory:
                source = io.BytesIO(payload)
                source.headers = headers
                output = io.StringIO()
                with patch.object(data, "urlopen", return_value=source), \
                        patch.object(data, "monotonic", side_effect=[0, 5, 6]), \
                        redirect_stdout(output):
                    path = data.download("https://example.org/audio.zip", Path(directory) / "audio.zip")
                self.assertEqual(path.read_bytes(), payload)
                self.assertIn(f"  audio.zip: {progress}", output.getvalue().splitlines())
                self.assertIn("Downloaded audio.zip: 2.00 MiB", output.getvalue().splitlines())


if __name__ == "__main__":
    unittest.main()
