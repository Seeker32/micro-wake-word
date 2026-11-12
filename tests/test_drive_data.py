"""Check the Drive dataset copy: idempotent, atomic, and verified."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microwakeword.audio import drive_data


def write(path, payload=b"audio"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class CopyEntryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_copies_a_tree_and_skips_identical_files_next_time(self):
        source = self.root / "drive/data_v1/prepared"
        write(source / "noise/testing/a.wav")
        write(source / "speech/validation/b.flac")
        target = self.root / "run/data_v1/prepared"

        first = drive_data.copy_entry(source, target)
        self.assertEqual((first["copied"], first["skipped"], first["files"]), (2, 0, 2))
        self.assertEqual((target / "noise/testing/a.wav").read_bytes(), b"audio")

        second = drive_data.copy_entry(source, target)
        self.assertEqual((second["copied"], second["skipped"]), (0, 2))

    def test_changed_file_is_copied_again(self):
        source = write(self.root / "drive/x.wav", b"short")
        target = write(self.root / "run/x.wav", b"older content")
        stats = drive_data.copy_entry(source, target)
        self.assertEqual(stats["copied"], 1)
        self.assertEqual(target.read_bytes(), b"short")

    def test_identical_file_is_kept_as_is(self):
        source = write(self.root / "drive/x.wav", b"same bytes")
        target = write(self.root / "run/x.wav", b"same bytes")
        before = target.stat().st_mtime_ns
        stats = drive_data.copy_entry(source, target)
        self.assertEqual((stats["copied"], stats["skipped"]), (0, 1))
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_no_partial_files_are_left_behind(self):
        source = self.root / "drive/data"
        write(source / "a.wav")
        target = self.root / "run/data"
        drive_data.copy_entry(source, target)
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_finder_droppings_are_skipped(self):
        source = self.root / "drive/data"
        write(source / "a.wav")
        write(source / ".DS_Store", b"junk")
        write(source / "._a.wav", b"junk")
        stats = drive_data.copy_entry(source, self.root / "run/data")
        self.assertEqual(stats["files"], 1)

    def test_missing_source_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            drive_data.copy_entry(self.root / "nowhere", self.root / "run")


class SyncDatasetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.drive = self.root / "drive"
        self.run_root = self.root / "run"

    def test_known_entries_land_where_the_pipeline_expects_them(self):
        write(self.drive / "data_v1/prepared/noise/testing/a.wav")
        write(self.drive / "features_v1/negative/ambient/testing_ambient/noise_mmap/data.ninja")
        write(self.drive / "chime6_v1.zip", b"zip")
        report = drive_data.sync_datasets(self.drive, self.run_root)

        self.assertTrue((self.run_root / "data_v1/prepared/noise/testing/a.wav").is_file())
        self.assertTrue((self.run_root / "features_v1/negative/ambient/testing_ambient"
                         / "noise_mmap/data.ninja").is_file())
        # ensure_chime6() looks for the archive inside downloads/
        self.assertTrue((self.run_root / "data_v1/downloads/chime6_v1.zip").is_file())
        self.assertTrue(report["data_v1/prepared"]["present"])
        self.assertGreater(report["data_v1/prepared"]["copied"], 0)

    def test_trained_models_are_copied_with_the_dataset(self):
        model = "trained_models/hello_licensed_inputs/tflite_stream_state_internal_quant"
        write(self.drive / model / "stream_state_internal_quant.tflite", b"tflite")
        drive_data.sync_datasets(self.drive, self.run_root, verify=False)
        self.assertEqual((self.run_root / model / "stream_state_internal_quant.tflite")
                         .read_bytes(), b"tflite")

    def test_missing_entries_are_reported_not_raised(self):
        self.drive.mkdir()
        report = drive_data.sync_datasets(self.drive, self.run_root)
        self.assertEqual(len(report), len(drive_data.ENTRIES))
        self.assertFalse(any(stats["present"] for stats in report.values()))

    def test_second_sync_copies_nothing(self):
        write(self.drive / "data_v1/downloads/rirs_noises.zip", b"rir")
        first = drive_data.sync_datasets(self.drive, self.run_root, verify=False)
        second = drive_data.sync_datasets(self.drive, self.run_root, verify=False)
        self.assertEqual(first["data_v1/downloads"]["copied"], 1)
        self.assertEqual(second["data_v1/downloads"]["copied"], 0)
        self.assertEqual(second["data_v1/downloads"]["skipped"], 1)

    def test_bad_archive_is_deleted_and_reported(self):
        with patch.dict(drive_data.ARCHIVE_MD5, {"dev-clean.tar.gz": "0" * 32}):
            write(self.drive / "data_v1/downloads/dev-clean.tar.gz", b"corrupt")
            with self.assertRaises(ValueError) as caught:
                drive_data.sync_datasets(self.drive, self.run_root)
        self.assertIn("dev-clean.tar.gz", str(caught.exception))
        self.assertFalse((self.run_root / "data_v1/downloads/dev-clean.tar.gz").exists())

    def test_matching_archive_md5_is_accepted(self):
        import hashlib

        payload = b"archive bytes"
        published = hashlib.md5(payload).hexdigest()
        with patch.dict(drive_data.ARCHIVE_MD5, {"dev-clean.tar.gz": published}):
            write(self.drive / "data_v1/downloads/dev-clean.tar.gz", payload)
            report = drive_data.sync_datasets(self.drive, self.run_root)
        self.assertEqual(report["data_v1/downloads"]["verified"],
                         {"checked": {"dev-clean.tar.gz": True}})

    def test_architecture_layout_is_untouched_when_absent(self):
        write(self.drive / "data_v1/prepared/manifest.json", json.dumps({"files": []}).encode())
        drive_data.sync_datasets(self.drive, self.run_root)
        self.assertFalse((self.run_root / "features_v1").exists())
        self.assertFalse((self.run_root / "chime6_v1").exists())


class MountTests(unittest.TestCase):
    def test_returns_false_without_colab(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(drive_data.mount_drive(directory))

    def test_returns_true_when_drive_is_already_mounted(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "MyDrive").mkdir()
            self.assertTrue(drive_data.mount_drive(directory))


class ArchivesPresentTests(unittest.TestCase):
    def test_requires_every_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(drive_data.archives_present(root))
            for name in [*drive_data.ARCHIVE_MD5, "rirs_noises.zip"]:
                write(root / name)
            self.assertTrue(drive_data.archives_present(root))


if __name__ == "__main__":
    unittest.main()
