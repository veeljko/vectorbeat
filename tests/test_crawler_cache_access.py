import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, call, patch

from mrvectorbeat.crawler.crawler import Crawler, WIKI


class CacheFileAccessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache_dir = Path(directory.name)
        cache = patch("mrvectorbeat.crawler.crawler.PAGE_CACHE_DIR", self.cache_dir)
        cache.start()
        self.addCleanup(cache.stop)
        self.url = WIKI + "Highly_Suspect"
        self.path = Crawler._cache_path(self.url)

    def test_locked_replace_is_retried_without_repeating_http_request(self):
        replace = Path.replace
        attempts = 0

        def temporarily_locked(source, target):
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise PermissionError("Temporary Windows file lock")
            return replace(source, target)

        with patch.object(Path, "replace", autospec=True, side_effect=temporarily_locked), \
                patch("mrvectorbeat.crawler.crawler.time.sleep") as sleep, \
                patch("mrvectorbeat.crawler.crawler.REQUEST_INTERVAL_SECONDS", 0.0), \
                patch("mrvectorbeat.crawler.crawler.requests.Session.get",
                      return_value=Mock(status_code=200, text="<p>Music band</p>")) as get:
            self.assertEqual(Crawler.fetch_page(self.url), (200, "<p>Music band</p>"))
            self.assertEqual(Crawler.fetch_page(self.url), (200, "<p>Music band</p>"))
        get.assert_called_once()
        self.assertEqual(attempts, 3)
        self.assertEqual(sleep.call_args_list, [call(0.05), call(0.1)])
        self.assertFalse(list(self.cache_dir.glob("*.tmp")))

    @unittest.skipUnless(os.name == "nt", "Windows file sharing semantics")
    def test_real_windows_handle_lock_is_retried_after_reader_closes(self):
        Crawler._save_cached_page(self.url, "Old content")
        with self.path.open("rb") as reader:
            with patch("mrvectorbeat.crawler.crawler.time.sleep", side_effect=lambda _: reader.close()) as sleep:
                Crawler._save_cached_page(self.url, "New content")
        sleep.assert_called_once()
        self.assertEqual(Crawler._read_cached_page(self.url), ("New content", self.url))

    def test_persistent_failure_preserves_old_cache_and_reports_original_error(self):
        Crawler._save_cached_page(self.url, "Existing content")
        previous = self.path.read_bytes()
        denied = PermissionError("Persistent permission error")
        with patch.object(Path, "replace", side_effect=denied) as replace, \
                patch("mrvectorbeat.crawler.crawler.time.sleep") as sleep:
            with self.assertRaises(PermissionError) as error:
                Crawler._save_cached_page(self.url, "New content")
        self.assertIs(error.exception, denied)
        self.assertEqual(replace.call_count, 10)
        self.assertEqual(sleep.call_count, 9)
        self.assertLessEqual(max(args.args[0] for args in sleep.call_args_list), 1.0)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertFalse(list(self.cache_dir.glob("*.tmp")))

    def test_failed_cleanup_does_not_mask_original_replace_error(self):
        denied = PermissionError("Original replace failure")
        with patch.object(Path, "replace", side_effect=denied), \
                patch.object(Path, "unlink", side_effect=PermissionError("Cleanup failure")), \
                patch("mrvectorbeat.crawler.crawler.time.sleep"):
            with self.assertRaises(PermissionError) as error:
                Crawler._save_cached_page(self.url, "New content")
        self.assertIs(error.exception, denied)
        self.assertFalse(self.path.exists())

    def test_reader_waits_for_replacement_and_reads_complete_new_json(self):
        Crawler._save_cached_page(self.url, "Existing content")
        replacing, release, reading, read_finished = Event(), Event(), Event(), Event()
        replace = Path.replace

        def slow_replace(source, target):
            replacing.set()
            if not release.wait(timeout=3):
                raise AssertionError("Replacement not released")
            return replace(source, target)

        def read():
            reading.set()
            try:
                return Crawler._read_cached_page(self.url)
            finally:
                read_finished.set()

        with patch.object(Path, "replace", autospec=True, side_effect=slow_replace), \
                ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(Crawler._save_cached_page, self.url, "New content")
            try:
                self.assertTrue(replacing.wait(timeout=3))
                reader = pool.submit(read)
                self.assertTrue(reading.wait(timeout=3))
                self.assertFalse(read_finished.wait(timeout=0.05))
            finally:
                release.set()
            writer.result(timeout=3)
            self.assertEqual(reader.result(timeout=3), ("New content", self.url))
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["text"], "New content")

    def test_transient_read_lock_does_not_trigger_another_http_request(self):
        Crawler._save_cached_page(self.url, "Existing content")
        open_file = Path.open
        attempts = 0

        def temporarily_locked(path, *args, **kwargs):
            nonlocal attempts
            if path == self.path:
                attempts += 1
                if attempts == 1:
                    raise PermissionError("Temporary reader lock")
            return open_file(path, *args, **kwargs)

        with patch.object(Path, "open", autospec=True, side_effect=temporarily_locked), \
                patch("mrvectorbeat.crawler.crawler.time.sleep") as sleep, \
                patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=AssertionError("HTTP")):
            self.assertEqual(Crawler.fetch_page(self.url), (200, "Existing content"))
        self.assertEqual(attempts, 2)
        sleep.assert_called_once_with(0.05)

    def test_other_io_errors_propagate_without_retries_or_destroying_existing_file(self):
        Crawler._save_cached_page(self.url, "Existing content")
        previous = self.path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("Disk failure")), \
                patch("mrvectorbeat.crawler.crawler.time.sleep") as sleep:
            with self.assertRaisesRegex(OSError, "Disk failure"):
                Crawler._save_cached_page(self.url, "New content")
        sleep.assert_not_called()
        self.assertEqual(self.path.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
