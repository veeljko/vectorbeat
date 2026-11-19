import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread
from unittest.mock import Mock, patch

import requests

from mrvectorbeat.crawler.crawler import Crawler


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.pauses = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.pauses.append(seconds)
        self.now += seconds


class RequestSpacingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache = patch("mrvectorbeat.crawler.crawler.PAGE_CACHE_DIR", Path(directory.name))
        cache.start()
        self.addCleanup(cache.stop)
        self.clock = FakeClock()
        self.starts = []
        self.finishes = []
        for target, value in (
            ("mrvectorbeat.crawler.crawler.time.monotonic", self.clock.monotonic),
            ("mrvectorbeat.crawler.crawler.time.sleep", self.clock.sleep),
            ("mrvectorbeat.crawler.crawler.Crawler._last_request_finished", None),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def request(self, url, **kwargs):
        self.starts.append(self.clock.now)
        self.clock.now += 0.2
        self.finishes.append(self.clock.now)
        return Mock(status_code=200, text="HTML", headers={})

    def test_first_request_has_no_initial_or_trailing_pause(self):
        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=self.request):
            self.assertEqual(Crawler.fetch_page("https://example.com/"), (200, "HTML"))
        self.assertEqual(self.starts, [0])
        self.assertEqual(self.clock.pauses, [])
        self.assertEqual(self.clock.now, 0.2)

    def test_waits_only_remaining_time_between_requests(self):
        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=self.request):
            Crawler().fetch_page("https://example.com/page")
            self.clock.now += 0.4  # Time spent processing the previous page.
            Crawler().fetch_page("https://example.com/robots.txt")
        self.assertAlmostEqual(self.clock.pauses[0], 0.6)
        self.assertAlmostEqual(self.starts[1] - self.finishes[0], 1)
        self.assertAlmostEqual(self.clock.now, self.finishes[-1])

    def test_no_pause_if_enough_time_has_already_elapsed(self):
        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=self.request):
            Crawler.fetch_page("https://example.com/a")
            self.clock.now += 2
            Crawler.fetch_page("https://example.com/b")
        self.assertEqual(self.clock.pauses, [])
        self.assertAlmostEqual(self.starts[1] - self.finishes[0], 2)

    def test_failed_and_non_200_requests_also_count_toward_interval(self):
        def request(url, **kwargs):
            response = self.request(url, **kwargs)
            if url.endswith("error"):
                raise requests.ConnectionError("Simulated failure")
            if url.endswith("missing"):
                response.status_code = 404
            return response

        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=request), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(Crawler.fetch_page("https://example.com/error"), (None, None))
            self.assertEqual(Crawler.fetch_page("https://example.com/missing"), (404, None))
            self.assertEqual(Crawler.fetch_page("https://example.com/ok"), (200, "HTML"))
        for i in range(1, len(self.starts)):
            self.assertAlmostEqual(self.starts[i] - self.finishes[i - 1], 1)

    def test_redirects_are_spaced_as_separate_http_requests(self):
        def request(url, **kwargs):
            response = self.request(url, **kwargs)
            if url.endswith("original"):
                response.status_code = 302
                response.headers = {"Location": "/destination"}
            return response

        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=request) as get:
            self.assertEqual(Crawler.fetch_page("https://example.com/original"), (200, "HTML"))
        self.assertEqual([c.args[0] for c in get.call_args_list],
                         ["https://example.com/original", "https://example.com/destination"])
        self.assertTrue(all(c.kwargs["allow_redirects"] is False for c in get.call_args_list))
        self.assertAlmostEqual(self.starts[1] - self.finishes[0], 1)

    def test_redirect_loop_stops_with_request_error(self):
        def request(url, **kwargs):
            response = self.request(url, **kwargs)
            response.status_code = 302
            response.headers = {"Location": "/loop"}
            return response

        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=request) as get, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(Crawler.fetch_page("https://example.com/loop"), (None, None))
        self.assertEqual(get.call_count, 31)
        self.assertIn("TooManyRedirects", output.getvalue())

    def test_concurrent_instances_share_request_lock_and_interval(self):
        first_entered, release_first, second_started = Event(), Event(), Event()
        errors = []
        results = []

        def request(url, **kwargs):
            if url.endswith("first"):
                first_entered.set()
                if not release_first.wait(timeout=2):
                    raise AssertionError("First request was not released")
            return self.request(url, **kwargs)

        def fetch(url, started=None):
            if started is not None:
                started.set()
            try:
                results.append(Crawler().fetch_page(url))
            except Exception as error:
                errors.append(error)

        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=request) as get:
            first = Thread(target=fetch, args=("https://example.com/first",), daemon=True)
            second = Thread(target=fetch, args=("https://example.com/second", second_started), daemon=True)
            first.start()
            try:
                self.assertTrue(first_entered.wait(timeout=2))
                second.start()
                self.assertTrue(second_started.wait(timeout=2))
                self.assertEqual(get.call_count, 1)
            finally:
                release_first.set()
                first.join(timeout=2)
                if second.ident is not None:
                    second.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [(200, "HTML"), (200, "HTML")])
        self.assertAlmostEqual(self.starts[1] - self.finishes[0], 1)


if __name__ == "__main__":
    unittest.main()
