import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from itertools import count
from pathlib import Path
from threading import Event, Thread
from unittest.mock import Mock, patch

import requests

from mrvectorbeat.crawler.crawler import Crawler, WIKI
from mrvectorbeat.crawler.generate_seed import WikipediaSeedResolver


class CrawlerCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_dir = Path(self.directory.name)
        for target, value in (
            ("mrvectorbeat.crawler.crawler.PAGE_CACHE_DIR", self.cache_dir),
            ("mrvectorbeat.crawler.crawler.Crawler._last_request_finished", None),
            ("mrvectorbeat.crawler.crawler.time.monotonic", lambda ticks=count(0, 2): next(ticks)),
            ("mrvectorbeat.crawler.crawler.time.sleep", Mock(side_effect=AssertionError("Unexpected pause"))),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_saved_page_survives_a_new_python_process_without_http(self):
        url = WIKI + "Bj%C3%B6rk"
        html = "<p>Bj\u00f6rk \u266b</p>"
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get",
                   return_value=Mock(status_code=200, text=html)) as get:
            self.assertEqual(Crawler().fetch_page(url), (200, html))
        get.assert_called_once()
        cached = json.loads(Crawler._cache_path(url).read_text(encoding="utf-8"))
        self.assertEqual(cached, {"url": url, "text": html, "final_url": url})
        script = """
import importlib, json, sys
from pathlib import Path
from unittest.mock import patch
module = importlib.import_module('mrvectorbeat.crawler.crawler')
module.PAGE_CACHE_DIR = Path(sys.argv[1])
with patch.object(module.requests.Session, 'get', side_effect=AssertionError('Unexpected HTTP')):
    print(json.dumps(module.Crawler().fetch_page(sys.argv[2])))
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.cache_dir), url],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=15, check=True)
        self.assertEqual(json.loads(result.stdout), [200, html])

    def test_equivalent_article_addresses_share_one_cache_entry(self):
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get",
                   return_value=Mock(status_code=200, text="ABBA")) as get:
            self.assertEqual(Crawler.fetch_page(WIKI + "ABBA"), (200, "ABBA"))
            for suffix in ("ABB%41", "ABBA?oldid=123", "ABBA#History"):
                self.assertEqual(Crawler().fetch_page(WIKI + suffix), (200, "ABBA"))
        get.assert_called_once()
        self.assertEqual(len(list(self.cache_dir.glob("*.json"))), 1)

    def test_redirect_source_and_destination_both_reuse_downloaded_content(self):
        source, target = WIKI + "Alias", WIKI + "Artist"
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=[
            Mock(status_code=302, headers={"Location": target}),
            Mock(status_code=200, text="HTML"),
        ]) as get:
            self.assertEqual(Crawler.fetch_page(source), (200, "HTML"))
            self.assertEqual(Crawler().fetch_page(source), (200, "HTML"))
            self.assertEqual(Crawler().fetch_page(target), (200, "HTML"))
        self.assertEqual(get.call_count, 2)

    def test_redirect_to_cached_destination_only_fetches_redirect(self):
        source, target = WIKI + "Alias", WIKI + "Artist"
        Crawler._save_cached_page(target, "HTML")
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", return_value=Mock(
            status_code=302, headers={"Location": target},
        )) as get:
            self.assertEqual(Crawler.fetch_page(source), (200, "HTML"))
        get.assert_called_once()

    def test_failed_requests_are_retried_instead_of_being_cached(self):
        url = WIKI + "Artist"
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=[
            requests.ConnectionError("Offline"), Mock(status_code=503),
            Mock(status_code=200, text="Recovered"),
        ]) as get, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(Crawler.fetch_page(url), (None, None))
            self.assertEqual(Crawler.fetch_page(url), (503, None))
            self.assertFalse(Crawler._cache_path(url).exists())
            self.assertEqual(Crawler.fetch_page(url), (200, "Recovered"))
            self.assertEqual(Crawler.fetch_page(url), (200, "Recovered"))
        self.assertEqual(get.call_count, 3)

    def test_empty_successful_page_is_cached_and_query_parameters_stay_distinct(self):
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", return_value=Mock(
            status_code=200, text="",
        )) as get:
            for url in ("https://example.com/page?q=one", "https://example.com/page?q=two"):
                self.assertEqual(Crawler.fetch_page(url), (200, ""))
                self.assertEqual(Crawler.fetch_page(url), (200, ""))
        self.assertEqual(get.call_count, 2)

    def test_corrupt_entry_is_downloaded_again_and_replaced(self):
        url = WIKI + "Artist"
        Crawler._cache_path(url).write_text("{partial", encoding="utf-8")
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", return_value=Mock(
            status_code=200, text="Repaired",
        )) as get:
            self.assertEqual(Crawler.fetch_page(url), (200, "Repaired"))
            self.assertEqual(Crawler.fetch_page(url), (200, "Repaired"))
        get.assert_called_once()
        self.assertFalse(list(self.cache_dir.glob("*.tmp")))

    def test_concurrent_instances_fetch_the_same_page_only_once(self):
        entered, release = Event(), Event()
        results, errors = [], []

        def request(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=2):
                raise AssertionError("Request not released")
            return Mock(status_code=200, text="Shared page")

        def fetch():
            try:
                results.append(Crawler().fetch_page(WIKI + "Artist"))
            except Exception as error:
                errors.append(error)

        first, second = Thread(target=fetch), Thread(target=fetch)
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=request) as get:
            first.start()
            try:
                self.assertTrue(entered.wait(timeout=2))
                second.start()
            finally:
                release.set()
                first.join(timeout=2)
                if second.ident is not None:
                    second.join(timeout=2)
        self.assertEqual(errors, [])
        self.assertEqual(results, [(200, "Shared page"), (200, "Shared page")])
        self.assertEqual(get.call_count, 1)

    def test_restarted_crawl_reads_robots_and_articles_without_network(self):
        seed, peer = WIKI + "Artist", WIKI + "Peer"
        responses = {
            "https://en.wikipedia.org/robots.txt": "User-agent: *",
            seed: '<p>Music band.</p><a href="/wiki/Peer">Peer</a>',
            peer: "<p>Music singer.</p>",
        }
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=lambda url, **kwargs:
                   Mock(status_code=200, text=responses[url])) as get, \
                contextlib.redirect_stdout(io.StringIO()):
            first = Crawler().focused_crawl("Artist", seed=seed, max_pages=2)
        self.assertEqual(get.call_count, 3)
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=AssertionError("HTTP")), \
                contextlib.redirect_stdout(io.StringIO()):
            second = Crawler().focused_crawl("Artist", seed=seed, max_pages=2)
        self.assertEqual(first, second)

    def test_api_errors_with_http_200_are_not_cached(self):
        resolver = WikipediaSeedResolver()
        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=[
            Mock(status_code=200, text=json.dumps({"error": {"code": "maxlag"}})),
            Mock(status_code=200, text=json.dumps({"query": {"search": []}})),
        ]) as get, patch("mrvectorbeat.crawler.generate_seed.time.sleep"):
            self.assertEqual(resolver._query(list="search", srsearch="Artist"), {"search": []})
        self.assertEqual(get.call_count, 2)
        self.assertFalse(list(self.cache_dir.glob("*.json")))


if __name__ == "__main__":
    unittest.main()
