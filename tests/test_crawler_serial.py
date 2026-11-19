import contextlib
import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, get_ident, local
from unittest.mock import Mock, patch

import pandas as pd

from mrvectorbeat import pipeline
from mrvectorbeat.crawler.crawler import Crawler, WIKI


class SerialCrawlerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for patcher in (
            patch("mrvectorbeat.crawler.crawler.PAGE_CACHE_DIR", root / "pages"),
            # Timing is checked separately with a fake clock. Even at zero,
            # callers from different threads must not overlap HTTP requests.
            patch("mrvectorbeat.crawler.crawler.REQUEST_INTERVAL_SECONDS", 0.0),
            patch.object(Crawler, "_last_request_finished", None),
            patch.object(Crawler, "_sessions", local()),
            patch.multiple(pipeline, SEED_PATH=root / "seed.parquet",
                           GRAF_PATH=root / "graf.pkl", ARTIST_NODES_PATH=root / "artist_nodes.pkl"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_session_is_reused_in_one_thread_but_not_shared_between_threads(self):
        barrier = Barrier(2)
        main_session = Crawler._get_session()

        def session_pair(_):
            session = Crawler._get_session()
            barrier.wait(timeout=3)
            return session, Crawler._get_session()

        with ThreadPoolExecutor(max_workers=2) as pool:
            pairs = list(pool.map(session_pair, range(2)))
        self.assertIs(main_session, Crawler._get_session())
        self.assertIs(pairs[0][0], pairs[0][1])
        self.assertIs(pairs[1][0], pairs[1][1])
        self.assertIsNot(pairs[0][0], pairs[1][0])
        self.assertIsNot(main_session, pairs[0][0])
        for session in (main_session, pairs[0][0], pairs[1][0]):
            session.close()

    def test_external_parallel_callers_cannot_overlap_http_requests(self):
        entered, release, second_started, second_http = Event(), Event(), Event(), Event()
        urls = [WIKI + "Alpha", WIKI + "Bravo"]

        def request(url, **options):
            if url == urls[0]:
                entered.set()
                if not release.wait(timeout=3):
                    raise AssertionError("First request not released")
            else:
                second_http.set()
            return Mock(status_code=200, text=f"<p>{url}</p>")

        def second_fetch():
            second_started.set()
            return Crawler.fetch_page(urls[1])

        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=request), \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(Crawler.fetch_page, urls[0])
            try:
                self.assertTrue(entered.wait(timeout=3))
                second = pool.submit(second_fetch)
                self.assertTrue(second_started.wait(timeout=3))
                self.assertFalse(second_http.wait(timeout=0.05))
            finally:
                release.set()
            self.assertEqual(first.result(timeout=3), (200, f"<p>{urls[0]}</p>"))
            self.assertEqual(second.result(timeout=3), (200, f"<p>{urls[1]}</p>"))

    def test_serial_http_redirects_share_cached_destination_and_keep_both_aliases(self):
        sources = [WIKI + "Alias_A", WIKI + "Alias_B"]
        target = WIKI + "Artist"

        def request(url, **options):
            if url in sources:
                return Mock(status_code=302, headers={"Location": target})
            self.assertEqual(url, target)
            return Mock(status_code=200, text="<p>Music singer</p>")

        with patch("mrvectorbeat.crawler.crawler.requests.Session.get", side_effect=request) as get, \
                ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(Crawler.fetch_page, sources))
        self.assertEqual(get.call_count, 3)
        self.assertEqual(results, [(200, "<p>Music singer</p>")] * 2)
        for source in sources:
            self.assertEqual(Crawler.canonical_url(source), target)
            self.assertEqual(Crawler._read_cached_page(source)[1], target)
        self.assertEqual(Crawler._page_locks, {})

    def test_preparation_crawls_artists_in_order_on_main_thread_and_saves_downloaded_graph(self):
        names = ["Alpha", "Bravo", "Charlie", "Delta"]
        seeds = pd.DataFrame({"artist_name": names + ["Missing"],
                              "seed": [WIKI + name for name in names] + [None]}).astype("string")
        seeds.to_parquet(pipeline.SEED_PATH, index=False)
        shared = WIKI + "Genre"
        Crawler._save_cached_page(shared, "".join(f'<a href="/wiki/{name}">{name}</a>' for name in names))
        visited, threads = [], []

        def crawl(artist, graph, **options):
            self.assertEqual(options["max_pages"], 10)
            self.assertFalse(options["canonicalize"])
            visited.append(artist)
            threads.append(get_ident())
            Crawler._save_cached_page(options["seed"], '<a href="/wiki/Genre">Genre</a>'
                                     '<a href="/wiki/Not_downloaded">Discovered only</a>')
            graph[options["seed"]] = [shared, WIKI + "Not_downloaded"]
            return graph

        with patch.object(Crawler, "crawl_into_graph", side_effect=crawl), \
                patch("concurrent.futures.ThreadPoolExecutor", side_effect=AssertionError("Thread pool used")), \
                contextlib.redirect_stdout(io.StringIO()):
            graph, nodes = pipeline.napravi_graf()
        self.assertEqual(visited, names)
        self.assertEqual(threads, [get_ident()] * len(names))
        self.assertTrue(all(graph[WIKI + name] == [shared] for name in names))
        self.assertEqual(set(graph[shared]), {WIKI + name for name in names})
        self.assertNotIn(WIKI + "Not_downloaded", graph)
        self.assertEqual(nodes, {name: WIKI + name for name in names})
        self.assertEqual(pipeline.ucitaj_graf(), (graph, nodes))


if __name__ == "__main__":
    unittest.main()
