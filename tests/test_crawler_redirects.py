import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mrvectorbeat import pipeline
from mrvectorbeat.crawler.crawler import Crawler, WIKI


class RedirectIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / "pages"
        for target, value in (
            ("mrvectorbeat.crawler.crawler.PAGE_CACHE_DIR", self.cache),
            ("mrvectorbeat.crawler.crawler.REQUEST_INTERVAL_SECONDS", 0.0),
            ("mrvectorbeat.crawler.crawler.Crawler._last_request_finished", None),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def remember(self, aliases, target):
        with Crawler._request_lock:
            Crawler._remember_aliases(aliases, target)

    def test_http_redirect_chain_returns_and_caches_the_final_url(self):
        alias, middle, target = [WIKI + title for title in ("Alias", "Middle", "Canonical")]
        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=[
            Mock(status_code=301, headers={"Location": middle}),
            Mock(status_code=302, headers={"Location": target}),
            Mock(status_code=200, text="<p>music band</p>"),
        ]) as get:
            self.assertEqual(Crawler.fetch_page(alias, return_url=True), (200, "<p>music band</p>", target))
            self.assertEqual(Crawler.fetch_page(alias), (200, "<p>music band</p>"))
            self.assertEqual(Crawler.fetch_page(middle, return_url=True), (200, "<p>music band</p>", target))
        self.assertEqual(get.call_count, 3)
        for url in (alias, middle, target):
            self.assertEqual(json.loads(Crawler._cache_path(url).read_text(encoding="utf-8"))["final_url"], target)
            self.assertEqual(Crawler.canonical_url(url), target)

    def test_http_200_canonical_link_also_identifies_the_target(self):
        alias, target = WIKI + "Alias", WIKI + "Canonical"
        html = '<link rel="canonical" href="/wiki/Canonical"><p>music band</p>'
        with patch("mrvectorbeat.crawler.crawler.requests.get", return_value=Mock(
            status_code=200, text=html,
        )) as get:
            self.assertEqual(Crawler.fetch_page(alias, return_url=True), (200, html, target))
            self.assertEqual(Crawler.fetch_page(target, return_url=True), (200, html, target))
        get.assert_called_once()

    def test_redirect_metadata_survives_a_fresh_python_process_without_http(self):
        alias, target = WIKI + "Alias", WIKI + "Canonical"
        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=[
            Mock(status_code=302, headers={"Location": target}), Mock(status_code=200, text="HTML"),
        ]):
            Crawler.fetch_page(alias)
        script = """
import importlib, json, sys
from pathlib import Path
from unittest.mock import patch
module = importlib.import_module('mrvectorbeat.crawler.crawler')
module.PAGE_CACHE_DIR = Path(sys.argv[1])
with patch.object(module.requests, 'get', side_effect=AssertionError('Unexpected HTTP')):
    print(json.dumps([module.Crawler.fetch_page(sys.argv[2], return_url=True),
                      module.Crawler.canonical_url(sys.argv[2])]))
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.cache), alias],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                text=True, timeout=15, check=True)
        self.assertEqual(json.loads(result.stdout), [[200, "HTML", target], target])

    def test_graph_merges_old_incoming_and_outgoing_edges_without_duplicates_or_self_links(self):
        source, alias, target, x, y = [WIKI + title for title in ("Source", "Alias", "Canonical", "X", "Y")]
        self.remember([alias], target)
        graph = {source: [alias, target], alias: [target, x], target: [alias, y], x: [], y: [alias]}
        identity = id(graph)
        self.assertTrue(Crawler.canonicalize_graph(graph))
        self.assertEqual(id(graph), identity)
        self.assertEqual(graph, {source: [target], target: [x, y], x: [], y: [target]})
        self.assertFalse(Crawler.canonicalize_graph(graph))

    def test_focused_crawl_counts_canonical_page_once_and_ignores_redirected_self_links(self):
        root, alias, target = [WIKI + title for title in ("Root", "Alias", "Canonical")]
        robots = "https://en.wikipedia.org/robots.txt"
        responses = {
            robots: Mock(status_code=200, text="User-agent: *"),
            root: Mock(status_code=200, text='<p>music band</p><a href="/wiki/Alias">Alias</a>'
                       '<a href="/wiki/Canonical">Same article</a>'),
            alias: Mock(status_code=302, headers={"Location": target}),
            target: Mock(status_code=200, text='<p>music band</p><a href="/wiki/Canonical">Self</a>'
                         '<a href="/wiki/Alias">Same self</a>'),
        }
        crawler = Crawler()
        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=lambda url, **kwargs: responses[url]) as get, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            graph = crawler.crawl_into_graph("Artist", {}, seed=root, max_pages=3)
        self.assertEqual(graph, {root: [target], target: []})
        self.assertEqual(get.call_count, 4)
        self.assertIn("Downloaded pages: 2", output.getvalue())

    def test_later_artist_crawl_repairs_previously_discovered_alias_edges_and_labels(self):
        root, alias, target = [WIKI + title for title in ("Root", "Alias", "Canonical")]
        responses = {
            "https://en.wikipedia.org/robots.txt": Mock(status_code=200, text="User-agent: *"),
            root: Mock(status_code=200, text='<p>music band</p><a href="/wiki/Alias">Artist</a>'),
            alias: Mock(status_code=302, headers={"Location": target}),
            target: Mock(status_code=200, text="<p>music singer</p>"),
        }
        crawler = Crawler()
        graph = {}
        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=lambda url, **kwargs: responses[url]), \
                contextlib.redirect_stdout(io.StringIO()):
            crawler.crawl_into_graph("First", graph, seed=root, max_pages=1)
            self.assertIn(alias, graph)
            crawler.crawl_into_graph("Second", graph, seed=alias, max_pages=1)
        self.assertEqual(graph, {root: [target], target: []})
        self.assertEqual(crawler.build_artist_nodes(["Second"], graph, artist_seeds={"Second": alias}),
                         {"Second": target})

    def test_legacy_cache_can_be_migrated_offline_using_html_canonical_metadata(self):
        alias, target = WIKI + "Alias", WIKI + "Canonical"
        self.cache.mkdir()
        html = f'<link rel="canonical" href="{target}"><p>music band</p>'
        Crawler._cache_path(alias).write_text(json.dumps({"url": alias, "text": html}), encoding="utf-8")
        with patch("mrvectorbeat.crawler.crawler.requests.get", side_effect=AssertionError("HTTP")):
            Crawler.migrate_cached_aliases()
            self.assertEqual(Crawler.canonical_url(alias), target)
            self.assertEqual(Crawler.fetch_page(alias, return_url=True), (200, html, target))
            before = (self.cache / "url_aliases.json").stat().st_mtime_ns
            Crawler.migrate_cached_aliases()
            self.assertEqual((self.cache / "url_aliases.json").stat().st_mtime_ns, before)

    def test_preparation_repairs_existing_graph_and_artist_map_without_crawling(self):
        alias, target, source = [WIKI + title for title in ("Alias", "Canonical", "Source")]
        self.remember([alias], target)
        with patch.multiple(pipeline, GRAF_PATH=self.root / "graph.pkl",
                            ARTIST_NODES_PATH=self.root / "artists.pkl", SEED_PATH=self.root / "seed.parquet"):
            pipeline._snimi_pickle({source: [alias, target], alias: [target], target: []}, pipeline.GRAF_PATH)
            pipeline._snimi_pickle({"Artist": alias}, pipeline.ARTIST_NODES_PATH)
            with patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP")), \
                    patch.object(pipeline, "napravi_seed", side_effect=AssertionError("Seed preparation")):
                graph, nodes = pipeline.napravi_graf()
            self.assertEqual(graph, {source: [target], target: []})
            self.assertEqual(nodes, {"Artist": target})
            self.assertEqual(pipeline.ucitaj_graf(), (graph, nodes))


if __name__ == "__main__":
    unittest.main()
