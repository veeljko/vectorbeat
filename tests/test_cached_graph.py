import contextlib
import io
import json
import pickle
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mrvectorbeat import pipeline
from mrvectorbeat.crawler.crawler import Crawler, WIKI
from mrvectorbeat.search.ranking.page_ranking import pagerank
from mrvectorbeat.search.ranking.similar_artists import build_similarity_index, find_similar_artists


class CachedGraphTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / "pages"
        self.cache.mkdir()
        for patcher in (
            patch("mrvectorbeat.crawler.crawler.PAGE_CACHE_DIR", self.cache),
            patch.multiple(pipeline, SEED_PATH=self.root / "seed.parquet",
                           GRAF_PATH=self.root / "graf.pkl", ARTIST_NODES_PATH=self.root / "artist_nodes.pkl"),
            patch.object(Crawler, "fetch_page", side_effect=AssertionError("Offline build attempted HTTP")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_offline_build_unions_aliases_and_edges_and_ignores_non_articles_and_corrupt_files(self):
        source, alias, peer, genre, discovered = [WIKI + name for name in
                                                ("Artist", "Alias", "Peer", "Genre", "Discovered")]
        Crawler._save_cached_page(source, '<a href="/wiki/Peer">Peer</a>'
                                 '<a href="/wiki/Discovered#top">Discovered</a>'
                                 '<a href="/wiki/File_talk:Music">Talk</a>')
        Crawler._save_cached_page(alias, '<a href="/wiki/Genre">Genre</a>'
                                 '<a href="/wiki/Alias">Self</a>', source)
        Crawler._save_cached_page(peer, '<a href="/wiki/Alias">Artist</a>'
                                 '<a href="/wiki/Artist?oldid=123">Artist again</a>')
        Crawler._save_cached_page(genre, "<p>Music genre</p>")
        Crawler._save_cached_page("https://en.wikipedia.org/robots.txt", "User-agent: *")
        (self.cache / "broken.json").write_text("{partial", encoding="utf-8")
        (self.cache / "invalid.json").write_text("[]", encoding="utf-8")
        graph = Crawler.build_graph_from_cache()
        self.assertEqual({node: set(targets) for node, targets in graph.items()}, {
            source: {peer, genre}, peer: {source}, genre: set(),
        })
        self.assertNotIn(alias, graph)
        self.assertNotIn(discovered, graph)

    def test_shared_target_urls_reuse_one_string_object_in_memory_and_in_pickle(self):
        source, peer, shared = WIKI + "Artist", WIKI + "Peer", WIKI + "Shared"
        for url in (source, peer):
            Crawler._save_cached_page(url, '<a href="/wiki/Shared">Shared</a>')
        Crawler._save_cached_page(shared, "<p>Downloaded page without outgoing links</p>")
        graph = Crawler.build_graph_from_cache()
        shared_key = next(node for node in graph if node == shared)
        self.assertIs(graph[source][0], shared_key)
        self.assertIs(graph[peer][0], shared_key)
        loaded = pickle.loads(pickle.dumps(graph))
        self.assertIs(loaded[source][0], loaded[peer][0])

    def test_old_cache_canonical_html_is_migrated_without_network(self):
        alias, source, peer = WIKI + "Alias", WIKI + "Artist", WIKI + "Peer"
        html = '<head><link rel="canonical" href="/wiki/Artist"></head>'
        html += '<a href="/wiki/Peer">Peer</a>'
        Crawler._cache_path(alias).write_text(json.dumps({"url": alias, "text": html}), encoding="utf-8")
        Crawler._save_cached_page(peer, "")
        graph = Crawler.build_graph_from_cache()
        self.assertEqual(graph, {source: [peer], peer: []})
        self.assertEqual(Crawler.canonical_url(alias), source)

    def test_previous_html_is_released_before_the_next_page_is_loaded(self):
        class TrackedHTML(str):
            pass

        for name in ("Alpha", "Bravo", "Charlie"):
            Crawler._save_cached_page(WIKI + name, f'<p>{name}</p><a href="/wiki/Shared">Link</a>')
        previous = []
        read_json = Crawler._read_cache_json

        def track(path):
            self.assertTrue(all(reference() is None for reference in previous), "Previous HTML retained")
            cached = read_json(path)
            cached["text"] = TrackedHTML(cached["text"])
            previous.append(weakref.ref(cached["text"]))
            return cached

        with patch.object(Crawler, "migrate_cached_aliases"), \
                patch.object(Crawler, "_load_alias_index", return_value={"aliases": {}}), \
                patch.object(Crawler, "_read_cache_json", side_effect=track):
            graph = Crawler.build_graph_from_cache()
        self.assertEqual(len(previous), 6)
        self.assertTrue(all(reference() is None for reference in previous))
        self.assertEqual(len(graph), 3)
        self.assertNotIn(WIKI + "Shared", graph)

    def test_pipeline_rebuilds_existing_graph_offline_and_preserves_artist_mapping(self):
        source, peer, missing = WIKI + "Artist", WIKI + "Peer", WIKI + "Not_downloaded"
        seeds = pd.DataFrame({"artist_name": ["Catalog artist", "Unresolved", "Missing article"],
                              "seed": [source, None, missing]})
        seeds.to_parquet(pipeline.SEED_PATH, index=False)
        pipeline._snimi_pickle({source: [missing], missing: []}, pipeline.GRAF_PATH)
        pipeline._snimi_pickle({"Catalog artist": source, "Missing article": missing}, pipeline.ARTIST_NODES_PATH)
        Crawler._save_cached_page(source, '<a href="/wiki/Peer">Peer</a><a href="/wiki/Not_downloaded">Missing</a>')
        Crawler._save_cached_page(peer, "")
        with patch.object(pipeline, "napravi_seed", side_effect=AssertionError("Seed search")), \
                patch.object(Crawler, "focused_crawl", side_effect=AssertionError("Crawl replayed")), \
                patch.object(Crawler, "canonicalize_graph", side_effect=AssertionError("Full graph copied")), \
                patch("concurrent.futures.ThreadPoolExecutor", side_effect=AssertionError("Parallel HTML retained")), \
                contextlib.redirect_stdout(io.StringIO()):
            graph, nodes = pipeline.napravi_graf(cache_only=True)
        self.assertEqual(graph, {source: [peer], peer: []})
        self.assertEqual(nodes, {"Catalog artist": source})
        self.assertEqual(pipeline.ucitaj_graf(), (graph, nodes))

    def test_forward_links_are_preserved_regardless_of_cache_scan_order(self):
        first, second = WIKI + "First", WIKI + "Second"
        Crawler._save_cached_page(first, '<a href="/wiki/Second">Later page</a>')
        Crawler._save_cached_page(second, '<a href="/wiki/First">First page</a>')
        paths = [Crawler._cache_path(first), Crawler._cache_path(second)]
        with patch.object(Path, "glob", return_value=iter(paths)):
            graph = Crawler.build_graph_from_cache()
        self.assertEqual(graph, {first: [second], second: [first]})

    def test_thousands_of_discovered_links_do_not_increase_downloaded_node_count(self):
        source, isolated = WIKI + "Artist", WIKI + "Isolated"
        html = "".join(f'<a href="/wiki/Not_downloaded_{number}">Link</a>' for number in range(5000))
        Crawler._save_cached_page(source, html)
        Crawler._save_cached_page(isolated, "<p>No outgoing or incoming links</p>")
        graph = Crawler.build_graph_from_cache()
        self.assertEqual(graph, {source: [], isolated: []})

    def test_alias_chains_from_later_records_merge_before_edges_are_filtered(self):
        first, alias, middle, canonical = [WIKI + name for name in ("First", "Alias", "Middle", "Canonical")]
        Crawler._save_cached_page(first, '<a href="/wiki/Alias">Alias</a>')
        Crawler._save_cached_page(alias, '<a href="/wiki/First">First</a>', middle)
        Crawler._save_cached_page(middle, '<a href="/wiki/Canonical">Self</a>', canonical)
        # An earlier crawl may already have marked migration complete. Newly
        # found metadata must still update aliases and artist labels offline.
        with Crawler._request_lock:
            Crawler._save_alias_index({"aliases": {}, "legacy_cache_scanned": True})
        paths = [Crawler._cache_path(url) for url in (first, alias, middle)]
        with patch.object(Path, "glob", return_value=iter(paths)):
            graph = Crawler.build_graph_from_cache()
        self.assertEqual(graph, {first: [canonical], canonical: [first]})
        self.assertEqual(Crawler.build_artist_nodes(["Artist"], graph, artist_seeds={"Artist": alias}),
                         {"Artist": canonical})

    def test_pagerank_and_artist_similarity_use_only_downloaded_nodes(self):
        artist_a, artist_b, genre, missing = [WIKI + title for title in
                                            ("Artist_A", "Artist_B", "Genre", "Missing_artist")]
        for url in (artist_a, artist_b):
            Crawler._save_cached_page(url, '<a href="/wiki/Genre">Genre</a>')
        Crawler._save_cached_page(genre, '<a href="/wiki/Artist_A">A</a>'
                                 '<a href="/wiki/Artist_B">B</a>'
                                 '<a href="/wiki/Missing_artist">Not downloaded</a>')
        graph = Crawler.build_graph_from_cache()
        seeds = {"A": artist_a, "B": artist_b, "Missing": missing}
        nodes = Crawler.build_artist_nodes(seeds, graph, artist_seeds=seeds)
        self.assertEqual(nodes, {"A": artist_a, "B": artist_b})
        ranks, _ = pagerank(graph)
        self.assertEqual(set(ranks), {artist_a, artist_b, genre})
        self.assertAlmostEqual(sum(ranks.values()), 1.0)
        similar = find_similar_artists(build_similarity_index(graph, nodes), "A")
        self.assertEqual([row["artist_name"] for row in similar], ["B"])
        self.assertEqual(similar[0]["score"], 0.5)

    def test_no_valid_cached_pages_does_not_replace_existing_artifacts(self):
        pd.DataFrame({"artist_name": ["Artist"], "seed": [WIKI + "Artist"]}).to_parquet(
            pipeline.SEED_PATH, index=False,
        )
        pipeline._snimi_pickle({"old": []}, pipeline.GRAF_PATH)
        pipeline._snimi_pickle({"old": "old"}, pipeline.ARTIST_NODES_PATH)
        original = (pipeline.GRAF_PATH.read_bytes(), pipeline.ARTIST_NODES_PATH.read_bytes())
        Crawler._save_cached_page("https://en.wikipedia.org/robots.txt", "User-agent: *")
        with self.assertRaisesRegex(ValueError, "no valid downloaded Wikipedia articles"):
            pipeline.napravi_graf(cache_only=True)
        self.assertEqual((pipeline.GRAF_PATH.read_bytes(), pipeline.ARTIST_NODES_PATH.read_bytes()), original)

    def test_missing_cache_directory_is_reported_without_fetching(self):
        with patch("mrvectorbeat.crawler.crawler.PAGE_CACHE_DIR", self.root / "missing"):
            with self.assertRaises(FileNotFoundError):
                Crawler.build_graph_from_cache()

    def test_anchor_filter_ignores_scripts_comments_and_does_not_change_link_order(self):
        html = '<script>var fake = \'<a href="/wiki/Fake">fake</a>\';</script>'
        html += '<!-- <a href="/wiki/Comment">Comment</a> -->'
        html += '<div><a href="/wiki/Peer?x=1&amp;y=2">Peer</a></div>'
        html += '<a href="/wiki/Other"/><p>No href</p><a>Ignored</a>'
        self.assertEqual(Crawler._raw_links(html, WIKI + "Source"),
                         [WIKI + "Peer?x=1&y=2", WIKI + "Other"])


if __name__ == "__main__":
    unittest.main()
