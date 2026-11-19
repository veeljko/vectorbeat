import contextlib
import io
import unittest
from copy import deepcopy
from unittest.mock import call, patch

from mrvectorbeat.crawler.crawler import Crawler, WIKI


class CrawlIntoGraphTests(unittest.TestCase):
    def test_merges_new_nodes_and_edges_without_losing_existing_graph(self):
        seed, old, genre, album, peer, retained = [WIKI + title for title in
                                                 ("Source", "Existing", "Rock_music", "Album", "Peer", "Retained")]
        graph = {seed: [old], old: [], genre: [old], retained: []}
        pages = {
            seed: '<a href="/wiki/Rock_music">Genre</a><a href="/wiki/Rock_music#History">Again</a>'
                  '<a href="/wiki/Existing">Old</a><a href="/wiki/Album">Album</a>',
            genre: '<a href="/wiki/Peer">Artist</a>',
        }
        crawler = Crawler()
        with patch.object(crawler, "focused_crawl", return_value={"Source": pages}) as focused:
            result = crawler.crawl_into_graph("Source", graph, max_pages=20, threshold=3)
        focused.assert_called_once_with("Source", max_pages=20, threshold=3)
        self.assertIs(result, graph)
        artist_nodes = crawler.build_artist_nodes(["Source"], graph)
        self.assertEqual(artist_nodes, {"Source": seed})
        self.assertIs(graph[artist_nodes["Source"]], graph[seed])
        self.assertEqual(graph, {seed: [old, album, genre], old: [], genre: [old, peer],
                                 retained: [], album: [], peer: []})

    def test_repeating_the_same_crawl_does_not_duplicate_edges(self):
        seed, peer = WIKI + "Source", WIKI + "Peer"
        pages = {seed: '<a href="/wiki/Peer">Peer</a><a href="/wiki/Peer#Top">Again</a>'
                       '<a href="#top">Self</a><a href="/wiki/Category:Music">Category</a>'}
        crawler = Crawler()
        graph = {}
        with patch.object(crawler, "focused_crawl", return_value={"Source": pages}):
            crawler.crawl_into_graph("Source", graph)
            original = deepcopy(graph)
            crawler.crawl_into_graph("Source", graph)
        self.assertEqual(graph, original)
        self.assertEqual(graph, {seed: [peer], peer: []})
        self.assertEqual(crawler.build_artist_nodes(["Source"], graph), {"Source": seed})

    def test_later_artist_crawl_fills_previously_discovered_node(self):
        source, peer, genre = WIKI + "Source", WIKI + "Peer", WIKI + "Jazz"
        crawler = Crawler()
        graph = {}
        with patch.object(crawler, "focused_crawl", side_effect=[
            {"Source": {source: '<a href="/wiki/Peer">Peer</a>'}},
            {"Peer": {peer: '<a href="/wiki/Jazz">Genre</a>'}},
        ]):
            result = crawler.crawl_into_graph("Source", graph)
            self.assertIs(result, graph)
            self.assertEqual(graph[peer], [])
            crawler.crawl_into_graph("Peer", graph)
        self.assertEqual(graph, {source: [peer], peer: [genre], genre: []})
        self.assertEqual(crawler.build_artist_nodes(["Source", "Peer"], graph),
                         {"Source": source, "Peer": peer})

    def test_empty_crawl_preserves_graph(self):
        graph = {WIKI + "Existing": []}
        original = deepcopy(graph)
        crawler = Crawler()
        with patch.object(crawler, "focused_crawl", return_value={"Missing": {}}):
            result = crawler.crawl_into_graph("Missing", graph)
        self.assertIs(result, graph)
        self.assertEqual(crawler.build_artist_nodes(["Missing"], graph), {})
        self.assertEqual(graph, original)

    def test_collects_links_even_from_downloaded_pages_below_topic_threshold(self):
        source, peer, album = WIKI + "Source", WIKI + "Peer", WIKI + "Album"
        robots = "https://en.wikipedia.org/robots.txt"
        responses = {
            robots: (200, "User-agent: *"),
            source: (200, '<p>music singer</p><a href="/wiki/Peer">Peer</a>'),
            peer: (200, '<p>Weather.</p><a href="/wiki/Album">Album</a>'),
        }
        graph = {}
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=lambda url: responses[url]) as fetch, \
                contextlib.redirect_stdout(io.StringIO()):
            result = crawler.crawl_into_graph("Source", graph, max_pages=2)
        self.assertIs(result, graph)
        self.assertEqual(crawler.build_artist_nodes(["Source"], graph), {"Source": source})
        self.assertEqual(graph, {source: [peer], peer: [album], album: []})
        self.assertEqual(fetch.call_args_list, [call(robots), call(source), call(peer)])

    def test_build_artist_nodes_filters_missing_names_without_changing_graph_or_fetching(self):
        existing = WIKI + "Existing"
        graph = {existing: []}
        original = deepcopy(graph)
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=AssertionError("Unexpected HTTP request")):
            labels = crawler.build_artist_nodes(["Existing", "Missing"], graph)
        self.assertEqual(labels, {"Existing": existing})
        self.assertEqual(graph, original)

    def test_build_artist_nodes_supports_existing_nodes_without_outgoing_links(self):
        source = WIKI + "Source"
        graph = {source: []}
        self.assertEqual(Crawler.build_artist_nodes(["Source"], graph), {"Source": source})

    def test_build_artist_nodes_handles_empty_inputs_and_encoded_artist_names(self):
        seed = WIKI + "P%21nk"
        self.assertEqual(Crawler.build_artist_nodes(["P!nk"], {seed: []}), {"P!nk": seed})
        self.assertEqual(Crawler.build_artist_nodes([], {seed: []}), {})
        self.assertEqual(Crawler.build_artist_nodes(["Missing"], {}), {})


if __name__ == "__main__":
    unittest.main()
