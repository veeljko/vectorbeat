import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mrvectorbeat import pipeline
from mrvectorbeat.crawler.crawler import Crawler
from mrvectorbeat.search.ranking.similar_artists import build_similarity_index, find_similar_artists


WIKI = "https://en.wikipedia.org/wiki/"


class SimilarArtistsTests(unittest.TestCase):
    def test_shared_incoming_pages_have_inverse_artist_degree_weights(self):
        nodes = {f"Artist {i:03}": WIKI + f"Artist_{i:03}" for i in range(100)}
        urls = list(nodes.values())
        genre, label, award = [WIKI + title for title in ("Small_genre", "Small_label", "Awards")]
        graph = {url: [] for url in urls}
        graph.update({genre: urls[:4], label: urls[:2], award: urls})
        original = deepcopy(graph)
        index = build_similarity_index(graph, nodes)
        result = find_similar_artists(index, "Artist 000", limit=100)
        self.assertEqual(result[0]["artist_name"], "Artist 001")
        self.assertAlmostEqual(result[0]["score"], 0.76)
        self.assertEqual([page["url"] for page in result[0]["shared_pages"]], [label, genre, award])
        self.assertEqual([page["weight"] for page in result[0]["shared_pages"]], [0.5, 0.25, 0.01])
        self.assertAlmostEqual(result[1]["score"], 0.26)
        self.assertEqual(graph, original)

    def test_common_outgoing_target_is_not_a_common_incoming_page(self):
        index = build_similarity_index({"a": ["genre"], "b": ["genre"], "genre": []},
                                       {"Alpha": "a", "Beta": "b"})
        self.assertEqual(find_similar_artists(index, "Alpha"), [])

    def test_duplicate_links_and_aliases_never_inflate_weight_or_recommend_self(self):
        index = build_similarity_index({"page": ["a", "a", "b", "b"], "a": [], "b": []},
                                       {"Alpha": "a", "Alpha alias": "a", "Beta": "b", "Beta alias": "b"})
        self.assertEqual(index["page_weights"]["page"], 0.5)
        results = find_similar_artists(index, "Alpha alias")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "b")
        self.assertEqual(results[0]["aliases"], ("Beta", "Beta alias"))
        reverse = find_similar_artists(index, "Beta alias")
        self.assertEqual(reverse[0]["score"], results[0]["score"])
        self.assertEqual(reverse[0]["url"], "a")

    def test_query_visits_only_candidates_from_selected_artists_incoming_pages(self):
        class Unvisited:
            def __iter__(self):
                raise AssertionError("Unrelated index entry was scanned")

        index = build_similarity_index({"shared": ["a", "b"], "unrelated": ["c"]},
                                       {"Alpha": "a", "Beta": "b", "Other": "c"})
        index["artist_pages"]["c"] = Unvisited()
        index["page_artists"]["unrelated"] = Unvisited()
        self.assertEqual(find_similar_artists(index, "Alpha")[0]["artist_name"], "Beta")

    def test_query_preserves_index_and_sorts_equal_scores_by_name(self):
        index = build_similarity_index({"p1": ["a", "z"], "p2": ["a", "b"]},
                                       {"Source": "a", "Zulu": "z", "Beta": "b"})
        before = deepcopy(index)
        results = find_similar_artists(index, "Source")
        self.assertEqual([row["artist_name"] for row in results], ["Beta", "Zulu"])
        self.assertEqual(find_similar_artists(index, "Source", limit=1), results[:1])
        self.assertEqual(index, before)
        for row in results:
            self.assertAlmostEqual(row["score"], sum(page["weight"] for page in row["shared_pages"]))

    def test_unknown_isolated_and_missing_nodes_have_no_results(self):
        index = build_similarity_index({"a": []}, {"Alpha": "a", "Absent": "absent"})
        self.assertNotIn("Absent", index["artist_nodes"])
        for artist in ("Unknown", "Alpha", "Absent"):
            self.assertEqual(find_similar_artists(index, artist), [])
        self.assertEqual(build_similarity_index({}, {"Alpha": "a"})["artist_nodes"], {})
        self.assertEqual(find_similar_artists(index, "Alpha", limit=0), [])
        for limit in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                find_similar_artists(index, "Alpha", limit=limit)

    def test_page_titles_are_decoded_from_existing_urls_without_fetching(self):
        page = WIKI + "Bj%C3%B6rk_and_music"
        with patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP")):
            index = build_similarity_index({page: ["a", "b"]}, {"A": "a", "B": "b"})
            result = find_similar_artists(index, "A")
        self.assertEqual(result[0]["shared_pages"][0]["title"], "Björk and music")
        self.assertEqual(result[0]["shared_pages"][0]["url"], page)


class SimilarArtistsPipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        patcher = patch.multiple(pipeline, GRAF_PATH=root / "graph.pkl",
                                  ARTIST_NODES_PATH=root / "artists.pkl", SEED_PATH=root / "seed.parquet",
                                  SLICNI_IZVODJACI_PATH=root / "similar.pkl")
        patcher.start()
        self.addCleanup(patcher.stop)
        pipeline._snimi_pickle({"p": ["a", "b"], "a": [], "b": []}, pipeline.GRAF_PATH)
        pipeline._snimi_pickle({"Alpha": "a", "Beta": "b"}, pipeline.ARTIST_NODES_PATH)

    def test_build_reuses_existing_artifacts_and_reload_needs_no_graph_or_crawling(self):
        with patch.object(pipeline, "napravi_graf", side_effect=AssertionError("Crawl")), \
                patch.object(pipeline, "ucitaj_katalog", side_effect=AssertionError("Catalog")), \
                patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP")):
            built = pipeline.napravi_slicne_izvodjace()
        self.assertTrue(pipeline.SLICNI_IZVODJACI_PATH.exists())
        with patch.object(pipeline, "ucitaj_graf", side_effect=AssertionError("Graph read")), \
                patch("mrvectorbeat.search.ranking.similar_artists.build_similarity_index",
                      side_effect=AssertionError("Rebuild")):
            self.assertEqual(pipeline.ucitaj_slicne_izvodjace(), built)
            self.assertEqual(pipeline.napravi_slicne_izvodjace(), built)
        self.assertEqual(find_similar_artists(built, "Alpha")[0]["score"], 0.5)

    def test_missing_artifact_shows_command_without_building(self):
        with self.assertRaisesRegex(FileNotFoundError, "--samo-slicni-izvodjaci"):
            pipeline.ucitaj_slicne_izvodjace()

    def test_missing_graph_is_not_crawled_during_index_preparation(self):
        pipeline.GRAF_PATH.unlink()
        with patch.object(pipeline, "napravi_graf", side_effect=AssertionError("Crawl")):
            with self.assertRaises(FileNotFoundError):
                pipeline.napravi_slicne_izvodjace()

    def test_changed_graph_invalidates_and_refreshes_similarity_index(self):
        pipeline.napravi_slicne_izvodjace()
        pipeline._snimi_pickle({"p": ["a", "b"], "q": ["a", "b"]}, pipeline.GRAF_PATH)
        older = pipeline.GRAF_PATH.stat().st_mtime_ns - 1_000_000_000
        os.utime(pipeline.SLICNI_IZVODJACI_PATH, ns=(older, older))
        with self.assertRaisesRegex(ValueError, "zastareo"):
            pipeline.ucitaj_slicne_izvodjace()
        rebuilt = pipeline.napravi_slicne_izvodjace()
        self.assertEqual(find_similar_artists(rebuilt, "Alpha")[0]["score"], 1)

    def test_stale_graph_requires_explicit_graph_preparation(self):
        pipeline.SEED_PATH.write_text("new seeds", encoding="utf-8")
        older = pipeline.SEED_PATH.stat().st_mtime_ns - 1_000_000_000
        os.utime(pipeline.GRAF_PATH, ns=(older, older))
        with patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP")):
            with self.assertRaisesRegex(ValueError, "--samo-graf-seed"):
                pipeline.napravi_slicne_izvodjace()


if __name__ == "__main__":
    unittest.main()
