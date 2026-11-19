import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mrvectorbeat import pipeline
from mrvectorbeat.search.ranking.artist_influence import build_artist_influence
from mrvectorbeat.search.ranking.page_ranking import pagerank


def fixture():
    graph = {"artist-a": [], "artist-b": [], "genre": ["artist-a", "artist-a"],
             "album": ["genre"], "label": ["genre"], "other": ["artist-b"]}
    artists = {"Artist A": "artist-a", "Artist B": "artist-b", "Outside catalog": "genre"}
    return graph, artists


class ArtistInfluenceTests(unittest.TestCase):
    def test_entire_graph_determines_influence_before_catalog_filtering(self):
        graph, artists = fixture()
        original = deepcopy(graph)
        result = build_artist_influence(graph, artists, ["Artist A", "Artist B", "Missing"])
        expected, _ = pagerank({key: list(dict.fromkeys(values)) for key, values in graph.items()})
        self.assertEqual(result["page_ranks"], expected)
        self.assertEqual(result["graph_nodes"], 6)
        self.assertEqual(graph, original)
        table = result["artists"]
        self.assertEqual(table["artist_name"].tolist(), ["Artist A", "Artist B"])
        self.assertEqual(table["incoming_links"].tolist(), [1, 1])
        # Equal incoming counts, but A receives a link from a more influential page.
        self.assertGreater(table.iloc[0]["pagerank"], table.iloc[1]["pagerank"])
        self.assertEqual(table["position"].tolist(), [1, 2])
        self.assertAlmostEqual(sum(result["page_ranks"].values()), 1)
        self.assertLess(table["pagerank"].sum(), 1)
        self.assertEqual(result["missing_artists"], ["Missing"])

    def test_incoming_counts_unique_source_pages_and_includes_destination_only_nodes(self):
        result = build_artist_influence({"genre": ["a", "a"], "album": ["a"]},
                                        {"A": "a"}, ["A"])
        self.assertEqual(result["incoming_links"]["a"], 2)
        self.assertEqual(result["graph_edges"], 2)
        self.assertEqual(result["graph_nodes"], 3)
        self.assertEqual(result["artists"].iloc[0]["incoming_links"], 2)

    def test_empty_graph_and_missing_artist_nodes(self):
        result = build_artist_influence({}, {"Missing": "absent"}, ["Missing"])
        self.assertTrue(result["artists"].empty)
        self.assertEqual(result["missing_artists"], ["Missing"])
        self.assertEqual(result["page_ranks"], {})
        self.assertTrue(result["converged"])

    def test_equal_scores_have_deterministic_name_order(self):
        result = build_artist_influence({"a": [], "b": []}, {"Beta": "b", "Alpha": "a"},
                                        ["Alpha", "Beta", "Alpha"])
        self.assertEqual(result["artists"]["artist_name"].tolist(), ["Alpha", "Beta"])
        self.assertEqual(result["catalog_artists"], 2)

    def test_nonconvergence_is_recorded(self):
        with self.assertWarns(RuntimeWarning):
            result = build_artist_influence({"a": ["b"], "b": ["b"]}, {"A": "a"},
                                            ["A"], max_iterations=1)
        self.assertFalse(result["converged"])


class ArtistInfluencePipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        patcher = patch.multiple(pipeline, GRAF_PATH=root / "graph.pkl",
                                  ARTIST_NODES_PATH=root / "artists.pkl",
                                  RANGOVI_PATH=root / "ranks.pkl", KATALOG_PATH=root / "catalog.parquet",
                                  SEED_PATH=root / "seed.parquet")
        patcher.start()
        self.addCleanup(patcher.stop)
        graph, artists = fixture()
        pipeline._snimi_pickle(graph, pipeline.GRAF_PATH)
        pipeline._snimi_pickle(artists, pipeline.ARTIST_NODES_PATH)
        pd.DataFrame({"track_artist": ["Artist A", "Artist B", "Missing"]}).to_parquet(
            pipeline.KATALOG_PATH, index=False)

    def test_preparation_saves_and_loading_never_recomputes_or_crawls(self):
        built = pipeline.napravi_rangove()
        with patch("mrvectorbeat.search.ranking.artist_influence.pagerank", side_effect=AssertionError("PageRank")), \
                patch.object(pipeline, "ucitaj_graf", side_effect=AssertionError("Graph read")), \
                patch.object(pipeline, "napravi_graf", side_effect=AssertionError("Crawl")):
            loaded = pipeline.ucitaj_rangove()
            reused = pipeline.napravi_rangove()
        pd.testing.assert_frame_equal(loaded["artists"], built["artists"])
        pd.testing.assert_frame_equal(reused["artists"], built["artists"])
        self.assertEqual(loaded["d"], 0.85)
        self.assertEqual(loaded["page_ranks"], built["page_ranks"])

    def test_missing_artifact_shows_preparation_command(self):
        with self.assertRaisesRegex(FileNotFoundError, "python -m scripts.pripremi --samo-rangovi"):
            pipeline.ucitaj_rangove()

    def test_newer_graph_invalidates_ranking_and_preparation_refreshes_it(self):
        pipeline.napravi_rangove()
        before = pipeline.GRAF_PATH.stat().st_mtime_ns - 1_000_000_000
        os.utime(pipeline.RANGOVI_PATH, ns=(before, before))
        with self.assertRaisesRegex(ValueError, "zastareli"):
            pipeline.ucitaj_rangove()
        pipeline.napravi_rangove()
        self.assertEqual(pipeline.ucitaj_rangove()["version"], 1)

    def test_changed_damping_regenerates_artifact(self):
        pipeline.napravi_rangove()
        changed = pipeline.napravi_rangove(d=0.5)
        self.assertEqual(changed["d"], 0.5)
        self.assertEqual(pipeline.ucitaj_rangove()["d"], 0.5)

    def test_new_seeds_require_graph_refresh_before_ranking(self):
        pd.DataFrame({"artist_name": ["Artist A"], "seed": ["artist-a"]}).to_parquet(
            pipeline.SEED_PATH, index=False)
        old = pipeline.SEED_PATH.stat().st_mtime_ns - 1_000_000_000
        os.utime(pipeline.GRAF_PATH, ns=(old, old))
        with patch("mrvectorbeat.search.ranking.artist_influence.pagerank", side_effect=AssertionError("PageRank")):
            with self.assertRaisesRegex(ValueError, "--samo-graf-seed"):
                pipeline.napravi_rangove()
        self.assertFalse(pipeline.RANGOVI_PATH.exists())


if __name__ == "__main__":
    unittest.main()
