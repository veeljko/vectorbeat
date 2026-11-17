import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from mrvectorbeat import pipeline
from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.lyrics_search import search_lyrics
from mrvectorbeat.search.ranking.evaluation import (
    GROUPS, MODELS, build_evaluation, group_tables, metric_grades, prepare_queries, query_metrics,
)
from mrvectorbeat.textprocessing.text_processor import TextProcessor


def fixture():
    ids = [2, 7, 11, 19, 23]
    lyrics = ["Love night.", "Love love.", "Love day.", "Night.", ""]
    catalog = pd.DataFrame({
        "katalog_id": ids, "track_id": ["original", "live", "other", "night", "empty"],
        "track_name": ["Source", "Source - Live", "Other", "Night", "Empty"],
        "track_artist": ["Artist"] * 5, "porodica": ["source", "source", "other", "night", "empty"],
        "lyrics": lyrics, "godina": [2020] * 5, "playlist_genre": ["pop"] * 5,
    })
    qrels = pd.DataFrame([
        ("poznata_stavka", "love", "original", 2, 0),
        ("poznata_stavka", "love", "live", 2, 0),
        ("poznata_stavka", "love", "other", 1, 0),
        ("poznata_stavka", "zzzznomatch", "original", 2, 1),
        ("tema", "night", "original", 2, 0),
        ("tema", "night", "night", 1, 0),
    ], columns=["skup", "upit", "track_id", "ocena", "tipfeler"])
    index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics], ids)
    return index, catalog, qrels


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.index, self.catalog, self.qrels = fixture()

    def build(self):
        return build_evaluation(self.index, self.catalog, self.qrels,
                                k1_values=(0.3, 1.5), b_values=(0.0, 0.5, 1.0))

    def test_family_judgments_and_known_item_only_grade_two(self):
        query = prepare_queries(self.qrels, self.catalog)[0]
        self.assertEqual(query["grades"], {"source": 2, "other": 1})
        self.assertEqual(metric_grades(query), {"source": 2, "other": 0})
        songs = {int(row.katalog_id): {"porodica": row.porodica}
                 for row in self.catalog.itertuples(index=False)}
        # Live verzija nasledjuje ocenu originala; ocena 1 na prvom mestu nije pogodak.
        metrics = query_metrics(query, [11, 7], songs)
        self.assertEqual(metrics["RR"], 0.5)
        self.assertEqual(metrics["AP"], 0.5)
        self.assertEqual(metrics["P@5"], 0.2)
        self.assertGreater(metrics["nDCG@5"], 0)
        self.assertTrue(all(value == 0 for value in query_metrics(query, [11], songs).values()))

    def test_saved_rankings_equal_lyrics_results_and_typos_stay_unknown(self):
        artifact = self.build()
        families = self.catalog.set_index("katalog_id")["porodica"].to_dict()
        selected = artifact["bm25_selected"]
        for query in artifact["queries"]:
            for model in MODELS:
                result = search_lyrics(self.index, query["upit"], "ranked", measure=model,
                                       families=families, k1=selected["k1"], b=selected["b"])
                self.assertEqual(query["rankings"][model], list(result.hits))
                self.assertEqual(len(result.hits), len({families[d] for d in result.hits}))
                if query["tipfeler"]:
                    self.assertEqual(query["rankings"][model], [])
        self.assertEqual(len(artifact["bm25_grid"]), 6)
        best = min(artifact["bm25_grid"], key=lambda row: (-row["MRR"], row["b"], row["k1"]))
        self.assertEqual(selected, best)

    def test_group_tables_and_graded_topic_metrics(self):
        artifact = self.build()
        tables = group_tables(artifact)
        self.assertEqual(set(tables), set(GROUPS))
        for table in tables.values():
            self.assertEqual(table.shape, (3, 4))
            self.assertEqual(list(table.columns), ["P@5", "MRR", "MAP", "nDCG@5"])
        self.assertTrue((tables["poznata_sa_greskom"] == 0).all().all())
        topic = next(q for q in artifact["queries"] if q["skup"] == "tema")
        self.assertEqual(metric_grades(topic), {"source": 2, "night": 1})
        metrics = query_metrics(topic, [19], artifact["songs"])
        self.assertEqual(metrics["RR"], 1)
        self.assertEqual(metrics["AP"], 0.5)
        self.assertAlmostEqual(metrics["nDCG@5"], 1 / (2 + 1 / math.log2(3)))

    def test_pipeline_roundtrip_is_read_only_and_rejects_stale_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "qrels.csv"
            self.qrels.to_csv(source, index=False)
            with patch.multiple(pipeline, EVALUACIJA_PATH=root / "evaluation.pkl", QRELS_PATH=source,
                                KATALOG_PATH=root / "catalog.parquet", INDEKS_PATH=root / "index.pkl"), \
                    patch.object(pipeline, "ucitaj_indeks", return_value=self.index), \
                    patch.object(pipeline, "ucitaj_katalog", return_value=self.catalog):
                self.assertEqual(len(pipeline.ucitaj_qrels("tema")), 2)
                artifact = pipeline.napravi_evaluaciju()
                with patch("pandas.read_csv", side_effect=AssertionError("Raw file read")), \
                        patch.object(self.index, "score_ranked", side_effect=AssertionError("Ranking")), \
                        patch("mrvectorbeat.search.ranking.evaluation.build_evaluation",
                              side_effect=AssertionError("Preparation")):
                    loaded = pipeline.ucitaj_evaluaciju()
                    self.assertEqual(loaded, artifact)
                    self.assertEqual(len(group_tables(loaded)), 3)
                stamp = pipeline.EVALUACIJA_PATH.stat().st_mtime_ns + 1_000_000_000
                os.utime(source, ns=(stamp, stamp))
                with self.assertRaisesRegex(ValueError, "zastarela"):
                    pipeline.ucitaj_evaluaciju()

    def test_missing_track_and_no_tuning_queries_are_reported(self):
        bad = self.qrels.copy()
        bad.loc[0, "track_id"] = "missing"
        with self.assertRaises(ValueError):
            prepare_queries(bad, self.catalog)
        with self.assertRaises(ValueError):
            build_evaluation(self.index, self.catalog, self.qrels[self.qrels.skup == "tema"])


class EvaluationAppTests(unittest.TestCase):
    def setUp(self):
        self.index, self.catalog, self.qrels = fixture()
        self.artifact = build_evaluation(self.index, self.catalog, self.qrels,
                                         k1_values=(0.3, 1.5), b_values=(0.0, 0.5, 1.0))
        st.cache_data.clear()
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)

    def test_page_uses_only_artifact_with_three_tables_and_colored_results(self):
        with patch.object(pipeline, "ucitaj_evaluaciju", return_value=self.artifact), \
                patch.object(pipeline, "ucitaj_indeks", side_effect=AssertionError("Model loading")), \
                patch.object(pipeline, "ucitaj_katalog", side_effect=AssertionError("Catalog loading")), \
                patch.object(pipeline, "ucitaj_trie", side_effect=AssertionError("Trie loading")), \
                patch.object(self.index, "score_ranked", side_effect=AssertionError("Ranking")):
            app = self.app()
            app.session_state["strana"] = "Evaluacija"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 3)
            for table in app.dataframe:
                self.assertEqual(table.value.shape, (3, 4))
            self.assertEqual(len(app.get("column")), 3)
            self.assertTrue(any("#138a36" in m.value for m in app.markdown))
            self.assertTrue(any("#2563eb" in m.value for m in app.markdown))
            app.selectbox[0].set_value(1).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Nema rezultata" in i.value for i in app.info))
            self.assertEqual(len(app.dataframe), 3)

    def test_lyrics_bm25_starts_with_selected_grid_parameters(self):
        with patch.object(pipeline, "ucitaj_evaluaciju", return_value=self.artifact), \
                patch.object(pipeline, "ucitaj_indeks", return_value=self.index), \
                patch.object(pipeline, "ucitaj_katalog", return_value=self.catalog), \
                patch.object(pipeline, "ucitaj_trie", return_value=None), \
                patch.object(self.index, "score_ranked", wraps=self.index.score_ranked) as ranked:
            app = self.app().run()
            app.sidebar.radio[0].set_value("Stihovi").run()
            next(r for r in app.radio if r.label == "Rezim pretrage").set_value("Rangirano").run()
            next(s for s in app.selectbox if s.label == "Mera slicnosti").set_value("BM25").run()
            app.text_input[0].set_value("love").run()
            self.assertFalse(app.exception)
            selected = self.artifact["bm25_selected"]
            ranked.assert_called_with(["love"], "bm25", k1=selected["k1"], b=selected["b"])
            self.assertEqual(next(n for n in app.number_input if n.label == "k1").value, selected["k1"])
            self.assertEqual(next(n for n in app.number_input if n.label == "b").value, selected["b"])


if __name__ == "__main__":
    unittest.main()
