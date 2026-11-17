import math
import pickle
import tempfile
import unittest
from pathlib import Path
from random import Random
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from mrvectorbeat import pipeline
from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.lyrics_search import make_snippets, search_lyrics
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class BM25Tests(unittest.TestCase):
    def test_lengths_positive_idf_and_term_frequency(self):
        index = PositionIndex([["love", "love", "night"], ["love"], []], [2, 8, 11])
        self.assertEqual(index.N, 3)
        self.assertEqual(index.doc_len, {2: 3, 8: 1, 11: 0})
        self.assertAlmostEqual(index.avgdl(), 4 / 3)
        idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        self.assertAlmostEqual(index.idf_bm25("love"), idf)
        expected = idf * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / (4 / 3)))
        self.assertAlmostEqual(index.score_bm25(["love"], 2), expected)
        self.assertAlmostEqual(index.score_bm25(["love", "love"], 2), 2 * expected)
        self.assertEqual(index.idf_bm25("unknown"), 0)
        common = PositionIndex([["love"], ["love"]])
        self.assertGreater(common.idf_bm25("love"), 0)
        self.assertTrue(all(score > 0 for score in common.score_ranked(["love"], "bm25").values()))

    def test_scores_against_independent_reference(self):
        random = Random(41)
        documents = [[random.choice("abcd") for _ in range(random.randrange(12))]
                     for _ in range(20)]
        index = PositionIndex(documents)
        average = sum(map(len, documents)) / len(documents)
        for k1, b in ((1.5, 0.75), (0, 1), (2.3, 0), (0.5, 1)):
            for _ in range(15):
                tokens = [random.choice("abcde") for _ in range(random.randrange(6))]
                expected = {}
                for doc_id, document in enumerate(documents):
                    if not set(tokens).intersection(document):
                        continue
                    score = 0.0
                    for term in tokens:
                        frequency = document.count(term)
                        if frequency:
                            df = sum(term in other for other in documents)
                            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                            score += idf * frequency * (k1 + 1) / (
                                frequency + k1 * (1 - b + b * len(document) / average))
                    expected[doc_id] = score
                actual = index.score_ranked(tokens, "bm25", k1=k1, b=b)
                self.assertEqual(set(actual), set(expected))
                for doc_id, score in expected.items():
                    self.assertAlmostEqual(actual[doc_id], score)
                    self.assertAlmostEqual(index.score_bm25(tokens, doc_id, k1, b), score)

    def test_only_postings_candidates_and_saved_statistics_after_reload(self):
        index = PositionIndex([["love", "love"], ["night"], []], [2, 8, 19])
        expected = index.score_ranked(["love"], "bm25")
        loaded = pickle.loads(pickle.dumps(index))
        # Nepovezane pesme nemaju dostupnu duzinu; obilazak korpusa je zabranjen.
        class CandidateLengths(dict):
            def values(self):
                raise AssertionError("Ponovna prosecna duzina")

            def __iter__(self):
                raise AssertionError("Obilazak svih pesama")

            def items(self):
                raise AssertionError("Obilazak svih pesama")

        loaded.doc_len = CandidateLengths({2: loaded.doc_len[2]})
        loaded.document_ids = None
        with patch("math.log", side_effect=AssertionError("Ponovni IDF")), \
                patch("math.log1p", side_effect=AssertionError("Ponovni IDF")), \
                patch("math.log10", side_effect=AssertionError("Ponovni TF")):
            self.assertEqual(loaded.score_ranked(["love"], "bm25"), expected)
            self.assertEqual(loaded.score_bm25(["love"], 2), expected[2])

    def test_empty_corpus_unknown_terms_and_invalid_parameters(self):
        for documents in ([], [[], []], [["love"], []]):
            index = PositionIndex(documents)
            for tokens in ([], ["unknown"]):
                self.assertEqual(index.score_ranked(tokens, "bm25"), {})
                self.assertEqual(index.score_bm25(tokens, 0), 0)
        index = PositionIndex([["love"]])
        for k1, b in ((-1, 0.75), (1.5, -0.1), (1.5, 1.1),
                      (float("inf"), 0.75), (1.5, float("nan")), (True, 0.75)):
            with self.subTest(k1=k1, b=b):
                with self.assertRaises(ValueError):
                    index.score_ranked(["love"], "bm25", k1=k1, b=b)
                with self.assertRaises(ValueError):
                    index.score_bm25(["love"], 0, k1, b)

    def test_sorted_family_results_preprocessing_and_snippets(self):
        lyrics = ["Love night night night.", "Love.", "Love night.", "Alone."]
        index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics], [8, 2, 11, 19])
        result = search_lyrics(index, "LOVE", "ranked", measure="bm25",
                               families={8: "same", 2: "same", 11: "other"}, k1=1.2, b=1)
        self.assertEqual(list(result.scores), [2, 11])
        self.assertGreater(result.scores[2], result.scores[11])
        self.assertIn("<mark>Love</mark>", make_snippets(lyrics[1], result.hits[2])[0])
        self.assertEqual([doc_id for doc_id, _ in index.rank_bm25("LOVE", k1=1.2, b=1)], [2, 11, 8])
        ties = PositionIndex([["love"], ["love"]], [8, 2])
        self.assertEqual([doc_id for doc_id, _ in ties.rank_bm25("LOVE")], [2, 8])

    def test_old_artifact_requires_rebuild(self):
        index = PositionIndex([["love"]])
        index.ranking_version = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.pkl"
            path.write_bytes(pickle.dumps(index))
            with patch.multiple(pipeline, INDEKS_PATH=path,
                                KATALOG_PATH=Path(directory) / "catalog.parquet",
                                TOKENI_PATH=Path(directory) / "tokens.parquet"):
                with self.assertRaisesRegex(ValueError, "scripts.pripremi"):
                    pipeline.ucitaj_bm25()


class BM25AppTests(unittest.TestCase):
    def test_measure_parameters_cache_and_cosine_similar_songs(self):
        lyrics = ["Love.", "Love night night night.", "Alone."]
        ids = [2, 8, 19]
        index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics], ids)
        catalog = pd.DataFrame({
            "katalog_id": ids, "track_name": ["Short", "Long", "Other"],
            "track_artist": ["Artist"] * 3, "lyrics": lyrics,
            "porodica": ["short", "long", "other"],
            "godina": [2020] * 3, "playlist_genre": ["pop"] * 3,
        })
        st.cache_data.clear()
        st.cache_resource.clear()
        try:
            with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                    patch.object(pipeline, "ucitaj_indeks", return_value=index), \
                    patch.object(pipeline, "ucitaj_evaluaciju", return_value={
                        "bm25_selected": {"k1": 1.5, "b": 0.75}}), \
                    patch.object(pipeline, "ucitaj_trie", return_value=None), \
                    patch.object(index, "score_ranked", wraps=index.score_ranked) as ranked, \
                    patch.object(index, "score_similar", wraps=index.score_similar) as similar:
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                        default_timeout=20).run()
                app.sidebar.radio[0].set_value("Stihovi").run()
                next(r for r in app.radio if r.label == "Rezim pretrage").set_value("Rangirano").run()
                next(s for s in app.selectbox if s.label == "Mera slicnosti").set_value("BM25").run()
                app.text_input[0].set_value("love").run()
                self.assertFalse(app.exception)
                ranked.assert_called_once_with(["love"], "bm25", k1=1.5, b=0.75)
                self.assertTrue(any("Ocena (BM25)" in c.value for c in app.caption))
                next(b for b in app.button if b.key == "slicne_2").click().run()
                self.assertFalse(app.exception)
                similar.assert_called_once_with(2, "cosine")
                ranked.assert_called_once()
                next(n for n in app.number_input if n.label == "k1").set_value(2.0).run()
                self.assertFalse(app.exception)
                ranked.assert_called_with(["love"], "bm25", k1=2.0, b=0.75)
                self.assertIsNone(app.session_state.get("stihovi_slicne_izvor"))
                next(n for n in app.number_input if n.label == "b").set_value(0.0).run()
                self.assertFalse(app.exception)
                ranked.assert_called_with(["love"], "bm25", k1=2.0, b=0.0)
                expected = index.score_bm25(["love"], 2, k1=2.0, b=0.0)
                self.assertTrue(any(f"{expected:.6f}" in c.value for c in app.caption))
        finally:
            st.cache_data.clear()
            st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
