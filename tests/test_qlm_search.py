import math
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from mrvectorbeat import pipeline
from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.lyrics_search import query_word_coverage, search_lyrics_qlm
from mrvectorbeat.search.ranking.qlm import QLM
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class QLMLyricsSearchTests(unittest.TestCase):
    def setUp(self):
        self.lyrics = ["We found love in a hopeless place.",
                       "Without hope, love remains.", "Music and dance."]
        self.index = PositionIndex([TextProcessor(text).text_tokenize()
                                    for text in self.lyrics], [2, 7, 9])
        self.model = QLM(self.index, mu=3)

    def test_missing_query_words_do_not_disqualify_songs(self):
        query = "found love in a place without hope"
        results = search_lyrics_qlm(self.model, query)
        self.assertEqual(set(results.scores), {2, 7, 9})
        self.assertTrue(all(math.isfinite(score) for score in results.scores.values()))
        self.assertNotIn(2, self.index.find_boolean(results.terms, "AND"))
        # A hand-computed smoothed score for the song missing without and hope.
        total_length = sum(self.index.doc_len.values())
        expected = sum(math.log(
            (self.index.directory[term].get(2, 0)
             + 3 * sum(self.index.directory[term].values()) / total_length)
            / (self.index.doc_len[2] + 3)) for term in results.terms)
        self.assertAlmostEqual(results.scores[2], expected)
        self.assertEqual(list(results.scores.values()),
                         sorted(results.scores.values(), reverse=True))

    def test_collection_oov_words_do_not_destroy_known_word_ranking(self):
        expected = search_lyrics_qlm(self.model, "love love hope")
        actual = search_lyrics_qlm(self.model, "love love zzzunknown hope")
        self.assertEqual(actual.scores, expected.scores)
        self.assertIn("zzzunknown", actual.terms)
        self.assertEqual(query_word_coverage("love zzzunknown", self.lyrics[0]),
                         (["love"], ["zzzunknown"]))

    def test_empty_stopword_and_unknown_queries_do_not_return_arbitrary_songs(self):
        with patch.object(self.model, "rank", side_effect=AssertionError("No usable query")):
            for query in ("", "  ", "!!!", "the and", "zzzunknown"):
                with self.subTest(query=query):
                    self.assertFalse(search_lyrics_qlm(self.model, query).scores)

    def test_best_family_version_is_kept_without_losing_other_families(self):
        results = search_lyrics_qlm(self.model, "hope", {2: "song", 7: "song", 9: "other"})
        self.assertEqual(set(results.scores), {7, 9})
        self.assertGreater(results.scores[7], results.scores[9])

    def test_query_uses_index_preprocessing_and_keeps_repetitions(self):
        with patch.object(TextProcessor, "text_tokenize", return_value=["love", "love"]) as tokenize:
            result = search_lyrics_qlm(self.model, "LOVING loving!")
        tokenize.assert_called_once()
        self.assertEqual(result.terms, ["love", "love"])
        self.assertAlmostEqual(result.scores[2], 2 * self.model.score(["love"], 2))

    def test_coverage_uses_whole_original_words_including_stopwords(self):
        present, missing = query_word_coverage("found love in a place without hope", self.lyrics[0])
        self.assertEqual(present, ["found", "love", "in", "a", "place"])
        self.assertEqual(missing, ["without", "hope"])
        self.assertEqual(query_word_coverage("LOVING love hope", "Love hopeless."),
                         (["love"], ["loving", "hope"]))

    def test_coverage_normalizes_case_punctuation_and_deduplicates_in_input_order(self):
        self.assertEqual(query_word_coverage("LOVE, love! DON'T hope/hope", "Love; don’t dance."),
                         (["love", "dont"], ["hope"]))
        self.assertEqual(query_word_coverage("the and", "THE music"), (["the"], ["and"]))
        self.assertEqual(query_word_coverage("", "Music"), ([], []))
        self.assertEqual(query_word_coverage("hope", ""), ([], ["hope"]))


class QLMLyricsAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.addCleanup(st.cache_data.clear)
        self.addCleanup(st.cache_resource.clear)

    def app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                default_timeout=20)
        app.session_state["strana"] = "Jezicki model"
        return app

    def test_search_shows_partial_matches_and_present_missing_words(self):
        lyrics = ["We found love in a hopeless place.",
                  "Without hope, love remains.", "Music and dance."]
        index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics], [2, 7, 9])
        catalog = pd.DataFrame({
            "katalog_id": [2, 7, 9], "track_name": ["Target", "Hope", "Dance"],
            "track_artist": ["Artist"] * 3, "lyrics": lyrics,
            "porodica": ["target", "hope", "dance"],
        })
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(pipeline, "ucitaj_indeks", return_value=index), \
                patch.object(pipeline, "ucitaj_trie", return_value=None), \
                patch("mrvectorbeat.search.ranking.wordnet_expansion._wordnet_synonyms", return_value=()), \
                patch.object(QLM, "rank", autospec=True, side_effect=QLM.rank) as rank:
            app = self.app().run()
            self.assertFalse(app.exception)
            rank.assert_not_called()
            app.text_input(key="jezicki_upit").set_value("found love in a place without hope").run()
            self.assertFalse(app.exception)
            rank.assert_called_once()
            self.assertEqual(rank.call_args.kwargs["mode"], "dir")
            self.assertEqual({text.value for text in app.text},
                             {"Target — Artist", "Hope — Artist", "Dance — Artist"})
            captions = [caption.value for caption in app.caption if caption.value.startswith("QLM ocena")]
            self.assertEqual(len(captions), 3)
            self.assertFalse(any("inf" in caption for caption in captions))
            markdown = [item.value for item in app.markdown]
            self.assertIn("Sadrži: *found*, *love*, *in*, *a*, *place*", markdown)
            self.assertIn("Nedostaje: *without*, *hope*", markdown)
            self.assertEqual(sum(value.startswith("Sadrži:") for value in markdown), 3)
            self.assertEqual(sum(value.startswith("Nedostaje:") for value in markdown), 3)
            app.selectbox(key="jezicki_velicina").set_value(20).run()
            self.assertFalse(app.exception)
            rank.assert_called_once()  # Pagination reuses the saved query result.
            app.text_input(key="jezicki_upit").set_value("love zzzunknown").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("*zzzunknown*" in item.value for item in app.markdown))
            self.assertFalse(any("inf" in caption.value for caption in app.caption))
            app.text_input(key="jezicki_upit").set_value("zzzunknown").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("ne postoji u katalogu" in info.value for info in app.info))
            self.assertEqual(len(app.text), 0)
            app.text_input(key="jezicki_upit").set_value("!!!").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("nije ostao nijedan termin" in info.value for info in app.info))

    def test_missing_index_shows_preparation_message(self):
        with patch.object(pipeline, "ucitaj_katalog", side_effect=FileNotFoundError()):
            app = self.app().run()
            app.text_input(key="jezicki_upit").set_value("love").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("scripts.pripremi" in info.value for info in app.info))
        self.assertEqual(len(app.text), 0)


if __name__ == "__main__":
    unittest.main()
