import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from mrvectorbeat import pipeline
from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.lyrics_search import search_lyrics_qlm, search_lyrics_with_synonyms
from mrvectorbeat.search.ranking.qlm import QLM
from mrvectorbeat.search.ranking.rocchio.rocchio import rocchio
from mrvectorbeat.search.ranking.wordnet_expansion import (
    _wordnet_synonyms, expand_with_wordnet, needs_query_expansion,
)
from mrvectorbeat.textprocessing.text_processor import TextProcessor


MODULE = "mrvectorbeat.search.ranking.wordnet_expansion"


def make_index(lyrics):
    return PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics])


def club_synonyms(word):
    return {"love": ("love",), "nightclub": ("nightclub", "club")}.get(word, ())


class WordNetExpansionTests(unittest.TestCase):
    def test_quality_gate_counts_unique_terms_oov_and_checks_first_five(self):
        index = PositionIndex([["first", "second", "third", "fourth"], ["first"], [], [], [],
                               ["first", "second", "third", "fourth", "fifth"]])
        terms = ["first", "second", "third", "fourth", "fifth"]
        self.assertFalse(needs_query_expansion(index, terms, [0]))  # Exactly 80%.
        self.assertTrue(needs_query_expansion(index, terms + ["unknown"], [0]))
        self.assertFalse(needs_query_expansion(index, ["first"] * 10 + terms, [0]))
        self.assertFalse(needs_query_expansion(index, terms, [1, 2, 3, 4, 0]))
        self.assertTrue(needs_query_expansion(index, terms, [1, 2, 3, 4, 2, 5]))
        self.assertTrue(needs_query_expansion(index, ["unknown"], []))
        self.assertFalse(needs_query_expansion(index, [], []))

    def test_reader_uses_synset_lemmas_limits_senses_and_skips_phrases(self):
        def synset(pos, names):
            return SimpleNamespace(pos=lambda: pos, lemma_names=lambda: names)

        fake_wordnet = SimpleNamespace(synsets=lambda word: [
            synset("n", ["Beloved", "love", "night_club"]),
            synset("n", ["dear", "love"]), synset("n", ["excluded"]),
            synset("v", ["enjoy"]), synset("s", ["darling"]),
            synset("a", ["honey"]), synset("a", ["excludedadj"]),
        ])
        _wordnet_synonyms.cache_clear()
        self.addCleanup(_wordnet_synonyms.cache_clear)
        with patch("nltk.corpus.wordnet", fake_wordnet):
            self.assertEqual(_wordnet_synonyms("beloved"),
                             ("beloved", "love", "dear", "enjoy", "darling", "honey"))

    def test_unstemmed_lookup_deduplicates_stems_and_preserves_original_weights(self):
        index = make_index(["love love dear honey", "night music"])
        original = {"night": 2}
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=lambda word:
                   {"beloved": ("beloved", "loving", "love", "loved", "dear", "absent")}.get(word, ())) as lookup:
            weights, words = expand_with_wordnet(index, "THE beloved night night", original)
        self.assertEqual([call.args[0] for call in lookup.call_args_list], ["beloved", "night"])
        self.assertEqual(words, ["loving", "dear"])
        self.assertEqual(weights, {"night": 2, "love": 0.25, "dear": 0.25})
        self.assertEqual(original, {"night": 2})

    def test_rocchio_prefers_supported_wordnet_terms_without_adding_other_feedback_terms(self):
        index = make_index(["night love intruder", "honey"])
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=lambda word:
                   ("honey", "love") if word == "beloved" else ()), \
                patch(f"{MODULE}.rocchio", wraps=rocchio) as feedback:
            weights, words = expand_with_wordnet(index, "night beloved", {"night": 1}, [0], max_terms=1)
        self.assertEqual(words, ["love"])
        self.assertEqual(weights, {"night": 1, "love": 0.25})
        self.assertNotIn("intruder", weights)
        feedback.assert_called_once()
        self.assertEqual(feedback.call_args.args[2:], ([0], []))
        self.assertEqual(feedback.call_args.kwargs, {"alpha": 1.0, "beta": 0.75, "gamma": 0.0})

    def test_expansion_caps_total_and_per_original_word(self):
        synonyms = {"nightclub": ("club", "cabaret", "nightspot"),
                    "beloved": ("love", "honey", "dear"),
                    "automobile": ("car", "auto", "motorcar")}
        index = make_index(["club cabaret nightspot love honey dear car auto motorcar"])
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=lambda word: synonyms.get(word, ())):
            weights, words = expand_with_wordnet(index, "nightclub beloved automobile", {})
        self.assertEqual(words, ["club", "cabaret", "love", "honey", "car"])
        self.assertEqual(len(weights), 5)
        self.assertTrue(all(weight == 0.25 for weight in weights.values()))

    def test_skip_synonyms_equivalent_to_typed_terms_or_outside_collection(self):
        index = make_index(["love music"])
        with patch(f"{MODULE}._wordnet_synonyms", return_value=("loving", "loved", "unknown", "night_club")), \
                patch(f"{MODULE}.rocchio", side_effect=AssertionError("No new candidates")):
            self.assertEqual(expand_with_wordnet(index, "love", {"love": 1}), ({"love": 1}, []))

    def test_validation(self):
        index = make_index(["music"])
        for weight in (0, 1, -1, math.inf, math.nan, True, "0.25"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                expand_with_wordnet(index, "music", {}, synonym_weight=weight)
        for size in (-1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                expand_with_wordnet(index, "music", {}, max_terms=size)
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=AssertionError("Disabled")):
            self.assertEqual(expand_with_wordnet(index, "music", {"music": 1}, max_terms=0),
                             ({"music": 1}, []))


class AutomaticQLMExpansionTests(unittest.TestCase):
    def setUp(self):
        self.index = make_index(["love night night night", "love club club club club", "music"])
        self.model = QLM(self.index, mu=3)

    def test_poor_results_are_reranked_once_with_lower_weighted_synonym(self):
        initial = search_lyrics_qlm(self.model, "love nightclub")
        self.assertEqual(next(iter(initial.scores)), 0)
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=club_synonyms), \
                patch.object(self.model, "rank", wraps=self.model.rank) as rank:
            result = search_lyrics_with_synonyms(self.model, "love nightclub")
        self.assertEqual(rank.call_count, 2)
        self.assertEqual(rank.call_args.args[0], {"love": 1, "club": 0.25})
        self.assertEqual(next(iter(result.scores)), 1)
        self.assertEqual(result.added_words, ["club"])
        self.assertEqual(result.terms, ["love", "nightclub"])
        self.assertEqual(set(result.scores), {0, 1, 2})
        for doc_id, score in result.scores.items():
            expected = math.log(self.model.p_dirichlet("love", doc_id))
            expected += 0.25 * math.log(self.model.p_dirichlet("club", doc_id))
            self.assertAlmostEqual(score, expected)

    def test_good_results_do_not_lookup_wordnet_or_rerank(self):
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=AssertionError("Good coverage")), \
                patch.object(self.model, "rank", wraps=self.model.rank) as rank:
            result = search_lyrics_with_synonyms(self.model, "love night")
        self.assertEqual(rank.call_count, 1)
        self.assertEqual(result.added_words, [])

    def test_all_oov_query_can_be_rescued_by_wordnet(self):
        with patch(f"{MODULE}._wordnet_synonyms", return_value=("club",)), \
                patch.object(self.model, "rank", wraps=self.model.rank) as rank:
            result = search_lyrics_with_synonyms(self.model, "nightclub")
        rank.assert_called_once()
        self.assertEqual(rank.call_args.args[0], {"club": 0.25})
        self.assertEqual(next(iter(result.scores)), 1)
        self.assertEqual(result.added_words, ["club"])
        self.assertTrue(all(math.isfinite(score) for score in result.scores.values()))

    def test_no_usable_synonyms_preserves_initial_ranking_without_second_pass(self):
        initial = search_lyrics_qlm(self.model, "love zzzunknown")
        with patch(f"{MODULE}._wordnet_synonyms", return_value=()), \
                patch.object(self.model, "rank", wraps=self.model.rank) as rank:
            result = search_lyrics_with_synonyms(self.model, "love zzzunknown")
        rank.assert_called_once()
        self.assertEqual(result, initial)
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=AssertionError("Empty query")):
            for query in ("", "!!!", "the and"):
                self.assertEqual(search_lyrics_with_synonyms(self.model, query).scores, {})

    def test_missing_dictionary_preserves_initial_results_with_install_hint(self):
        initial = search_lyrics_qlm(self.model, "love nightclub")
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=LookupError("not installed")):
            result = search_lyrics_with_synonyms(self.model, "love nightclub")
        self.assertEqual(result.scores, initial.scores)
        self.assertEqual(result.added_words, [])
        self.assertIn("python -m nltk.downloader wordnet", result.expansion_warning)

    def test_expanded_results_still_show_only_best_version_per_family(self):
        with patch(f"{MODULE}._wordnet_synonyms", side_effect=club_synonyms):
            result = search_lyrics_with_synonyms(self.model, "love nightclub", {0: "same", 1: "same", 2: "other"})
        self.assertEqual(result.added_words, ["club"])
        self.assertEqual(set(result.scores), {1, 2})


class WordNetExpansionAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.addCleanup(st.cache_data.clear)
        self.addCleanup(st.cache_resource.clear)

    def test_automatic_expansion_notice_is_above_results_and_clears_for_good_query(self):
        lyrics = ["love night night night", "love club club club club", "music"]
        index = make_index(lyrics)
        catalog = pd.DataFrame({
            "katalog_id": [0, 1, 2], "track_name": ["Night", "Club", "Other"],
            "track_artist": ["Artist"] * 3, "lyrics": lyrics,
            "porodica": ["night", "club", "other"],
        })
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(pipeline, "ucitaj_indeks", return_value=index), \
                patch.object(pipeline, "ucitaj_trie", return_value=None), \
                patch(f"{MODULE}._wordnet_synonyms", side_effect=club_synonyms), \
                patch.object(QLM, "rank", autospec=True, side_effect=QLM.rank) as rank:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
            app.session_state["strana"] = "Jezicki model"
            app.run()
            app.text_input(key="jezicki_upit").set_value("love nightclub").run()
            self.assertFalse(app.exception)
            self.assertEqual(rank.call_count, 2)
            self.assertEqual(rank.call_args.args[1], {"love": 1, "club": 0.25})
            markdown = [item.value for item in app.markdown]
            notice = "Uključeni su i rezultati za: *club*"
            self.assertIn(notice, markdown)
            self.assertLess(markdown.index(notice), markdown.index("Nedostaje: *nightclub*"))
            self.assertIn("Sadrži: *love*", markdown)  # Added synonyms aren't user words.
            app.selectbox(key="jezicki_velicina").set_value(20).run()
            self.assertFalse(app.exception)
            self.assertEqual(rank.call_count, 2)
            app.text_input(key="jezicki_upit").set_value("love night").run()
            self.assertFalse(app.exception)
            self.assertEqual(rank.call_count, 3)
            self.assertFalse(any(item.value.startswith("Uključeni su i rezultati za:") for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
