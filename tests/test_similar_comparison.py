import math
import unittest
from unittest.mock import patch

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.lyrics_search import compare_lyrics, highlight_terms, search_similar_lyrics
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class SimilarComparisonTests(unittest.TestCase):
    def test_contributions_sum_to_displayed_similarity(self):
        lyrics = ["Love love night dream.", "Love night night.", "Dream alone.", ""]
        index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics])
        contributions = index.similarity_contributions(0, 1)
        expected = index.score_similar(0)[1]
        self.assertAlmostEqual(sum(contributions.values()), expected)
        comparison = compare_lyrics(index, 0, 1, lyrics[0], lyrics[1])
        self.assertAlmostEqual(comparison.similarity, expected)
        self.assertEqual([item.term for item in comparison.terms], ["love", "night"])
        self.assertEqual([(item.source_count, item.target_count) for item in comparison.terms],
                         [(2, 1), (1, 2)])
        self.assertAlmostEqual(sum(item.share for item in comparison.terms), 1)
        for item in comparison.terms:
            self.assertAlmostEqual(item.contribution, contributions[item.term])
            self.assertAlmostEqual(item.share, item.contribution / expected)

    def test_top_ten_percentages_use_all_shared_terms(self):
        words = [f"term{i}" for i in range(15)]
        lyrics = " ".join(words)
        index = PositionIndex([words, words, []])
        comparison = compare_lyrics(index, 0, 1, lyrics, lyrics)
        self.assertEqual(len(comparison.terms), 10)
        self.assertAlmostEqual(comparison.similarity, 1)
        self.assertAlmostEqual(sum(item.share for item in comparison.terms), 10 / 15)
        for item in comparison.terms:
            self.assertEqual((item.source_count, item.target_count), (1, 1))
            self.assertAlmostEqual(item.share, 1 / 15)
        self.assertEqual(comparison.source_html.count("<mark>"), 10)

    def test_highlighting_preserves_original_forms_and_escapes_html(self):
        text = "<script>evil</script> Running runs RUN!\n& runner"
        highlighted = highlight_terms(text, {"run"})
        self.assertIn("&lt;script&gt;evil&lt;/script&gt;", highlighted)
        self.assertNotIn("<script>", highlighted)
        self.assertIn("<mark>Running</mark> <mark>runs</mark> <mark>RUN</mark>!\n&amp; runner",
                      highlighted)
        self.assertEqual(highlighted.count("<mark>"), 3)
        self.assertEqual(highlight_terms("<b>& text", set()), "&lt;b&gt;&amp; text")

    def test_empty_disjoint_and_zero_idf_pairs(self):
        for documents, first, second in [([[], []], "", ""),
                                         ([["love"], ["night"]], "love", "night"),
                                         ([["love"], ["love"]], "love", "love")]:
            index = PositionIndex(documents)
            result = compare_lyrics(index, 0, 1, first, second)
            self.assertEqual(result.similarity, 0)
            self.assertEqual(result.terms, [])
            self.assertNotIn("<mark>", result.source_html)

    def test_comparison_reads_saved_statistics_for_only_selected_pair(self):
        index = PositionIndex([["love", "love", "night"], ["love"], ["alone"], []])
        expected = index.score_similar(0)[1]
        index.document_weights = {doc_id: index.document_weights[doc_id] for doc_id in (0, 1)}
        index.document_norms = {1: index.document_norms[1]}
        index.document_query_norms = {0: index.document_query_norms[0]}
        with patch("math.log10", side_effect=AssertionError("Ponovni IDF/TF")), \
                patch("math.sqrt", side_effect=AssertionError("Ponovna norma")), \
                patch.object(index, "score_similar", side_effect=AssertionError("Ponovno rangiranje")):
            comparison = compare_lyrics(index, 0, 1, "love love night", "love")
        self.assertAlmostEqual(comparison.similarity, expected)
        self.assertEqual(comparison.terms[0].source_count, 2)

    def test_similar_list_is_sorted_and_keeps_best_other_family_version(self):
        documents = [["love", "night"], ["love", "night"], ["love"],
                     ["love", "love", "night"], ["alone"], []]
        index = PositionIndex(documents, [2, 7, 13, 19, 23, 29])
        families = {2: "source", 7: "source", 13: "other", 19: "other", 23: "alone", 29: "empty"}
        result = search_similar_lyrics(index, 2, families=families, include_hits=False)
        scores = index.score_similar(2)
        best = max((13, 19), key=lambda doc_id: scores[doc_id])
        self.assertEqual(list(result.scores), [best])
        self.assertEqual(result.hits, {best: []})
        self.assertAlmostEqual(result.scores[best], scores[best])

    def test_pair_contributions_match_independent_lnc_ltc_formula(self):
        documents = [["love", "love", "night"], ["love", "night", "night"], ["night"]]
        index = PositionIndex(documents)
        tf = 1 + math.log10(2)
        idf = math.log10(3 / 2)
        # night je u svakom dokumentu: njegov IDF je 0 i ne doprinosi oceni.
        expected = (tf * idf) / ((tf * idf) * math.sqrt(1 + tf * tf))
        contributions = index.similarity_contributions(0, 1)
        self.assertEqual(set(contributions), {"love"})
        self.assertAlmostEqual(contributions["love"], expected)


if __name__ == "__main__":
    unittest.main()
