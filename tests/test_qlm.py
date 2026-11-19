import math
import pickle
import unittest
from unittest.mock import patch

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.ranking import QLM
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class QLMTests(unittest.TestCase):
    def setUp(self):
        self.index = PositionIndex(
            [["love", "love", "music"], ["music", "dance"], [], ["dance"]],
            [10, 20, 30, 40],
        )
        self.model = QLM(self.index, mu=3)

    def test_collection_probabilities_use_token_counts_and_include_all_documents(self):
        self.assertEqual(set(self.model.pc), {"love", "music", "dance"})
        for probability in self.model.pc.values():
            self.assertAlmostEqual(probability, 1 / 3)
        self.assertAlmostEqual(sum(self.model.pc.values()), 1.0)
        self.assertEqual(QLM(self.index).mu, 1500.0)

    def test_mle_uses_raw_frequency_instead_of_logarithmic_tf(self):
        self.assertNotEqual(self.index.tf("love", 10), 2)
        self.assertAlmostEqual(self.model.p_mle("love", 10), 2 / 3)
        self.assertAlmostEqual(self.model.p_mle("music", 20), 1 / 2)
        self.assertEqual(self.model.p_mle("love", 20), 0.0)
        self.assertEqual(self.model.p_mle("love", 30), 0.0)

    def test_dirichlet_probabilities_for_present_missing_and_empty_documents(self):
        self.assertAlmostEqual(self.model.p_dirichlet("love", 10), 1 / 2)
        self.assertAlmostEqual(self.model.p_dirichlet("love", 20), 1 / 5)
        self.assertAlmostEqual(self.model.p_dirichlet("love", 30), 1 / 3)
        self.assertAlmostEqual(self.model.p_dirichlet("dance", 40), 1 / 2)
        for doc_id in self.index.doc_len:
            self.assertAlmostEqual(sum(self.model.p_dirichlet(term, doc_id)
                                       for term in self.model.pc), 1.0)

    def test_log_scores_count_repeated_query_terms(self):
        self.assertAlmostEqual(self.model.score(["love", "love", "music"], 10),
                               2 * math.log(1 / 2) + math.log(1 / 3))
        self.assertAlmostEqual(self.model.score(["love", "love", "music"], 10, "mle"),
                               2 * math.log(2 / 3) + math.log(1 / 3))
        self.assertEqual(self.model.score(["love", "music"], 20, "mle"), -math.inf)

    def test_dirichlet_ranks_all_documents_including_those_without_query_words(self):
        ranked = self.model.rank(["love", "music"], topk=10)
        self.assertEqual([doc_id for doc_id, _ in ranked], [10, 30, 20, 40])
        expected = [math.log(1 / 6), math.log(1 / 9), math.log(2 / 25), math.log(1 / 16)]
        for (_, score), reference in zip(ranked, expected):
            self.assertAlmostEqual(score, reference)
        self.assertEqual(self.model.rank(["love", "music"], topk=2), ranked[:2])

    def test_mle_retains_negative_infinity_for_disqualified_documents(self):
        ranked = self.model.rank(["love", "music"], mode="mle")
        self.assertEqual([doc_id for doc_id, _ in ranked], [10, 20, 30, 40])
        self.assertAlmostEqual(ranked[0][1], math.log(2 / 9))
        self.assertTrue(all(score == -math.inf for _, score in ranked[1:]))

    def test_unseen_collection_words_are_not_smoothed_to_positive_probability(self):
        for mode in ("mle", "dir"):
            with self.subTest(mode=mode):
                self.assertEqual(self.model.score(["unknown"], 10, mode), -math.inf)
                self.assertEqual(self.model.score(["love", "unknown"], 10, mode), -math.inf)
                self.assertTrue(all(score == -math.inf
                                    for _, score in self.model.rank(["unknown"], mode)))
        self.assertEqual(self.model.p_dirichlet("unknown", 30), 0.0)

    def test_empty_corpus_and_empty_query_follow_the_scoring_definition(self):
        empty = QLM(PositionIndex([]))
        self.assertEqual(empty.pc, {})
        self.assertEqual(empty.rank(["love"]), [])
        only_empty_documents = QLM(PositionIndex([[], []]))
        self.assertEqual(only_empty_documents.pc, {})
        self.assertEqual(only_empty_documents.rank(["love"]), [(0, -math.inf), (1, -math.inf)])
        self.assertEqual(self.model.score([], 10), 0.0)
        self.assertEqual(self.model.rank([], topk=2), [(10, 0.0), (20, 0.0)])
        self.assertEqual(self.model.rank(["love"], topk=0), [])

    def test_raw_query_uses_the_same_preprocessing_as_the_index(self):
        lyrics = ["Singing songs and loving music.", "Love dancing!", "Music."]
        index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics])
        model = QLM(index)
        query = "LOVING, musical dancing!"
        tokens = TextProcessor(query).text_tokenize()
        for mode in ("mle", "dir"):
            self.assertEqual(model.rank(query, mode), model.rank(tokens, mode))
            self.assertEqual(model.score(query, 0, mode), model.score(tokens, 0, mode))
        with patch.object(TextProcessor, "text_tokenize", return_value=["love"]) as preprocess:
            self.model.rank("LOVE")
        preprocess.assert_called_once()

    def test_token_generator_is_consumed_once_for_the_whole_ranking(self):
        expected = self.model.rank(["love", "love", "music"])
        self.assertEqual(self.model.rank(term for term in ["love", "love", "music"]), expected)

    def test_saved_model_reuses_collection_statistics_without_rescanning_index(self):
        expected = self.model.rank(["love", "music"])
        model = pickle.loads(pickle.dumps(self.model))

        class LookupOnlyDirectory(dict):
            def items(self):
                raise AssertionError("Collection probabilities recomputed")

        class LookupOnlyLengths(dict):
            def values(self):
                raise AssertionError("Collection length recomputed")

        model.idx.directory = LookupOnlyDirectory(model.idx.directory)
        model.idx.doc_len = LookupOnlyLengths(model.idx.doc_len)
        with patch.object(model.idx, "tf", side_effect=AssertionError("Log-TF used as count")):
            self.assertEqual(model.rank(["love", "music"]), expected)
            self.assertEqual(model.rank(["love", "music"]), expected)

    def test_invalid_parameters_are_rejected(self):
        for mu in (0, -1, math.inf, math.nan, True, "1500"):
            with self.subTest(mu=mu), self.assertRaises(ValueError):
                QLM(self.index, mu)
        for mode in ("unknown", "dirichlet", None):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    self.model.score(["love"], 10, mode)
                with self.assertRaises(ValueError):
                    self.model.rank(["love"], mode)
        for topk in (-1, 1.5, True, "10"):
            with self.subTest(topk=topk), self.assertRaises(ValueError):
                self.model.rank(["love"], topk=topk)


if __name__ == "__main__":
    unittest.main()
