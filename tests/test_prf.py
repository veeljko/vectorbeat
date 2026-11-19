import math
import pickle
import unittest
from unittest.mock import patch

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.ranking import PRFRanking, QLM, prf_terms
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class PRFRankingTests(unittest.TestCase):
    def setUp(self):
        self.index = PositionIndex([
            ["seed", "rare", "rare", "common"],
            ["seed", "common", "common", "other"],
            ["common"] * 8,
        ], [10, 20, 30])
        self.model = PRFRanking(self.index, mu=4)

    def test_selects_new_positive_kl_terms_from_initial_qlm_top_documents(self):
        terms, used = self.model.prf_terms(["seed"], k=2, m=5)
        self.assertEqual(used, [10, 20])
        self.assertEqual([term for term, _ in terms], ["rare", "other"])
        self.assertAlmostEqual(terms[0][1], 0.5 * math.log(0.5 / (2 / 16)))
        self.assertAlmostEqual(terms[1][1], 0.25 * math.log(0.25 / (1 / 16)))
        self.assertNotIn("common", dict(terms))
        self.assertNotIn("seed", dict(terms))

    def test_limits_number_of_terms_and_uses_raw_tf_counts(self):
        self.assertNotEqual(self.index.tf("rare", 10), 2)
        with patch.object(self.index, "tf", side_effect=AssertionError("Log TF used as a count")):
            terms, used = self.model.prf_terms(["seed"], k=1, m=1)
        self.assertEqual(used, [10])
        self.assertEqual([term for term, _ in terms], ["rare"])
        self.assertAlmostEqual(terms[0][1], math.log(2))

    def test_negative_individual_contributions_are_included_before_selection(self):
        index = PositionIndex([["seed", "term", "term"],
                               ["seed"] + ["filler"] * 8 + ["term"],
                               ["filler"] * 7])
        model = PRFRanking(index, mu=3)
        terms, used = model.prf_terms(["seed"], k=2, m=10)
        self.assertEqual(used, [0, 1])
        expected = (2 / 3) * math.log((2 / 3) / (3 / 20))
        expected += (1 / 10) * math.log((1 / 10) / (3 / 20))
        self.assertAlmostEqual(dict(terms)["term"], expected)

    def test_final_ranking_uses_expanded_query_with_the_same_dirichlet_model(self):
        original = self.model.model.rank(["seed"])
        ranked = self.model.rank(["seed"], k=2, m=1, topk=2)
        self.assertEqual([doc_id for doc_id, _ in ranked], [10, 20])
        self.assertAlmostEqual(ranked[0][1], math.log(3 / 16) + math.log(5 / 16))
        self.assertAlmostEqual(ranked[1][1], math.log(3 / 16) + math.log(1 / 16))
        self.assertNotEqual(ranked, original[:2])
        self.assertEqual(ranked, self.model.model.rank(["seed", "rare"], topk=2))

    def test_original_query_repetitions_and_input_are_preserved(self):
        query = ["seed", "seed"]
        ranked = self.model.rank(query, k=2, m=1)
        self.assertAlmostEqual(ranked[0][1], 2 * math.log(3 / 16) + math.log(5 / 16))
        self.assertEqual(query, ["seed", "seed"])
        self.assertEqual(ranked, self.model.rank((term for term in query), k=2, m=1))

    def test_string_is_preprocessed_once_for_both_ranking_passes(self):
        with patch.object(TextProcessor, "text_tokenize", return_value=["seed", "seed"]) as preprocess:
            actual = self.model.rank("SEED SEEDS", k=2, m=1)
        preprocess.assert_called_once()
        self.assertEqual(actual, self.model.rank(["seed", "seed"], k=2, m=1))

    def test_zero_k_and_m_preserve_the_original_qlm_ranking(self):
        expected = QLM(self.index, mu=4).rank(["seed"], topk=2)
        self.assertEqual(self.model.rank(["seed"], k=0, topk=2), expected)
        self.assertEqual(self.model.rank(["seed"], m=0, topk=2), expected)
        self.assertEqual(self.model.prf_terms(["seed"], k=0), ([], []))
        self.assertEqual(self.model.prf_terms(["seed"], k=2, m=0), ([], [10, 20]))
        self.assertEqual(self.model.rank(["seed"], topk=0), [])

    def test_empty_and_unseen_queries_do_not_create_arbitrary_feedback(self):
        for query in ([], ["unknown"], ["seed", "unknown"]):
            with self.subTest(query=query):
                self.assertEqual(self.model.prf_terms(query), ([], []))
                self.assertEqual(self.model.rank(query), [])
        empty = PRFRanking(PositionIndex([]))
        self.assertEqual(empty.prf_terms(["seed"]), ([], []))
        self.assertEqual(empty.rank(["seed"]), [])

    def test_empty_feedback_documents_and_requests_larger_than_the_corpus_are_safe(self):
        index = PositionIndex([["seed", "rare"], [], ["noise"]])
        model = PRFRanking(index, mu=1)
        terms, used = model.prf_terms(["seed"], k=100, m=100)
        self.assertEqual(len(used), 3)
        self.assertIn(1, used)
        self.assertTrue(all(math.isfinite(score) for _, score in terms))
        self.assertEqual(len(model.rank(["seed"], k=100, m=100, topk=100)), 3)

    def test_equal_contributions_use_alphabetical_term_order(self):
        index = PositionIndex([["seed", "banana"], ["seed", "apple"], ["noise"]])
        terms, used = PRFRanking(index, mu=1).prf_terms(["seed"], k=2, m=1)
        self.assertEqual(used, [0, 1])
        self.assertEqual([term for term, _ in terms], ["apple"])

    def test_collection_probabilities_are_reused_and_only_feedback_terms_are_inspected(self):
        model = pickle.loads(pickle.dumps(self.model))
        expected = model.prf_terms(["seed"], k=2)

        class LookupOnlyDirectory(dict):
            def items(self):
                raise AssertionError("Collection probabilities recomputed")

        model.idx.directory = LookupOnlyDirectory(model.idx.directory)
        model.idx.doc_terms = {10: model.idx.doc_terms[10], 20: model.idx.doc_terms[20]}
        pc = model.model.pc
        with patch.object(model.idx, "tf", side_effect=AssertionError("Logarithmic TF used")):
            self.assertEqual(model.prf_terms(["seed"], k=2), expected)
            self.assertEqual(model.prf_terms(["seed"], k=2), expected)
            model.rank(["seed"], k=2)
        self.assertIs(model.model.pc, pc)

    def test_function_wrapper_matches_the_reusable_model(self):
        self.assertEqual(prf_terms(self.index, ["seed"], k=2, m=1, mu=4),
                         self.model.prf_terms(["seed"], k=2, m=1))

    def test_invalid_parameters_are_rejected(self):
        for value in (-1, 1.5, True, "5"):
            for name in ("k", "m"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.model.prf_terms(["seed"], **{name: value})
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.model.rank(["seed"], **{name: value})
            with self.assertRaises(ValueError):
                self.model.rank(["seed"], topk=value)
        for mu in (0, -1, math.inf, math.nan, True):
            with self.subTest(mu=mu), self.assertRaises(ValueError):
                PRFRanking(self.index, mu=mu)


if __name__ == "__main__":
    unittest.main()
