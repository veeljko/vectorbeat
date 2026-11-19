import math
import pickle
import unittest
from unittest.mock import patch

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.ranking import pseudo_relevance_feedback, rank_vector, rocchio
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class PseudoRelevanceFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.index = PositionIndex([["love"], ["love", "dance"], ["night"]], [2, 8, 19])
        self.q0 = {"love": 1.0}

    def test_vector_ranking_matches_existing_cosine_search(self):
        text = "LOVE loving DANCING!"
        tokens = TextProcessor(text).text_tokenize()
        q0, _ = self.index.build_vec(tokens, use_idf=True)
        expected = sorted(self.index.score_ranked(tokens, "cosine").items(),
                          key=lambda pair: (-pair[1], pair[0]))
        actual = rank_vector(self.index, q0)
        self.assertEqual([doc_id for doc_id, _ in actual], [doc_id for doc_id, _ in expected])
        for (_, score), (_, reference) in zip(actual, expected):
            self.assertAlmostEqual(score, reference)

    def test_prf_uses_initial_top_k_and_positive_feedback_without_mutating_input(self):
        query = self.q0.copy()
        original_documents = {doc_id: vector.copy() for doc_id, vector in self.index.document_weights.items()}
        q2, used = pseudo_relevance_feedback(self.index, query, k=1)
        self.assertEqual(used, [2])
        self.assertEqual(q2, {"love": 1.75})
        self.assertEqual(query, self.q0)
        self.assertIsNot(q2, query)
        self.assertEqual(self.index.document_weights, original_documents)

    def test_centroid_divides_by_actual_hits_when_fewer_than_k_match(self):
        q2, used = pseudo_relevance_feedback(self.index, self.q0, k=5)
        self.assertEqual(used, [2, 8])
        love_idf, dance_idf = math.log10(3 / 2), math.log10(3)
        norm = math.sqrt(love_idf ** 2 + dance_idf ** 2)
        self.assertAlmostEqual(q2["love"], 1 + 0.75 / 2 * (1 + love_idf / norm))
        self.assertAlmostEqual(q2["dance"], 0.75 / 2 * dance_idf / norm)

    def test_k_five_and_ten_use_different_documents_and_expansion_terms(self):
        index = PositionIndex([["palm"] * 4 + ["beach"] for _ in range(5)]
                              + [["palm"] + ["oil"] * 4 for _ in range(5)]
                              + [["other"]])
        q0, _ = index.build_vec(["palm"], use_idf=True)
        q5, used5 = pseudo_relevance_feedback(index, q0, k=5)
        q10, used10 = pseudo_relevance_feedback(index, q0, k=10)
        self.assertEqual(used5, list(range(5)))
        self.assertEqual(used10, list(range(10)))
        self.assertIn("beach", q5)
        self.assertNotIn("oil", q5)
        self.assertIn("oil", q10)
        self.assertLess(q10["beach"], q5["beach"])

    def test_expansion_finds_new_candidates_and_is_not_weighted_a_second_time(self):
        index = PositionIndex([["love", "dance"], ["dance"], ["night"]], [2, 8, 19])
        q0, _ = index.build_vec(["love"], use_idf=True)
        self.assertEqual([doc_id for doc_id, _ in rank_vector(index, q0)], [2])
        q2, used = pseudo_relevance_feedback(index, q0, k=1)
        self.assertEqual(used, [2])
        self.assertIn("dance", q2)
        ranked = rank_vector(index, q2)
        self.assertEqual({doc_id for doc_id, _ in ranked}, {2, 8})
        query_norm = math.sqrt(sum(weight ** 2 for weight in q2.values()))
        self.assertAlmostEqual(dict(ranked)[2], (q2["love"] + q2["dance"]) / (query_norm * math.sqrt(2)))
        self.assertAlmostEqual(dict(ranked)[8], q2["dance"] / query_norm)

    def test_no_hits_zero_vectors_and_zero_k_have_no_feedback_documents(self):
        for query in ({}, {"love": 0.0}, {"unknown": 2.0}):
            with self.subTest(query=query):
                q2, used = pseudo_relevance_feedback(self.index, query, alpha=2)
                self.assertEqual(used, [])
                self.assertEqual(q2, {term: 2 * weight for term, weight in query.items() if weight})
        self.assertEqual(pseudo_relevance_feedback(self.index, self.q0, k=0, alpha=2), ({"love": 2.0}, []))
        self.assertEqual(pseudo_relevance_feedback(PositionIndex([]), self.q0), (self.q0, []))

    def test_zero_idf_terms_do_not_create_arbitrary_pseudo_relevant_documents(self):
        index = PositionIndex([["love"], ["love"]])
        q0, _ = index.build_vec(["love"], use_idf=True)
        self.assertEqual(pseudo_relevance_feedback(index, q0), ({}, []))

    def test_coefficients_scale_the_original_query_and_feedback(self):
        q2, used = pseudo_relevance_feedback(self.index, self.q0, k=1, alpha=2, beta=0.5)
        self.assertEqual(q2, {"love": 2.5})
        self.assertEqual(used, [2])
        self.assertEqual(pseudo_relevance_feedback(self.index, self.q0, beta=0)[0], self.q0)
        self.assertEqual(pseudo_relevance_feedback(self.index, self.q0, k=1, alpha=0)[0], {"love": 0.75})

    def test_rocchio_supports_negative_examples_clips_negative_weights_and_deduplicates_ids(self):
        query = {"love": 0.2, "night": 0.1}
        actual = rocchio(self.index, query, [2, 2], [19, 19], beta=0.5, gamma=0.75)
        self.assertEqual(actual, {"love": 0.7})
        self.assertEqual(query, {"love": 0.2, "night": 0.1})
        self.assertEqual(rocchio(self.index, query, [2, 2, 8], []),
                         rocchio(self.index, query, [2, 8], []))

    def test_empty_feedback_documents_contribute_zero_without_division_by_zero(self):
        index = PositionIndex([["love"], [], ["night"]], [2, 8, 19])
        self.assertEqual(rocchio(index, self.q0, [8], []), self.q0)
        self.assertEqual(rocchio(index, self.q0, [2, 8], []), {"love": 1.375})

    def test_only_postings_and_selected_vectors_are_read_using_saved_statistics(self):
        index = pickle.loads(pickle.dumps(self.index))
        index.document_weights.pop(19)
        index.document_norms.pop(19)
        index.document_query_norms.pop(19)
        index.document_ids = None
        with patch.object(index, "build_vec", side_effect=AssertionError("Query vector reweighted")), \
                patch.object(index, "idf", side_effect=AssertionError("IDF recalculated")), \
                patch("math.log10", side_effect=AssertionError("TF/IDF recalculated")):
            q2, used = pseudo_relevance_feedback(index, self.q0)
            self.assertEqual(used, [2, 8])
            self.assertEqual({doc_id for doc_id, _ in rank_vector(index, q2)}, {2, 8})

    def test_invalid_parameters_are_rejected(self):
        for k in (-1, 1.5, True, "5"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                pseudo_relevance_feedback(self.index, self.q0, k=k)
        for value in (-1, math.inf, math.nan, True, "1"):
            with self.subTest(value=value):
                for name in ("alpha", "beta"):
                    with self.assertRaises(ValueError):
                        pseudo_relevance_feedback(self.index, self.q0, **{name: value})
                with self.assertRaises(ValueError):
                    rocchio(self.index, self.q0, [], [], gamma=value)
                with self.assertRaises(ValueError):
                    rank_vector(self.index, {"love": value})


if __name__ == "__main__":
    unittest.main()
