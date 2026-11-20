import math
import unittest
from unittest.mock import patch

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.ranking.rocchio.playlist import build_feedback_playlist
from mrvectorbeat.search.ranking.rocchio.rocchio import rocchio


class FeedbackPlaylistTests(unittest.TestCase):
    def setUp(self):
        self.index = PositionIndex([["love", "guitar"], ["love", "beach"],
                                    ["guitar"], ["beach"], ["other"]])

    def test_sparse_vector_matches_rocchio_formula_clipping_and_normalization(self):
        original_vectors = {doc: weights.copy() for doc, weights in self.index.document_weights.items()}
        result = build_feedback_playlist(self.index, "love", [0], [1],
                                         displayed_ids=[0, 1], beta=0.25)
        raw = rocchio(self.index, {"love": 1.0}, [0], [1], beta=0.25)
        norm = math.sqrt(sum(weight * weight for weight in raw.values()))
        self.assertEqual(set(result.weights), set(raw))
        for term, weight in raw.items():
            self.assertAlmostEqual(result.weights[term], weight / norm)
        self.assertAlmostEqual(sum(weight * weight for weight in result.weights.values()), 1.0)
        self.assertEqual([doc for doc, _ in result.songs], [2])
        self.assertEqual(result.terms[0].term, "guitar")
        self.assertEqual(result.terms[0].source_ids, (0,))
        self.assertAlmostEqual(result.terms[0].contribution, result.songs[0][1])
        self.assertIn("love", result.avoided_terms)
        self.assertEqual(self.index.document_weights, original_vectors)

    def test_switching_or_removing_judgments_changes_songs_and_feedback_words(self):
        rock = build_feedback_playlist(self.index, "love", [0], [1], displayed_ids=[0, 1])
        beach = build_feedback_playlist(self.index, "love", [1], [0], displayed_ids=[0, 1])
        self.assertEqual([doc for doc, _ in rock.songs], [2])
        self.assertEqual([doc for doc, _ in beach.songs], [3])
        self.assertEqual([item.term for item in rock.terms], ["guitar"])
        self.assertEqual([item.term for item in beach.terms], ["beach"])
        removed = build_feedback_playlist(self.index, "love", displayed_ids=[0, 1],
                                          initial_results=[0, 1], pseudo_k=1)
        self.assertTrue(removed.assumed)
        self.assertEqual(removed.relevant, [0])
        self.assertEqual([doc for doc, _ in removed.songs], [doc for doc, _ in rock.songs])

    def test_negative_judgments_reduce_overlapping_features_and_do_not_create_positive_examples(self):
        positive = build_feedback_playlist(self.index, "love", [0], displayed_ids=[0, 1])
        negative = build_feedback_playlist(self.index, "love", [0], [2], displayed_ids=[0, 1, 2], gamma=1)
        self.assertIn("guitar", positive.weights)
        self.assertNotIn("guitar", negative.weights)
        self.assertIn("guitar", negative.avoided_terms)
        negative_only = build_feedback_playlist(self.index, "love", [], [1],
                                                displayed_ids=[0, 1], initial_results=[0, 1])
        self.assertFalse(negative_only.assumed)
        self.assertEqual(negative_only.relevant, [])
        self.assertEqual(negative_only.non_relevant, [1])

    def test_expansion_is_strictly_weaker_than_every_surviving_typed_term(self):
        index = PositionIndex([["love"] + ["guitar"] * 10, ["love", "violin"]]
                              + [["love"] for _ in range(7)] + [["other"]])
        result = build_feedback_playlist(index, "love violin", [0], displayed_ids=[0])
        typed = [result.weights[term] for term in ("love", "violin")]
        self.assertIn("guitar", result.weights)
        self.assertLess(result.weights["guitar"], min(typed))
        self.assertAlmostEqual(result.weights["guitar"], 0.5 * min(typed))

    def test_filters_displayed_families_and_playlist_duplicates_before_top_ten(self):
        index = PositionIndex([["love", "guitar"], ["love", "guitar"]]
                              + [["guitar"] for _ in range(24)] + [["other"]])
        families = {doc: f"song_{doc}" for doc in range(27)}
        families[1] = families[0]
        families[15] = families[14]
        shown = [0] + list(range(2, 14))
        result = build_feedback_playlist(index, "love", [0], displayed_ids=shown, families=families)
        self.assertEqual(len(result.songs), 10)
        ids = [doc for doc, _ in result.songs]
        self.assertNotIn(1, ids)  # Unseen remix of a displayed song.
        self.assertFalse(set(ids).intersection(shown))
        self.assertEqual(len({families[doc] for doc in ids}), 10)
        self.assertFalse({14, 15}.issubset(ids))
        self.assertEqual([score for _, score in result.songs],
                         sorted((score for _, score in result.songs), reverse=True))

    def test_assumed_feedback_uses_rank_order_only_displayed_and_one_per_family(self):
        result = build_feedback_playlist(self.index, "love", displayed_ids=[0, 1],
                                         initial_results=[3, 1, 0, 2], pseudo_k=5,
                                         families={0: "same", 1: "same"})
        self.assertTrue(result.assumed)
        self.assertEqual(result.relevant, [1])
        self.assertEqual([doc for doc, _ in result.songs], [3])

    def test_no_query_vector_can_still_use_liked_lyrics_and_empty_vectors_are_safe(self):
        result = build_feedback_playlist(self.index, "zzzunknown", [0], displayed_ids=[0])
        self.assertTrue(result.weights)
        self.assertTrue(result.songs)
        self.assertTrue(all(item.source_ids == (0,) for item in result.terms))
        self.assertEqual(build_feedback_playlist(PositionIndex([[], []]), "love", [0]).songs, [])
        self.assertEqual(build_feedback_playlist(PositionIndex([]), "").weights, {})
        self.assertEqual(build_feedback_playlist(self.index, "zzzunknown").songs, [])

    def test_feedback_term_limit_and_zero_requested_songs(self):
        result = build_feedback_playlist(self.index, "love", [0], displayed_ids=[0], max_new_terms=0)
        self.assertNotIn("guitar", result.weights)
        self.assertEqual(build_feedback_playlist(self.index, "love", [0], limit=0).songs, [])

    def test_only_sparse_selected_vectors_and_postings_are_read(self):
        class LookupOnly(dict):
            def __iter__(self):
                raise AssertionError("Full-catalog traversal")

            def items(self):
                raise AssertionError("Full-catalog traversal")

            def values(self):
                raise AssertionError("Full-catalog traversal")

        for name in ("document_weights", "document_norms", "document_query_norms", "doc_len", "index"):
            setattr(self.index, name, LookupOnly(getattr(self.index, name)))
        self.index.document_ids = None
        with patch.object(self.index, "df", side_effect=AssertionError("IDF recomputed")):
            result = build_feedback_playlist(self.index, "love", [0], [1], displayed_ids=[0, 1])
        self.assertEqual([doc for doc, _ in result.songs], [2])

    def test_invalid_parameters_and_conflicting_or_missing_feedback_are_rejected(self):
        for kwargs in ({"limit": -1}, {"pseudo_k": True}, {"max_new_terms": 0.5},
                       {"alpha": -1}, {"beta": math.nan}, {"gamma": math.inf},
                       {"expansion_ratio": 1}, {"expansion_ratio": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_feedback_playlist(self.index, "love", **kwargs)
        with self.assertRaises(ValueError):
            build_feedback_playlist(self.index, "love", [0], [0])
        with self.assertRaises(ValueError):
            build_feedback_playlist(self.index, "love", [100])


if __name__ == "__main__":
    unittest.main()
