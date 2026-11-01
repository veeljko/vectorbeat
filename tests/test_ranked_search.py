import math
import pickle
import unittest
from collections import Counter
from random import Random
from unittest.mock import patch

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.lyrics_search import make_snippets, search_lyrics, search_similar_lyrics
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class RankedSearchTests(unittest.TestCase):
    def test_formulas_and_full_document_norm(self):
        index = PositionIndex([["love", "love", "night"], ["night"], []], [2, 8, 11])
        a, b, tf = math.log10(3), math.log10(1.5), 1 + math.log10(2)
        self.assertAlmostEqual(index.idf("love"), a)
        self.assertAlmostEqual(index.document_norms[2], math.sqrt(tf * tf + 1))
        self.assertEqual(index.document_norms[11], 0)
        self.assertAlmostEqual(index.document_query_norms[2], math.sqrt((tf * a) ** 2 + b * b))
        self.assertAlmostEqual(index.score_tf_idf(["love", "night"])[2], tf * a + b)
        expected = (tf * a + b) / (math.sqrt(a * a + b * b) * math.sqrt(tf * tf + 1))
        self.assertAlmostEqual(index.score_ranked(["love", "night"], "cosine")[2], expected)
        self.assertAlmostEqual(index.cosine_lnc_ltc(["love", "night"], 2), expected)
        # I rec koja nije u upitu ulazi u normu dokumenta.
        self.assertAlmostEqual(index.cosine_lnc_ltc(["love"], 2), tf / math.sqrt(tf * tf + 1))

    def test_repeated_query_terms(self):
        index = PositionIndex([["love", "night"], ["love"], ["night"], []])
        single = index.score_tf_idf(["love"])[0]
        self.assertAlmostEqual(index.score_tf_idf(["love", "love"])[0], 2 * single)
        a, b = (1 + math.log10(2)) * index.idf("love"), index.idf("night")
        expected = (a + b) / (math.sqrt(a * a + b * b) * math.sqrt(2))
        self.assertAlmostEqual(index.score_ranked(["love", "love", "night"], "cosine")[0], expected)

    def test_only_postings_candidates_and_no_recomputed_statistics(self):
        index = PositionIndex([["love", "night"], ["alone"], []], [3, 9, 17])
        # Ako rangiranje pokusa da oceni nepovezanu pesmu ili obidje sve pesme,
        # nedostajace joj sacuvane tezine. Broj logaritama zavisi samo od upita.
        index.document_weights = {3: index.document_weights[3]}
        index.document_norms = {3: index.document_norms[3]}
        with patch.object(index, "df", side_effect=AssertionError("Ponovni IDF")), \
                patch.object(index, "tf", side_effect=AssertionError("Ponovni TF pesme")):
            with patch("math.log10", side_effect=AssertionError("Ponovna gradnja")):
                self.assertEqual(set(index.score_tf_idf(["love"])), {3})
            with patch("math.log10", wraps=math.log10) as logarithm:
                self.assertEqual(set(index.score_ranked(["love", "love"], "cosine")), {3})
                self.assertEqual(logarithm.call_count, 1)

    def test_empty_unknown_and_zero_idf(self):
        for index in (PositionIndex([]), PositionIndex([[], ["love"]])):
            for measure in ("tfidf", "cosine"):
                for query in ([], ["unknown"]):
                    self.assertEqual(index.score_ranked(query, measure), {})
        index = PositionIndex([["love"], ["love"]])
        self.assertEqual(index.score_ranked(["love"], "cosine"), {0: 0.0, 1: 0.0})
        self.assertEqual(index.cosine_lnc_ltc([], 0), 0.0)
        with self.assertRaises(ValueError):
            index.score_ranked(["love"], "wrong")

    def test_sorted_best_family_version_with_snippets(self):
        lyrics = ["Love night.", "Love love love.", "Night alone.", "Love forever."]
        index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics], [2, 8, 11, 19])
        families = {2: "artist::song", 8: "artist::song", 11: "other", 19: "third"}
        for measure in ("tfidf", "cosine"):
            result = search_lyrics(index, "LOVE", "ranked", measure=measure, families=families)
            self.assertEqual(list(result.hits), [8, 19])
            self.assertGreater(result.scores[8], result.scores[19])
            self.assertEqual(result.hits[8], [(0, 1), (1, 2), (2, 3)])
            self.assertIn("<mark>Love</mark>", make_snippets(lyrics[1], result.hits[8])[0])
        for query in ("", "the and", "zzzzzz"):
            self.assertEqual(search_lyrics(index, query, "ranked").hits, {})

    def test_equal_scores_use_catalog_id(self):
        index = PositionIndex([["love"], ["love"], []], [8, 2, 12])
        result = search_lyrics(index, "love", "ranked", families={8: "same", 2: "same"})
        self.assertEqual(list(result.hits), [2])

    def test_similar_songs_only_share_words_and_exclude_source_family(self):
        index = PositionIndex([["love", "night"], ["love"], ["alone"], ["night"], []],
                              [2, 8, 11, 19, 23])
        families = {2: "same", 8: "same", 11: "other", 19: "third", 23: "empty"}
        for measure in ("tfidf", "cosine"):
            self.assertEqual(set(index.score_similar(2, measure)), {8, 19})
            result = search_similar_lyrics(index, 2, measure, families)
            self.assertEqual(list(result.hits), [19])
            self.assertEqual(index.score_similar(23, measure), {})

    def test_similar_uses_cached_source_weights_and_norms(self):
        index = PositionIndex([["love", "love", "night"], ["love"], ["alone"], []])
        expected = index.cosine_lnc_ltc(["love", "love", "night"], 1)
        index.document_weights.pop(2)
        index.document_weights.pop(3)
        with patch("math.log10", side_effect=AssertionError("Ponovne tezine")), \
                patch("math.sqrt", side_effect=AssertionError("Ponovna norma")):
            scores = index.score_similar(0)
            self.assertEqual(set(scores), {1})
            self.assertAlmostEqual(scores[1], expected)

    def test_artifact_roundtrip_uses_saved_statistics(self):
        index = PositionIndex([["love", "love"], ["night"], []], [2, 8, 19])
        loaded = pickle.loads(pickle.dumps(index))
        self.assertEqual(loaded.idf_values, index.idf_values)
        self.assertEqual(loaded.document_norms, index.document_norms)
        self.assertEqual(loaded.document_query_norms, index.document_query_norms)
        with patch("math.log10", side_effect=AssertionError("IDF se racuna pri ucitavanju/upitu")):
            self.assertAlmostEqual(loaded.score_tf_idf(["love"])[2],
                                   (1 + math.log(2, 10)) * math.log(3, 10))

    def test_scores_against_independent_naive_reference(self):
        random = Random(37)
        documents = [[random.choice("abcd") for _ in range(random.randrange(12))]
                     for _ in range(20)]
        index = PositionIndex(documents)
        for _ in range(30):
            query = [random.choice("abcde") for _ in range(random.randrange(6))]
            counts = Counter(query)
            idfs = {t: math.log10(len(documents) / df) if df else 0
                    for t in counts
                    for df in [sum(t in doc for doc in documents)]}
            q = {t: (1 + math.log10(n)) * idfs[t] for t, n in counts.items()}
            nq = math.sqrt(sum(w * w for w in q.values()))
            tfidf, cosine = {}, {}
            for doc_id, document in enumerate(documents):
                if not set(query).intersection(document):
                    continue
                weights = {t: 1 + math.log10(n) for t, n in Counter(document).items()}
                nd = math.sqrt(sum(w * w for w in weights.values()))
                tfidf[doc_id] = sum(weights.get(t, 0) * idfs[t] * n for t, n in counts.items())
                dot = sum(w * weights.get(t, 0) for t, w in q.items())
                cosine[doc_id] = dot / (nq * nd) if nq * nd else 0.0
            for measure, expected in (("tfidf", tfidf), ("cosine", cosine)):
                actual = index.score_ranked(query, measure)
                self.assertEqual(set(actual), set(expected))
                for doc_id in expected:
                    self.assertAlmostEqual(actual[doc_id], expected[doc_id])


if __name__ == "__main__":
    unittest.main()
