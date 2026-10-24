import unittest
from unittest.mock import patch

from mrvectorbeat.kgram.kgram import KGram
from mrvectorbeat.trie.name_search import NameSearch


class KGramTests(unittest.TestCase):
    def test_boundary_grams_and_case(self):
        index = KGram(["Beat", "BEAT"])
        self.assertEqual(set(index.calc_kgram("BEAT")), {"$be", "bea", "eat", "at$"})
        self.assertEqual(index.wildcard_pretraga("BE*T"), ["Beat"])

    def test_candidates_are_verified(self):
        index = KGram(["beat", "beach", "abc", "abbc", "singing", "song"])
        self.assertEqual(index.search_kgrams("be*t"), {"beat", "beach"})
        self.assertEqual(index.wildcard_pretraga("be*t"), ["beat"])
        self.assertEqual(index.wildcard_pretraga("ab*bc"), ["abbc"])
        self.assertEqual(index.wildcard_pretraga("s*ing*"), ["singing"])

    def test_no_complete_grams(self):
        index = KGram(["bat", "boat", "cat"])
        self.assertIsNone(index.search_kgrams("b*a*t"))
        self.assertEqual(index.wildcard_pretraga("b*a*t"), ["bat", "boat"])
        self.assertEqual(index.wildcard_pretraga("***"), ["bat", "boat", "cat"])

    def test_literal_special_characters(self):
        index = KGram(["Song [Live]", "Song Live", "Who?", "Who!"])
        self.assertEqual(index.wildcard_pretraga("*[Live]"), ["Song [Live]"])
        self.assertEqual(index.wildcard_pretraga("Who?"), ["Who?"])

    def test_shortest_posting_first_and_no_mutation(self):
        copied = []
        class Posting(set):
            def copy(self):
                copied.append(len(self))
                return set(self)
        index = KGram(["beat", "beach", "beast"])
        index.index = {gram: Posting(words) for gram, words in index.index.items()}
        before = {gram: set(words) for gram, words in index.index.items()}
        index.search_kgrams("beat")
        self.assertEqual(copied, [1])
        self.assertEqual(index.index, before)
        self.assertEqual(index.search_kgrams("zzzz"), set())
        self.assertEqual(index.index, before)

    def test_jaccard_threshold(self):
        index = KGram(["Hello"])
        # hello: 5 grama; hellp: 5; presek 3, unija 7.
        word, score = index.correct("hellp")
        self.assertEqual(word, "Hello")
        self.assertAlmostEqual(score, 3 / 7)
        self.assertIsNone(index.correct("hellp", threshold=score))
        self.assertIsNone(index.correct("zzzz"))
        self.assertIsNone(index.correct(""))
        self.assertIsNone(index.correct("hel*"))

    def test_correction_candidates_only_from_index(self):
        index = KGram(["Hello"])
        index.index.clear()
        self.assertIsNone(index.correct("Hello", threshold=0))

    def test_routing_and_ranking(self):
        search = NameSearch([("Beat", "Pesma")] * 3 + [("Best", "Pesma"), ("Hello", "Izvodjac")])
        with patch.object(search.kgram, "correct", side_effect=AssertionError("Unexpected correction")), patch.object(search.kgram, "wildcard_pretraga", side_effect=AssertionError("Unexpected wildcard lookup")):
            self.assertEqual(search.suggest("BE"), ["Beat", "Best"])
            self.assertEqual(search.suggest("be*"), ["Beat", "Best"])
            self.assertEqual(search.suggest("*at"), ["Beat"])
            self.assertEqual(search.suggest("zz*"), [])
        with patch.object(search.kgram, "correct", side_effect=AssertionError("Correction on wildcard")):
            self.assertEqual(search.suggest("b*t", 1), ["Beat"])
            self.assertEqual(search.suggest("*ea*"), ["Beat"])
        with patch.object(search.kgram, "wildcard_pretraga", side_effect=AssertionError("Wildcard on typo")):
            self.assertEqual(search.suggest("hellp"), ["Hello"])

    def test_empty_index(self):
        index = KGram([])
        self.assertEqual(index.wildcard_pretraga("*"), [])
        self.assertIsNone(index.correct("hello"))


if __name__ == "__main__":
    unittest.main()
