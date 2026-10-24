import unittest
from random import Random
from unittest.mock import patch

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.lyrics_search import search_lyrics, make_snippets
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class BooleanTests(unittest.TestCase):
    def test_catalog_ids_and_sorted_postings(self):
        index = PositionIndex([["love"], ["night", "love", "love"], []], [9, 3, 6])
        self.assertEqual(index.index["love"], [(3, [1, 2]), (9, [0])])
        self.assertEqual(index.document_ids, [3, 6, 9])
        with self.assertRaises(ValueError):
            PositionIndex([[], []], [1, 1])
        with self.assertRaises(ValueError):
            PositionIndex([[]], [])

    def test_empty_single_and_missing_terms(self):
        index = PositionIndex([["a"], ["b"]])
        for operator in ("AND", "OR", "DIFFERENCE"):
            self.assertEqual(index.find_boolean([], operator), [])
            self.assertEqual(index.find_boolean(["a"], operator), [0])
            self.assertEqual(index.find_boolean(["missing"], operator), [])
        self.assertEqual(index.find_boolean(["missing", "a"], "OR"), [0])
        self.assertEqual(index.find_boolean(["a", "a"], "DIFFERENCE"), [])
        with self.assertRaises(ValueError):
            index.find_boolean(["a"], "wrong")

    def test_shortest_first_and_early_exit(self):
        index = PositionIndex([["a", "c"], ["a"], ["b"], ["c"]])
        with patch.object(index, "intersection", wraps=index.intersection) as intersect:
            self.assertEqual(index.find_boolean(["a", "b", "c"]), [])
            self.assertEqual(intersect.call_count, 1)
            self.assertEqual(intersect.call_args.args, ([2], [0, 1]))
        with patch.object(index, "intersection", wraps=index.intersection) as intersect:
            self.assertEqual(index.find_boolean(["a", "missing", "c"]), [])
            intersect.assert_not_called()

    def test_boolean_against_naive_scan(self):
        random = Random(21)
        documents = [[random.choice("abc") for _ in range(random.randrange(10))]
                     for _ in range(40)]
        index = PositionIndex(documents)
        for _ in range(100):
            words = [random.choice("abcd") for _ in range(random.randrange(1, 8))]
            expected = {
                "AND": [i for i, doc in enumerate(documents) if all(w in doc for w in words)],
                "OR": [i for i, doc in enumerate(documents) if any(w in doc for w in words)],
                "DIFFERENCE": [i for i, doc in enumerate(documents)
                               if words[0] in doc and all(w not in doc for w in words[1:])],
            }
            for operator, result in expected.items():
                self.assertEqual(index.find_boolean(words, operator), result)


class LyricsSearchTests(unittest.TestCase):
    def setUp(self):
        self.lyrics = ["Love the night, love!", "Night brings love.", "Running alone."]
        self.index = PositionIndex([TextProcessor(text).text_tokenize() for text in self.lyrics], [2, 7, 9])

    def test_modes_and_positions(self):
        self.assertEqual(list(search_lyrics(self.index, "LOVE night").hits), [2, 7])
        self.assertEqual(list(search_lyrics(self.index, "love running", operator="OR").hits), [2, 7, 9])
        self.assertEqual(list(search_lyrics(self.index, "love night", operator="DIFFERENCE").hits), [])
        self.assertEqual(search_lyrics(self.index, "love night", "phrase").hits, {2: [(0, 2)]})
        self.assertEqual(search_lyrics(self.index, "love night", "proximity", distance=2).hits,
                         {2: [(0, 2), (1, 3)]})
        self.assertIn(7, search_lyrics(self.index, "love night", "proximity", distance=3).hits)

    def test_empty_and_invalid_queries(self):
        for mode in ("boolean", "phrase", "proximity"):
            for query in ("", "the and", "!!!"):
                self.assertEqual(search_lyrics(self.index, query, mode).hits, {})
        self.assertEqual(search_lyrics(self.index, "absent", "phrase").hits, {})
        for query in ("love", "love night running"):
            with self.assertRaises(ValueError):
                search_lyrics(self.index, query, "proximity")

    def test_spans_match_default_processing(self):
        for text in ["DON’T don't dont I'm nothin’", "rock-n-roll love/hate a.m. hello,world",
                     "Love💙night 24k 123", "«Running»\n  in the NIGHT", "", "the and", "ΟΣ"]:
            with self.subTest(text=text):
                processor = TextProcessor(text)
                tokens, spans = processor.tokenize_with_spans()
                self.assertEqual(tokens, processor.text_tokenize())
                self.assertEqual(len(tokens), len(spans))
                for start, end in spans:
                    self.assertTrue(0 <= start < end <= len(text))

    def test_original_phrase_and_html_escaping(self):
        hits = search_lyrics(self.index, "love night", "phrase").hits[2]
        self.assertIn("<mark>Love the night</mark>", make_snippets(self.lyrics[0], hits)[0])
        text = "<script>alert('x')</script> Love & night"
        tokens = TextProcessor(text).text_tokenize()
        position = tokens.index("love")
        snippet = make_snippets(text, [(position, position + 1)], context=20)[0]
        self.assertIn("&lt;script&gt;", snippet)
        self.assertNotIn("<script>", snippet)
        self.assertIn("<mark>Love</mark>", snippet)

    def test_snippet_limits_and_invalid_spans(self):
        text = " ".join(["love"] + ["night"] * 30 + ["love"] + ["night"] * 30 + ["love"])
        snippets = make_snippets(text, [(0, 1), (31, 32), (62, 63)], context=2, limit=2)
        self.assertEqual(len(snippets), 2)
        self.assertTrue(all("<mark>love</mark>" in s for s in snippets))
        self.assertTrue(all(len(s) < 100 for s in snippets))
        self.assertEqual(make_snippets(text, []), [])
        with self.assertRaises(ValueError):
            make_snippets(text, [(100, 101)])


if __name__ == "__main__":
    unittest.main()
