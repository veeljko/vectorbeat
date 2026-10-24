import unittest
from itertools import product
from random import Random
from unittest.mock import patch

from mrvectorbeat.search.indexes.position_index import PositionIndex


class PhraseTests(unittest.TestCase):
    def test_document_positions(self):
        index = PositionIndex([['love', 'night', 'love'], [], ['night']])
        self.assertEqual(index.index['love'], [(0, [0, 2])])
        self.assertEqual(index.index['night'], [(0, [1]), (2, [0])])

    def test_order_and_adjacency(self):
        index = PositionIndex([
            ['love', 'night', 'love', 'night'],
            ['night', 'love'], ['love', 'long', 'night'],
        ])
        self.assertEqual(index.find_phrase(['love', 'night']), [(0, 0), (0, 2)])
        self.assertEqual(index.find_phrase(['night', 'love']), [(0, 1), (1, 0)])

    def test_repeated_words_and_overlap(self):
        index = PositionIndex([['a', 'a', 'a']])
        self.assertEqual(index.find_phrase(['a', 'a']), [(0, 0), (0, 1)])
        self.assertEqual(index.find_phrase(['a', 'a', 'a']), [(0, 0)])
        self.assertEqual(index.find_phrase(['a'] * 4), [])

    def test_empty_missing_and_single_term(self):
        index = PositionIndex([[], ['a', 'a']])
        self.assertEqual(index.find_phrase([]), [])
        self.assertEqual(index.find_phrase(['missing']), [])
        self.assertNotIn('missing', index.index)
        self.assertEqual(index.find_phrase(['a']), [(1, 0), (1, 1)])
        self.assertEqual(PositionIndex([]).find_phrase(['a']), [])

    def test_shortest_first_and_early_exit(self):
        index = PositionIndex([['a', 'c'], ['a'], ['b'], ['c']])
        with patch.object(index, 'intersection', wraps=index.intersection) as intersect:
            self.assertEqual(index.find_phrase(['a', 'b', 'c']), [])
            self.assertEqual(intersect.call_count, 1)
            self.assertEqual(intersect.call_args.args, ([2], [0, 1]))

    def test_matches_naive_phrase_scan(self):
        random = Random(17)
        documents = [[random.choice('abc') for _ in range(random.randrange(15))]
                     for _ in range(50)]
        index = PositionIndex(documents)
        for length in range(1, 5):
            for phrase in product('abc', repeat=length):
                words = list(phrase)
                expected = [(doc_id, pos) for doc_id, document in enumerate(documents)
                            for pos in range(len(document) - length + 1)
                            if document[pos:pos + length] == words]
                with self.subTest(words=words):
                    self.assertEqual(index.find_phrase(words), expected)


if __name__ == '__main__':
    unittest.main()
