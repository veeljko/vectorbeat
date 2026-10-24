import unittest
from random import Random

from mrvectorbeat.search.indexes.position_index import PositionIndex


class ProximityTests(unittest.TestCase):
    def test_order_and_strict_boundary(self):
        index = PositionIndex([['a', 'b'], ['b', 'a'], ['a', 'x', 'b']])
        self.assertEqual(index.find_proximity('a', 'b', 2), [(0, 0, 1), (1, 1, 0)])
        self.assertEqual(index.find_proximity('a', 'b', 3),
                         [(0, 0, 1), (1, 1, 0), (2, 0, 2)])

    def test_all_pairs_and_repeated_term(self):
        index = PositionIndex([['a', 'b', 'a', 'b'], ['a', 'a', 'a']])
        self.assertEqual(index.find_proximity('a', 'b', 2),
                         [(0, 0, 1), (0, 2, 1), (0, 2, 3)])
        self.assertEqual(index.find_proximity('a', 'a', 2), [(1, 0, 1), (1, 1, 2)])

    def test_empty_missing_and_invalid_distance(self):
        index = PositionIndex([['a'], ['b']])
        self.assertEqual(index.find_proximity('a', 'b', 5), [])
        self.assertEqual(index.find_proximity('', 'b', 5), [])
        self.assertEqual(index.find_proximity('a', 'missing', 5), [])
        self.assertNotIn('missing', index.index)
        self.assertEqual(PositionIndex([]).find_proximity('a', 'b', 5), [])
        for distance in [0, 1]:
            self.assertEqual(index.find_proximity('a', 'a', distance), [])
        for distance in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                index.find_proximity('a', 'b', distance)

    def test_matches_naive_scan(self):
        random = Random(19)
        documents = [[random.choice('abc') for _ in range(random.randrange(15))]
                     for _ in range(50)]
        index = PositionIndex(documents)
        for a in 'abc':
            for b in 'abc':
                for distance in range(7):
                    expected = [(d, px, py) for d, document in enumerate(documents)
                                for px, first in enumerate(document)
                                for py, second in enumerate(document)
                                if first == a and second == b
                                and abs(px - py) < distance
                                and (a != b or px < py)]
                    with self.subTest(a=a, b=b, distance=distance):
                        self.assertEqual(index.find_proximity(a, b, distance), expected)


if __name__ == '__main__':
    unittest.main()
