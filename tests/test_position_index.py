"""Testovi operacija nad sortiranim listama jedinstvenih ID-eva dokumenata.

Ne zahtevaju katalog niti artefakte. Ulazi nemaju duplikate unutar jedne
postings liste; isti ID moze da se pojavi u obe liste.
"""

import unittest
from itertools import combinations

from mrvectorbeat.search.indexes.position_index import PositionIndex


class PositionIndexOperationsTests(unittest.TestCase):
    def setUp(self):
        self.index = PositionIndex([])

    # leva, desna, presek, unija, razlika (leva minus desna)
    CASES = [
        ([], [], [], [], []),
        ([], [1, 3], [], [1, 3], []),
        ([1, 3], [], [], [1, 3], [1, 3]),
        ([2], [2], [2], [2], []),
        ([1], [2], [], [1, 2], [1]),
        ([2], [1], [], [1, 2], [2]),
        ([1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], []),
        ([1, 2], [4, 5], [], [1, 2, 4, 5], [1, 2]),
        ([4, 5], [1, 2], [], [1, 2, 4, 5], [4, 5]),
        ([1, 3, 5], [2, 4, 6], [], [1, 2, 3, 4, 5, 6], [1, 3, 5]),
        ([1, 2, 4], [0, 2, 3, 4, 7], [2, 4], [0, 1, 2, 3, 4, 7], [1]),
        ([2, 4], [1, 2, 3, 4, 5], [2, 4], [1, 2, 3, 4, 5], []),
        ([1, 2, 3, 4, 5], [2, 4], [2, 4], [1, 2, 3, 4, 5], [1, 3, 5]),
        ([1, 4, 9], [1, 6, 9], [1, 9], [1, 4, 6, 9], [4]),
        ([0, 100, 1000000], [100, 200], [100], [0, 100, 200, 1000000], [0, 1000000]),
    ]

    def test_intersection(self):
        for left, right, expected, _, _ in self.CASES:
            with self.subTest(left=left, right=right):
                self.assertEqual(self.index.intersection(left, right), expected)

    def test_union(self):
        for left, right, _, expected, _ in self.CASES:
            with self.subTest(left=left, right=right):
                self.assertEqual(self.index.union(left, right), expected)

    def test_difference(self):
        for left, right, _, _, expected in self.CASES:
            with self.subTest(left=left, right=right):
                self.assertEqual(self.index.difference(left, right), expected)

    def test_inputs_are_not_modified_or_returned_as_result(self):
        for operation in (self.index.intersection, self.index.union, self.index.difference):
            for left, right, *_ in self.CASES:
                with self.subTest(operation=operation.__name__, left=left, right=right):
                    first, second = left.copy(), right.copy()
                    result = operation(first, second)
                    self.assertEqual(first, left)
                    self.assertEqual(second, right)
                    self.assertIsNot(result, first)
                    self.assertIsNot(result, second)

    def test_difference_is_directional(self):
        self.assertEqual(self.index.difference([1, 2, 4], [2, 3, 4]), [1])
        self.assertEqual(self.index.difference([2, 3, 4], [1, 2, 4]), [3])

    def test_exhaustive_small_lists(self):
        # Svih 64 sortirane liste nad ID-evima 0..5, odnosno 4096 parova.
        lists = [list(values) for size in range(7)
                 for values in combinations(range(6), size)]
        for left in lists:
            for right in lists:
                with self.subTest(left=left, right=right):
                    # Nezavisna referenca, bez skupova i bez dva pokazivaca.
                    intersection = [value for value in left if value in right]
                    union = sorted(left + [value for value in right if value not in left])
                    difference = [value for value in left if value not in right]
                    self.assertEqual(self.index.intersection(left, right), intersection)
                    self.assertEqual(self.index.union(left, right), union)
                    self.assertEqual(self.index.difference(left, right), difference)


if __name__ == "__main__":
    unittest.main()
