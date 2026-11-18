import unittest
from copy import deepcopy

from mrvectorbeat.search.ranking import one_step, pagerank


class PageRankTests(unittest.TestCase):
    def test_one_step_redistributes_dangling_mass_and_conserves_rank(self):
        graph = {"A": ["B"], "B": []}
        rank = {"A": 0.5, "B": 0.5}
        original_graph, original_rank = deepcopy(graph), rank.copy()
        result = one_step(graph, rank)
        self.assertAlmostEqual(result["A"], 0.2875)
        self.assertAlmostEqual(result["B"], 0.7125)
        self.assertAlmostEqual(sum(result.values()), 1)
        self.assertEqual(graph, original_graph)
        self.assertEqual(rank, original_rank)

    def test_cycle_converges_to_uniform_rank(self):
        graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
        ranks, changes = pagerank(graph)
        for value in ranks.values():
            self.assertAlmostEqual(value, 1 / 3)
        self.assertEqual(changes, [0.0])

    def test_dangling_graph_matches_analytical_solution(self):
        ranks, changes = pagerank({"A": ["B"], "B": []}, eps=1e-12)
        self.assertAlmostEqual(ranks["A"], 1 / 2.85, places=11)
        self.assertAlmostEqual(ranks["B"], 1.85 / 2.85, places=11)
        self.assertLess(changes[-1], 1e-12)
        self.assertAlmostEqual(sum(ranks.values()), 1)

    def test_personalized_landing_nodes_and_duplicate_destinations(self):
        graph = {"A": ["B"], "B": []}
        ranks, changes = pagerank(graph, landing_nodes=["A", "A"], eps=1e-12)
        self.assertAlmostEqual(ranks["A"], 1 / 1.85, places=11)
        self.assertAlmostEqual(ranks["B"], 0.85 / 1.85, places=11)
        self.assertAlmostEqual(sum(ranks.values()), 1)
        self.assertLess(changes[-1], 1e-12)

    def test_destination_only_nodes_are_included_without_mutating_graph(self):
        graph = {"A": ["B"]}
        ranks, _ = pagerank(graph, eps=1e-12)
        self.assertEqual(set(ranks), {"A", "B"})
        self.assertAlmostEqual(ranks["B"], 1.85 / 2.85, places=11)
        self.assertEqual(graph, {"A": ["B"]})
        streamed, _ = pagerank({"A": iter(["B"])}, eps=1e-12)
        self.assertEqual(streamed, ranks)

    def test_empty_graph_dangling_nodes_and_zero_damping(self):
        self.assertEqual(pagerank({}), ({}, []))
        self.assertEqual(one_step({}, {}), {})
        self.assertEqual(pagerank({"A": []}), ({"A": 1.0}, [0.0]))
        ranks, _ = pagerank({"A": [], "B": []})
        self.assertEqual(ranks, {"A": 0.5, "B": 0.5})
        ranks, _ = pagerank({"A": ["B"], "B": ["A"]}, d=0, landing_nodes=["B"])
        self.assertEqual(ranks, {"A": 0.0, "B": 1.0})

    def test_iteration_limit_returns_last_estimate_and_change_history(self):
        graph = {"A": ["B"], "B": ["B"]}
        with self.assertWarnsRegex(RuntimeWarning, "did not converge after 1 iterations"):
            ranks, changes = pagerank(graph, max_iterations=1)
        self.assertAlmostEqual(ranks["A"], 0.075)
        self.assertAlmostEqual(ranks["B"], 0.925)
        self.assertAlmostEqual(changes[0], 0.85)
        self.assertEqual(len(changes), 1)

    def test_invalid_parameters_and_rank_values(self):
        graph = {"A": ["B"], "B": []}
        for parameters in ({"d": -0.1}, {"d": 1.1}, {"d": float("nan")},
                           {"eps": 0}, {"eps": float("inf")},
                           {"max_iterations": 0}, {"max_iterations": True},
                           {"landing_nodes": []}, {"landing_nodes": ["missing"]}):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    pagerank(graph, **parameters)
        for ranks in ({"A": 1.0}, {"A": -0.5, "B": 1.5}, {"A": float("nan"), "B": 1}):
            with self.subTest(ranks=ranks):
                with self.assertRaises(ValueError):
                    one_step(graph, ranks)


if __name__ == "__main__":
    unittest.main()
