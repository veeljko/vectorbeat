"""PageRank with dangling nodes and optional uniform teleportation to a subset."""

from collections.abc import Hashable, Iterable, Mapping
import math
import warnings


def _prepare_graph(graph: Mapping[Hashable, Iterable[Hashable]], d: float,
                   landing_nodes: Iterable[Hashable] | None):
    if (isinstance(d, bool) or not isinstance(d, (int, float))
            or not math.isfinite(d) or not 0 <= d <= 1):
        raise ValueError("d must be a finite number between 0 and 1.")

    adjacency = {node: tuple(targets) for node, targets in graph.items()}
    # A destination without its own adjacency entry is a dangling node.
    for targets in tuple(adjacency.values()):
        for target in targets:
            adjacency.setdefault(target, ())
    destinations = tuple(dict.fromkeys(adjacency if landing_nodes is None else landing_nodes))
    if adjacency and not destinations:
        raise ValueError("landing_nodes must not be empty for a non-empty graph.")
    if any(node not in adjacency for node in destinations):
        raise ValueError("All landing_nodes must belong to the graph.")
    return adjacency, destinations


def _step(graph: dict, rank: Mapping[Hashable, float], d: float,
          landing_nodes: tuple) -> dict[Hashable, float]:
    new_rank = {node: 0.0 for node in graph}
    jump_mass = 0.0
    for source, targets in graph.items():
        if not targets:
            jump_mass += rank[source]
        else:
            jump_mass += (1 - d) * rank[source]
            contribution = d * rank[source] / len(targets)
            for target in targets:
                new_rank[target] += contribution
    jump_share = jump_mass / len(landing_nodes)
    for node in landing_nodes:
        new_rank[node] += jump_share
    return new_rank


def one_step(graph: Mapping[Hashable, Iterable[Hashable]], rank: Mapping[Hashable, float],
             d: float = 0.85, landing_nodes: Iterable[Hashable] | None = None
             ) -> dict[Hashable, float]:
    """Distribute rank along links and redistribute all dangling-node mass.

    The remaining mass jumps uniformly to landing_nodes (all nodes by default).
    rank must contain one finite, nonnegative value for every node, including
    nodes appearing only as destinations. The inputs are not modified.
    """
    adjacency, destinations = _prepare_graph(graph, d, landing_nodes)
    if set(rank) != set(adjacency):
        raise ValueError("rank must contain exactly the graph's nodes.")
    if any(not math.isfinite(value) or value < 0 for value in rank.values()):
        raise ValueError("Rank values must be finite and nonnegative.")
    if not adjacency:
        return {}
    return _step(adjacency, rank, d, destinations)


def pagerank(graph: Mapping[Hashable, Iterable[Hashable]], d: float = 0.85,
             landing_nodes: Iterable[Hashable] | None = None, eps: float = 1e-8,
             max_iterations: int = 1000) -> tuple[dict[Hashable, float], list[float]]:
    """Return (ranks, changes), starting from a uniform rank distribution.

    changes records the sum of absolute rank differences at each iteration.
    Stop when that sum is below eps. At the iteration limit, issue a warning
    and return the last estimate. An empty graph returns ({}, []).
    Nodes appearing only as destinations are included as dangling nodes.
    """
    if (isinstance(eps, bool) or not isinstance(eps, (int, float))
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError("eps must be a finite positive number.")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations <= 0):
        raise ValueError("max_iterations must be a positive integer.")
    adjacency, destinations = _prepare_graph(graph, d, landing_nodes)
    if not adjacency:
        return {}, []

    rank = {node: 1.0 / len(adjacency) for node in adjacency}
    changes = []
    for _ in range(max_iterations):
        new_rank = _step(adjacency, rank, d, destinations)
        change = math.fsum(abs(new_rank[node] - rank[node]) for node in adjacency)
        changes.append(change)
        rank = new_rank
        if change < eps:
            return rank, changes
    warnings.warn(f"PageRank did not converge after {max_iterations} iterations.",
                  RuntimeWarning, stacklevel=2)
    return rank, changes
