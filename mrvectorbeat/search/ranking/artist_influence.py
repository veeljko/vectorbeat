"""Prepare catalog artist influence from PageRank on the complete page graph."""

from collections.abc import Hashable, Iterable, Mapping

import pandas as pd

from .page_ranking import pagerank


DEFAULT_DAMPING = 0.85


def build_artist_influence(graph: Mapping[Hashable, Iterable[Hashable]],
                           artist_nodes: Mapping[str, Hashable],
                           catalog_artists: Iterable[str], *, d: float = DEFAULT_DAMPING,
                           eps: float = 1e-8, max_iterations: int = 1000) -> dict:
    """Rank all pages first, then select catalog artists without renormalizing.

    Each source page counts once toward a target's incoming-page count. Genres,
    albums, labels and other non-artist pages participate in both calculations.
    Only catalog names mapped to nodes in the graph are included in the table.
    Full-graph ranks are temporary: the returned artifact contains only the
    catalog table and small preparation metadata, not per-page dictionaries.
    """
    adjacency = {source: tuple(dict.fromkeys(targets)) for source, targets in graph.items()}
    for targets in tuple(adjacency.values()):
        for target in targets:
            adjacency.setdefault(target, ())
    ranks, changes = pagerank(adjacency, d=d, eps=eps, max_iterations=max_iterations)
    catalog_names = set(catalog_artists)
    mapped_artists = {artist: node for artist, node in artist_nodes.items()
                      if artist in catalog_names and node in ranks}
    incoming = dict.fromkeys(mapped_artists.values(), 0)
    for targets in adjacency.values():
        for target in targets:
            if target in incoming:
                incoming[target] += 1

    rows = [{"artist_name": artist, "url": node, "pagerank": ranks[node],
             "incoming_links": incoming[node]}
            for artist, node in mapped_artists.items()]
    rows.sort(key=lambda row: (-row["pagerank"], row["artist_name"].casefold(), row["artist_name"]))
    table = pd.DataFrame(rows, columns=["artist_name", "url", "pagerank", "incoming_links"])
    table.insert(0, "position", range(1, len(table) + 1))
    return {
        "version": 2, "d": d, "eps": eps, "max_iterations": max_iterations,
        "converged": not changes or changes[-1] < eps, "changes": changes,
        "artists": table,
        "graph_nodes": len(adjacency), "graph_edges": sum(map(len, adjacency.values())),
        "catalog_artists": len(catalog_names),
        "missing_artists": sorted(catalog_names - {row["artist_name"] for row in rows}),
    }
