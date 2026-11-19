"""Artist similarity from weighted shared incoming pages of the existing graph."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from urllib.parse import unquote, urlsplit


def build_similarity_index(graph: Mapping[str, Iterable[str]],
                           artist_nodes: Mapping[str, str]) -> dict:
    """Build sparse incoming-page indexes without crawling or comparing all pairs.

    A page linking to k distinct catalog artist nodes contributes 1/k to each
    pair it connects. Duplicate links and catalog aliases do not increase k.
    """
    names_by_node = defaultdict(list)
    for name, node in artist_nodes.items():
        if isinstance(node, str) and node:
            names_by_node[node].append(name)
    artist_urls = set(names_by_node)
    available = set(graph)
    incoming = {node: [] for node in artist_urls}
    page_artists, page_weights, page_titles = {}, {}, {}
    for page, targets in graph.items():
        targets = set(targets)
        available.update(targets)
        linked = artist_urls.intersection(targets)
        if not linked:
            continue
        page_artists[page] = tuple(sorted(linked))
        page_weights[page] = 1.0 / len(linked)
        title = unquote(urlsplit(page).path).removeprefix("/wiki/").replace("_", " ")
        page_titles[page] = title or page
        for node in linked:
            incoming[node].append(page)
    names_by_node = {node: tuple(sorted(names, key=lambda name: (name.casefold(), name)))
                     for node, names in names_by_node.items() if node in available}
    return {
        "version": 1,
        "artist_nodes": {name: node for node, names in names_by_node.items() for name in names},
        "node_artists": names_by_node,
        "artist_pages": {node: tuple(sorted(incoming[node])) for node in names_by_node},
        "page_artists": page_artists, "page_weights": page_weights, "page_titles": page_titles,
    }


def find_similar_artists(index: dict, artist: str, limit: int = 10) -> list[dict]:
    """Score only artists reached through the chosen artist's incoming pages.

    Return one result per other artist node, with its catalog name, score and
    all shared source pages (including each page's contribution). No inputs
    are modified. Alternate catalog names of the selected node are excluded.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer.")
    source = index["artist_nodes"].get(artist)
    if source is None or limit == 0:
        return []
    candidates = defaultdict(list)
    for page in index["artist_pages"][source]:
        for target in index["page_artists"][page]:
            if target != source:
                candidates[target].append(page)
    results = []
    for node, pages in candidates.items():
        shared_pages = [{"url": page, "title": index["page_titles"][page],
                         "weight": index["page_weights"][page]} for page in pages]
        shared_pages.sort(key=lambda page: (-page["weight"], page["title"].casefold(), page["url"]))
        names = index["node_artists"][node]
        results.append({"artist_name": names[0], "aliases": names, "url": node,
                        "score": math.fsum(page["weight"] for page in shared_pages),
                        "shared_pages": shared_pages})
    results.sort(key=lambda row: (-row["score"], row["artist_name"].casefold(), row["artist_name"]))
    return results[:limit]
