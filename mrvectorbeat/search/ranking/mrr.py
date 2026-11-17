"""Srednji reciprocni rang preko vise upita."""

from collections.abc import Hashable, Mapping, Sequence
import math

from .reciprocal_rank import reciprocal_rank


def mrr(ranked: Mapping[Hashable, Sequence[Hashable]],
        qrels: Mapping[Hashable, Mapping[Hashable, float]]) -> float:
    """Prosek RR za upite iz ranked; bez ocena ili pogotka RR je 0.

    Prazan skup upita daje 0. Prazne liste rezultata ulaze u prosek kao 0.
    """
    if not ranked:
        return 0.0
    return math.fsum(reciprocal_rank(results, qrels.get(query_id, {}))
                     for query_id, results in ranked.items()) / len(ranked)
