"""Srednja prosecna preciznost preko vise upita."""

from collections.abc import Hashable, Mapping, Sequence
import math

from .average_precision import average_precision


def map_score(ranked: Mapping[Hashable, Sequence[Hashable]],
              qrels: Mapping[Hashable, Mapping[Hashable, float]]) -> float:
    """Prosek AP za upite iz ranked; upiti bez ocena imaju AP 0.

    Prazan skup upita daje 0. Prazne liste rezultata ulaze u prosek kao 0.
    """
    if not ranked:
        return 0.0
    return math.fsum(average_precision(results, qrels.get(query_id, {}))
                     for query_id, results in ranked.items()) / len(ranked)
