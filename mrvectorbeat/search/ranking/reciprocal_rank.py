"""Reciprocni rang prvog relevantnog rezultata."""

from collections.abc import Hashable, Iterable, Mapping


def reciprocal_rank(vraceni: Iterable[Hashable], qrels_q: Mapping[Hashable, float]) -> float:
    """RR = 1 / pozicija prvog rezultata sa ocenom > 0; bez pogotka vraca 0."""
    for position, doc_id in enumerate(vraceni, start=1):
        if qrels_q.get(doc_id, 0) > 0:
            return 1.0 / position
    return 0.0
