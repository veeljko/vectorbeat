"""Preciznost na prvih k mesta rang-liste."""

from collections.abc import Hashable, Mapping, Sequence


def precision_at_k(vraceni: Sequence[Hashable], qrels_q: Mapping[Hashable, float],
                   k: int = 5) -> float:
    """P@k: broj relevantnih medju prvih k, podeljen sa k.

    Relevantna je pesma sa ocenom vecom od 0. Nepopunjena mesta do k
    racunaju se kao promasaji. Rang-lista sadrzi jedinstvene ID-eve pesama.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k mora biti pozitivan ceo broj.")
    return sum(qrels_q.get(doc_id, 0) > 0 for doc_id in vraceni[:k]) / k
