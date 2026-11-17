"""DCG i normalizovani DCG sa direktnim ocenama relevantnosti."""

from collections.abc import Hashable, Mapping, Sequence
import math


def dcg(ocene: Sequence[float], k: int = 5) -> float:
    """Zbir ocena / log2(pozicija + 1) na prvih k mesta (pozicije krecu od 1)."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k mora biti pozitivan ceo broj.")
    return math.fsum(grade / math.log2(position + 1)
                     for position, grade in enumerate(ocene[:k], start=1))


def ndcg_at_k(vraceni: Sequence[Hashable], qrels_q: Mapping[Hashable, float],
              k: int = 5) -> float:
    """nDCG@k: DCG vracene liste / DCG idealnog poretka svih ocenjenih pesama.

    Koristi direktne ocene, pa ocena 2 doprinosi vise od ocene 1.
    Nepoznate i nerelevantne pesme dobijaju 0. Bez relevantnih vraca 0.
    Rang-lista sadrzi jedinstvene ID-eve pesama.
    """
    ideal = sorted((grade for grade in qrels_q.values() if grade > 0), reverse=True)
    ideal_dcg = dcg(ideal, k)
    if not ideal_dcg:
        return 0.0
    grades = [max(0.0, qrels_q.get(doc_id, 0.0)) for doc_id in vraceni[:k]]
    return dcg(grades, k) / ideal_dcg
