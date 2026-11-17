"""Prosecna preciznost za jedan upit."""

from collections.abc import Hashable, Iterable, Mapping


def average_precision(vraceni: Iterable[Hashable], qrels_q: Mapping[Hashable, float],
                      ispisi: bool = False) -> float:
    """AP: zbir preciznosti na pogocima / broj SVIH relevantnih pesama.

    U imenilac ulaze i relevantne pesme koje nisu vracene. Relevantnost je
    ocena > 0, a rang-lista sadrzi jedinstvene ID-eve. Bez relevantnih vraca 0.
    Opcioni ispis prikazuje prvih sest pogodaka, kao u demonstraciji.
    """
    relevant = {doc_id for doc_id, grade in qrels_q.items() if grade > 0}
    if not relevant:
        return 0.0

    hits = 0
    total = 0.0
    for position, doc_id in enumerate(vraceni, start=1):
        if doc_id in relevant:
            hits += 1
            precision = hits / position
            total += precision
            if ispisi and hits <= 6:
                print(f"    pogodak {hits} na poziciji {position:>3}: P@{position} = {precision:.3f}")
    return total / len(relevant)
