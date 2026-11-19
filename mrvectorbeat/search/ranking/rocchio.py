"""Sparse query vectors, cosine ranking and Rocchio relevance feedback."""

from collections.abc import Iterable, Mapping
import math


def _validate_coefficient(name: str, value: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number.")


def _query_vector(q_vec: Mapping[str, float]) -> dict[str, float]:
    """Copy already weighted query terms, omitting zero coordinates."""
    weights = {}
    for term, weight in q_vec.items():
        _validate_coefficient("Query weight", weight)
        if weight:
            weights[term] = float(weight)
    return weights


def rank_vector(index, q_vec: Mapping[str, float], topk: int = 10) -> list[tuple[int, float]]:
    """Rank a weighted query using the index's existing lnc.ltc cosine scorer.

    A feedback vector already contains term weights: do not tokenize it or
    apply TF/IDF again. Only documents in its terms' postings are candidates.
    """
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 0:
        raise ValueError("topk must be a nonnegative integer.")
    weights = _query_vector(q_vec)
    if not weights or topk == 0:
        return []
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    scores = index._score_weights(weights, "cosine", norm)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:topk]


def rocchio(index, q_vec: Mapping[str, float], relevant: Iterable[int],
            non_relevant: Iterable[int] = (), alpha: float = 1.0,
            beta: float = 0.75, gamma: float = 0.15) -> dict[str, float]:
    """Return alpha*q + beta*mean(relevant) - gamma*mean(non-relevant).

    Feedback documents use normalized TF-IDF (ltc) vectors, the same term
    weighting as query vectors. Read saved IDFs and document query norms;
    visit only the supplied documents. Duplicate judgments count once.
    Empty feedback sets contribute zero, and negative coordinates are clipped
    to zero. The original vector and index remain unchanged.
    """
    for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        _validate_coefficient(name, value)
    weights = {term: alpha * weight for term, weight in _query_vector(q_vec).items()}
    for documents, coefficient in ((relevant, beta), (non_relevant, -gamma)):
        documents = list(dict.fromkeys(documents))
        if not documents or coefficient == 0:
            continue
        scale = coefficient / len(documents)
        for doc_id in documents:
            norm = index.document_query_norms[doc_id]
            if not norm:
                continue
            for term, tf_weight in index.document_weights[doc_id].items():
                contribution = scale * tf_weight * index.idf_values[term] / norm
                if contribution:
                    weights[term] = weights.get(term, 0.0) + contribution
    return {term: weight for term, weight in weights.items() if weight > 0}
