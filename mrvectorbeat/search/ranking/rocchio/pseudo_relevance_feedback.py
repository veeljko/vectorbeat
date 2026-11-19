"""Pseudo-relevance feedback using top-ranked documents and positive Rocchio."""

from collections.abc import Mapping

from .rocchio import _validate_coefficient, rank_vector, rocchio


def pseudo_relevance_feedback(index, q_vec: Mapping[str, float], k: int = 5,
                              alpha: float = 1.0, beta: float = 0.75
                              ) -> tuple[dict[str, float], list[int]]:
    """Treat the initial top-k cosine results as relevant and expand the query.

    Return (new query vector, document IDs used). Rocchio receives no negative
    examples and gamma=0. The centroid divides by the number of actual hits,
    even when fewer than k documents match. No hits means only alpha*q remains.
    The index is explicit because ranking and feedback use its saved vectors.
    """
    _validate_coefficient("alpha", alpha)
    _validate_coefficient("beta", beta)
    top_k = [doc_id for doc_id, _ in rank_vector(index, q_vec, topk=k)]
    return rocchio(index, q_vec, top_k, [], alpha, beta, gamma=0.0), top_k
