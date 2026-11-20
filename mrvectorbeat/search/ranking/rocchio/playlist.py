"""Sparse Rocchio recommendations from explicit or assumed song judgments."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math

from mrvectorbeat.textprocessing.text_processor import TextProcessor
from .rocchio import _validate_coefficient, rank_vector, rocchio


@dataclass(frozen=True)
class PlaylistTerm:
    term: str
    weight: float
    source_ids: tuple[int, ...]
    contribution: float


@dataclass
class FeedbackPlaylist:
    songs: list[tuple[int, float]] = field(default_factory=list)
    terms: list[PlaylistTerm] = field(default_factory=list)
    avoided_terms: list[str] = field(default_factory=list)
    relevant: list[int] = field(default_factory=list)
    non_relevant: list[int] = field(default_factory=list)
    assumed: bool = False
    weights: dict[str, float] = field(default_factory=dict)


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    norm = math.sqrt(math.fsum(weight * weight for weight in weights.values()))
    return {term: weight / norm for term, weight in weights.items() if weight > 0} if norm else {}


def build_feedback_playlist(index, query: str, relevant: Iterable[int] = (),
                            non_relevant: Iterable[int] = (), *,
                            displayed_ids: Iterable[int] = (),
                            initial_results: Iterable[int] = (),
                            families: Mapping[int, str] | None = None,
                            limit: int = 10, pseudo_k: int = 5,
                            alpha: float = 1.0, beta: float = 0.75,
                            gamma: float = 0.15, expansion_ratio: float = 0.5,
                            max_new_terms: int = 20) -> FeedbackPlaylist:
    """Recommend unseen song families using existing sparse Rocchio/cosine.

    With no judgments, use the first pseudo_k displayed results (one per
    family) as positives. Negative-only judgments do not imply positive ones.
    Each surviving typed term stays stronger than every added term. New terms
    come only from positive song vectors, and are uniformly scaled to preserve
    their relative weights before L2 normalization. No document-term matrix or
    corpus-wide vector scan is built.
    """
    for name, value in (("limit", limit), ("pseudo_k", pseudo_k), ("max_new_terms", max_new_terms)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        _validate_coefficient(name, value)
    if (isinstance(expansion_ratio, bool) or not isinstance(expansion_ratio, (int, float))
            or not math.isfinite(expansion_ratio) or not 0 < expansion_ratio < 1):
        raise ValueError("expansion_ratio must be between 0 and 1, exclusively.")

    relevant = list(dict.fromkeys(relevant))
    non_relevant = list(dict.fromkeys(non_relevant))
    if set(relevant).intersection(non_relevant):
        raise ValueError("A song cannot be both relevant and non-relevant.")
    for doc_id in relevant + non_relevant:
        if doc_id not in index.doc_len:
            raise ValueError(f"Unknown feedback document: {doc_id}")
    displayed = set(displayed_ids)
    family = lambda doc_id: families.get(doc_id, doc_id) if families is not None else doc_id
    assumed = not relevant and not non_relevant
    if assumed and pseudo_k:
        seen_families = set()
        for doc_id in initial_results:
            if doc_id not in displayed or doc_id not in index.doc_len or family(doc_id) in seen_families:
                continue
            relevant.append(doc_id)
            seen_families.add(family(doc_id))
            if len(relevant) == pseudo_k:
                break

    result = FeedbackPlaylist(relevant=relevant, non_relevant=non_relevant, assumed=assumed)
    query_terms = TextProcessor(query).text_tokenize()
    original_terms = set(query_terms)
    base, _ = index.build_vec(query_terms, use_idf=True)
    base = _normalize(base)
    # Two applications give exactly alpha*q + beta*centroid(rel) -
    # gamma*centroid(non-rel), and expose the actual reduction for explanation.
    positive = rocchio(index, base, relevant, [], alpha=alpha, beta=beta, gamma=0.0)
    adjusted = rocchio(index, positive, [], non_relevant, alpha=1.0, beta=0.0, gamma=gamma)
    reductions = {term: weight - adjusted.get(term, 0.0) for term, weight in positive.items()
                  if weight > adjusted.get(term, 0.0)}
    result.avoided_terms = sorted(reductions, key=lambda term: (-reductions[term], term))[:10]

    added = sorted((term for term in adjusted if term not in original_terms),
                   key=lambda term: (-adjusted[term], term))[:max_new_terms]
    typed = {term: weight for term, weight in adjusted.items() if term in original_terms}
    # If all typed coordinates were removed or OOV, there is no surviving
    # typed weight to compare to; use the original query scale (or unit scale).
    reference = min(typed.values()) if typed else min(base.values(), default=1.0)
    scale = min(1.0, expansion_ratio * reference / max((adjusted[term] for term in added), default=1.0))
    weights = dict(typed)
    weights.update((term, adjusted[term] * scale) for term in added)
    result.weights = _normalize(weights)
    if not limit or not result.weights:
        return result

    excluded = {family(doc_id) for doc_id in displayed.union(relevant, non_relevant)}
    # Filter and collapse families BEFORE taking ten, so excluded high scores
    # and remixes cannot consume the recommendation slots.
    for doc_id, score in rank_vector(index, result.weights, topk=len(index.doc_len)):
        if score <= 0 or family(doc_id) in excluded:
            continue
        result.songs.append((doc_id, score))
        excluded.add(family(doc_id))
        if len(result.songs) == limit:
            break

    query_norm = math.sqrt(sum(weight * weight for weight in result.weights.values()))
    explanations = []
    for term, weight in result.weights.items():
        sources = tuple(doc_id for doc_id in relevant if term in index.document_weights[doc_id])
        if not sources:
            continue
        contribution = math.fsum(
            weight * index.document_weights[doc_id].get(term, 0.0)
            / (query_norm * index.document_norms[doc_id])
            for doc_id, _ in result.songs if index.document_norms[doc_id] > 0)
        if contribution > 0:
            explanations.append(PlaylistTerm(term, weight, sources, contribution))
    # Prefer actual new words from liked lyrics over repeating the user's query.
    explanations.sort(key=lambda item: (item.term in original_terms, -item.contribution, item.term))
    result.terms = explanations[:10]
    return result
