"""QLM pseudo-relevance feedback with KL term contributions."""

from collections import defaultdict
from collections.abc import Iterable
import math

from .qlm import QLM


def _validate_size(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


class PRFRanking:
    """Expand queries using KL contributions from top-k Dirichlet QLM hits.

    Reuse one instance for multiple queries: its QLM computes collection
    probabilities once. Only feedback documents supply expansion candidates;
    the original and expanded queries are ranked by the same QLM.
    """

    def __init__(self, index, mu: float = 1500.0):
        self.idx = index
        self.model = QLM(index, mu=mu)

    @staticmethod
    def _query_tokens(query_terms: str | Iterable[str]) -> list[str]:
        if isinstance(query_terms, str):
            from mrvectorbeat.textprocessing.text_processor import TextProcessor

            return TextProcessor(query_terms).text_tokenize()
        return list(query_terms)

    def _has_likelihood(self, tokens: list[str]) -> bool:
        # Empty queries or unseen collection words must not produce arbitrary
        # pseudo-relevant documents from ties at zero or negative infinity.
        return bool(tokens) and all(self.model.pc.get(term, 0.0) > 0 for term in tokens)

    def _feedback_terms(self, tokens: list[str], k: int, m: int
                        ) -> tuple[list[tuple[str, float]], list[int]]:
        if not k or not self._has_likelihood(tokens):
            return [], []
        top_k = [doc_id for doc_id, score in self.model.rank(tokens, mode="dir", topk=k)
                 if math.isfinite(score)]
        if not m:
            return [], top_k
        contributions: dict[str, float] = defaultdict(float)
        for doc_id in top_k:
            if not self.idx.doc_len[doc_id]:
                continue
            # doc_terms already contains each term once. p_mle uses raw counts,
            # unlike the index's logarithmic TF-IDF tf() method.
            for term in self.idx.doc_terms.get(doc_id, ()):
                p_d = self.model.p_mle(term, doc_id)
                p_c = self.model.pc.get(term, 0.0)
                if p_d > 0 and p_c > 0:
                    contributions[term] += p_d * math.log(p_d / p_c)
        original = set(tokens)
        candidates = [(term, score) for term, score in contributions.items()
                      if term not in original and score > 0]
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        return candidates[:m], top_k

    def prf_terms(self, query_terms: str | Iterable[str], k: int = 5, m: int = 5
                  ) -> tuple[list[tuple[str, float]], list[int]]:
        """Return (selected (term, KL contribution) pairs, feedback document IDs).

        Contributions sum p_MLE(term|document) * log(p_MLE/p_collection) over
        the initial top-k documents, including negative per-document terms.
        Select at most m new terms with positive total contributions. Ties
        use alphabetical order. Empty feedback documents contribute zero.
        """
        _validate_size("k", k)
        _validate_size("m", m)
        return self._feedback_terms(self._query_tokens(query_terms), k, m)

    def rank(self, query_terms: str | Iterable[str], k: int = 5, m: int = 5,
             topk: int = 10) -> list[tuple[int, float]]:
        """Append each selected term once and rank again with Dirichlet QLM.

        Original query repetitions are preserved. KL contributions select the
        expansion terms; they are not interpreted as token occurrence counts.
        k=0 or m=0 gives the original QLM ranking. Empty queries and queries
        with unseen collection words return no results.
        """
        for name, value in (("k", k), ("m", m), ("topk", topk)):
            _validate_size(name, value)
        if not topk:
            return []
        tokens = self._query_tokens(query_terms)
        if not self._has_likelihood(tokens):
            return []
        selected, _ = self._feedback_terms(tokens, k, m) if k and m else ([], [])
        expanded = tokens + [term for term, _ in selected]
        return self.model.rank(expanded, mode="dir", topk=topk)


def prf_terms(index, query_terms: str | Iterable[str], k: int = 5, m: int = 5,
              mu: float = 1500.0) -> tuple[list[tuple[str, float]], list[int]]:
    """Convenience term selector; reuse PRFRanking for repeated queries."""
    return PRFRanking(index, mu=mu).prf_terms(query_terms, k=k, m=m)
