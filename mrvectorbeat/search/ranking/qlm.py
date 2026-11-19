"""Query Likelihood Model with MLE and Dirichlet smoothing."""

from collections import Counter
from collections.abc import Iterable
import math


class QLM:
    """Score queries using the raw frequencies stored in PositionIndex.

    The index supplies directory (term -> document -> occurrence count) and
    doc_len. Its tf() method is a logarithmic TF-IDF weight, so it must not be
    used as a count here. Collection probabilities are computed once when the
    model is created and can be saved together with it using pickle.
    """

    def __init__(self, index, mu: float = 1500.0):
        if (isinstance(mu, bool) or not isinstance(mu, (int, float))
                or not math.isfinite(mu) or mu <= 0):
            raise ValueError("mu must be a finite positive number.")
        self.idx = index
        self.mu = float(mu)
        collection_length = sum(index.doc_len.values())
        self.pc = ({term: sum(posting.values()) / collection_length
                    for term, posting in index.directory.items()}
                   if collection_length else {})

    def p_mle(self, term: str, doc_id: int) -> float:
        """Return tf(term, document) / document length, or zero if empty."""
        length = self.idx.doc_len[doc_id]
        frequency = self.idx.directory.get(term, {}).get(doc_id, 0)
        return frequency / length if length else 0.0

    def p_dirichlet(self, term: str, doc_id: int) -> float:
        """Interpolate document counts with the saved collection probability."""
        frequency = self.idx.directory.get(term, {}).get(doc_id, 0)
        length = self.idx.doc_len[doc_id]
        return (frequency + self.mu * self.pc.get(term, 0.0)) / (length + self.mu)

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in ("mle", "dir"):
            raise ValueError("mode must be 'mle' or 'dir'.")

    @staticmethod
    def _query_counts(query: str | Iterable[str]) -> Counter[str]:
        if isinstance(query, str):
            from mrvectorbeat.textprocessing.text_processor import TextProcessor

            query = TextProcessor(query).text_tokenize()
        return Counter(query)

    def _score_counts(self, counts: Counter[str], doc_id: int, mode: str) -> float:
        probability = self.p_mle if mode == "mle" else self.p_dirichlet
        score = 0.0
        for term, count in counts.items():
            value = probability(term, doc_id)
            if value == 0:
                return float("-inf")
            score += count * math.log(value)
        return score

    def score(self, query: str | Iterable[str], doc_id: int, mode: str = "dir") -> float:
        """Return log p(query | document), counting repeated query words.

        Zero probability gives -inf. Dirichlet smoothing is positive only for
        words present in the collection; an unseen collection word still gives
        -inf. An empty query has a score of zero.
        """
        self._validate_mode(mode)
        return self._score_counts(self._query_counts(query), doc_id, mode)

    def rank(self, query: str | Iterable[str], mode: str = "dir",
             topk: int = 10) -> list[tuple[int, float]]:
        """Score all documents and return the first topk (document ID, score).

        Preprocess a string query once. Ties keep the index's document order;
        -inf results are retained, matching the scoring definition.
        """
        self._validate_mode(mode)
        if isinstance(topk, bool) or not isinstance(topk, int) or topk < 0:
            raise ValueError("topk must be a nonnegative integer.")
        if topk == 0:
            return []
        counts = self._query_counts(query)
        scores = [(doc_id, self._score_counts(counts, doc_id, mode))
                  for doc_id in self.idx.doc_len]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:topk]
