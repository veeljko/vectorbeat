"""TF-IDF i kosinusno rangiranje za pozicioni indeks stihova."""

from collections import Counter
import math


class RankingLyricsSearch:
    """Metode rangiranja koje koriste podatke sacuvane u PositionIndex."""

    def tf(self, term: str, doc_id: int) -> float:
        broj = self.directory.get(term, {}).get(doc_id)

        if not broj:
            return 0.0

        return 1 + math.log10(broj)

    def df(self, term : str) -> int:
        return len(self.directory.get(term,{}))

    def idf(self, term : str) -> float:
        return self.idf_values.get(term, 0.0)

    def tf_idf(self, term: str, doc_id: int) -> float:
        return self.document_weights.get(doc_id, {}).get(term, 0.0) * self.idf(term)

    def score_tf_idf(self, tokens: list[str]) -> dict[int, float]:
        """Suma log-TF * IDF tezina za svaki token upita (ponavljanja se broje)."""
        return self.score_ranked(tokens, measure="tfidf")

    def score_ranked(self, tokens: list[str], measure: str = "tfidf") -> dict[int, float]:
        """Ocenjuje samo uniju postings lista, bez obilaska celog kataloga."""
        return self._score_counts(Counter(tokens), measure)

    def _score_counts(self, counts: dict[str, int], measure: str) -> dict[int, float]:
        query_norm = 0.0
        if measure == "cosine":
            weights = {term: (1 + math.log10(count)) * self.idf(term)
                       for term, count in counts.items()}
            query_norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        else:
            weights = {term: count * self.idf(term) for term, count in counts.items()}
        return self._score_weights(weights, measure, query_norm)

    def _score_weights(self, weights: dict[str, float], measure: str,
                       query_norm: float = 0.0) -> dict[int, float]:
        if measure not in ("tfidf", "cosine"):
            raise ValueError("Mera mora biti tfidf ili cosine.")
        scores: dict[int, float] = {}
        for term, query_weight in weights.items():
            for doc_id in self.index.get(term, {}):
                weight = self.document_weights[doc_id][term]
                scores[doc_id] = scores.get(doc_id, 0.0) + query_weight * weight
        if measure == "cosine":
            for doc_id in scores:
                denominator = query_norm * self.document_norms[doc_id]
                scores[doc_id] = scores[doc_id] / denominator if denominator else 0.0
        return scores

    def score_similar(self, doc_id: int, measure: str = "cosine") -> dict[int, float]:
        """Izabrana pesma je upit; kandidati su samo pesme sa zajednickim recima."""
        if measure == "cosine":
            weights = {term: weight * self.idf(term)
                       for term, weight in self.document_weights[doc_id].items()}
        else:
            weights = {term: self.directory[term][doc_id] * self.idf(term)
                       for term in self.document_weights[doc_id]}
        scores = self._score_weights(weights, measure, self.document_query_norms[doc_id])
        scores.pop(doc_id, None)
        return scores

    def similarity_contributions(self, source_id: int, target_id: int) -> dict[str, float]:
        """Doprinosi termina istoj lnc.ltc oceni kao u score_similar(source_id)."""
        source = self.document_weights[source_id]
        target = self.document_weights[target_id]
        denominator = self.document_query_norms[source_id] * self.document_norms[target_id]
        if not denominator:
            return {}
        return {
            term: source[term] * self.idf_values[term] * target[term] / denominator
            for term in sorted(source.keys() & target.keys())
            if self.idf_values[term] > 0
        }

    def build_vec(self, tokens: list[str], use_idf: bool) -> tuple[dict[str, float], float]:
        """Vraca tezine po terminu i normu vektora."""
        tezine = {}
        for term, broj in Counter(tokens).items():
            w = 1 + math.log10(broj)
            if use_idf:
                w *= self.idf(term)
            tezine[term] = w

        norma = math.sqrt(sum(w * w for w in tezine.values()))
        return tezine, norma

    def cosine_lnc_ltc(self, query_tokens: list[str], doc_id: int) -> float:
        """Upit ltc i indeksirana pesma lnc, sa sacuvanom normom pesme."""
        wq, nq = self.build_vec(query_tokens, use_idf=True)     # ltc - upit ima idf
        wd = self.document_weights[doc_id]
        nd = self.document_norms[doc_id]

        if nq == 0 or nd == 0:
            return 0.0

        # skalarni proizvod samo po zajednickim terminima
        suma = sum(wq[t] * wd[t] for t in set(wq) & set(wd))
        return suma / (nq * nd)
