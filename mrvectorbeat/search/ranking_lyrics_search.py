"""TF-IDF, kosinusno i BM25 rangiranje za pozicioni indeks stihova."""

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

    def score_ranked(self, tokens: list[str], measure: str = "tfidf", *,
                     k1: float = 1.5, b: float = 0.75) -> dict[int, float]:
        """Ocenjuje samo uniju postings lista, bez obilaska celog kataloga."""
        if measure == "bm25":
            return self._score_bm25_counts(Counter(tokens), k1, b)
        return self._score_counts(Counter(tokens), measure)

    def idf_bm25(self, term: str) -> float:
        """Nenegativni BM25 IDF, izracunat pri gradnji indeksa."""
        return self.bm25_idf_values.get(term, 0.0)

    def avgdl(self) -> float:
        """Procitaj sacuvanu prosecnu duzinu obradjenog teksta pesama."""
        return self.average_document_length

    @staticmethod
    def _validate_bm25(k1: float, b: float) -> None:
        if (isinstance(k1, bool) or not isinstance(k1, (int, float))
                or not math.isfinite(k1) or k1 < 0):
            raise ValueError("k1 mora biti konacan nenegativan broj.")
        if (isinstance(b, bool) or not isinstance(b, (int, float))
                or not math.isfinite(b) or not 0 <= b <= 1):
            raise ValueError("b mora biti broj izmedju 0 i 1.")

    def _bm25_term_score(self, term: str, frequency: int, doc_id: int,
                         k1: float, b: float) -> float:
        denominator = frequency + k1 * (1 - b + b * self.doc_len[doc_id] / self.avgdl())
        return self.idf_bm25(term) * frequency * (k1 + 1) / denominator

    def score_bm25(self, tokens: list[str], doc_id: int,
                    k1: float = 1.5, b: float = 0.75) -> float:
        """BM25 ocena jedne pesme; ponovljene reci upita ponavljaju doprinos."""
        self._validate_bm25(k1, b)
        if not self.avgdl():
            return 0.0
        score = 0.0
        for term, count in Counter(tokens).items():
            frequency = self.directory.get(term, {}).get(doc_id, 0)
            if frequency:
                score += count * self._bm25_term_score(term, frequency, doc_id, k1, b)
        return score

    def _score_bm25_counts(self, counts: dict[str, int], k1: float,
                           b: float) -> dict[int, float]:
        """Obidji samo postings liste upita, koristeci sacuvani IDF i duzine."""
        self._validate_bm25(k1, b)
        scores: dict[int, float] = {}
        if not self.avgdl():
            return scores
        for term, count in counts.items():
            for doc_id in self.index.get(term, {}):
                frequency = self.directory[term][doc_id]
                contribution = self._bm25_term_score(term, frequency, doc_id, k1, b)
                scores[doc_id] = scores.get(doc_id, 0.0) + count * contribution
        return scores

    def rank_bm25(self, query: str, k1: float = 1.5,
                   b: float = 0.75) -> list[tuple[int, float]]:
        """Obradi upit i sortiraj kandidate po BM25 oceni, pa po ID-u za iste ocene."""
        from mrvectorbeat.textprocessing.text_processor import TextProcessor

        tokens = TextProcessor(query).text_tokenize()
        scores = self.score_ranked(tokens, "bm25", k1=k1, b=b)
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))

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
