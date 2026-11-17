"""BM25 ocenjivanje i rangiranje stihova."""

from collections import Counter
import math


class BM25Ranking:
    """BM25 metode nad sacuvanim frekvencijama, duzinama i IDF vrednostima."""

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
