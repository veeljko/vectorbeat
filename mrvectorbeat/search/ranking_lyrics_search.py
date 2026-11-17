"""Zajednicki izbor modela za rangiranje stihova."""

from collections import Counter

from mrvectorbeat.search.ranking import BM25Ranking, TFIDFRanking


class RankingLyricsSearch(TFIDFRanking, BM25Ranking):
    """Objedinjuje TF-IDF, kosinusnu slicnost i BM25 za PositionIndex."""

    def score_ranked(self, tokens: list[str], measure: str = "tfidf", *,
                     k1: float = 1.5, b: float = 0.75) -> dict[int, float]:
        """Ocenjuje samo uniju postings lista, bez obilaska celog kataloga."""
        if measure == "bm25":
            return self._score_bm25_counts(Counter(tokens), k1, b)
        return self._score_counts(Counter(tokens), measure)
