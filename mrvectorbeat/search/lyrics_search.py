"""Pretraga nad jednim indeksom i iseci originalnih stihova."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape
import math

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.ranking.qlm import QLM
from mrvectorbeat.textprocessing.text_processor import TextProcessor


@dataclass
class LyricsResults:
    terms: list[str]
    # katalog_id -> svi pogodjeni rasponi obradjenih tokena [pocetak, kraj).
    hits: dict[int, list[tuple[int, int]]]
    scores: dict[int, float] = field(default_factory=dict)
    added_words: list[str] = field(default_factory=list)
    expansion_warning: str = ""


def _ranked_results(index: PositionIndex, terms: list[str], scores: dict[int, float],
                    families: Mapping[int, str] | None = None,
                    exclude_doc: int | None = None,
                    include_hits: bool = True) -> LyricsResults:
    """Najbolja verzija svake porodice; redosled recnika je rang rezultata."""
    results = LyricsResults(terms, {})
    unique_terms = set(terms)
    seen = set()
    if families is not None and exclude_doc in families:
        seen.add(families[exclude_doc])
    for doc_id in sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id)):
        family = families.get(doc_id, doc_id) if families is not None else doc_id
        if family in seen:
            continue
        seen.add(family)
        results.scores[doc_id] = scores[doc_id]
        results.hits[doc_id] = sorted(
            (position, position + 1)
            for term in unique_terms
            for position in index.index.get(term, {}).get(doc_id, [])
        ) if include_hits else []
    return results


def search_similar_lyrics(index: PositionIndex, doc_id: int, measure: str = "cosine",
                         families: Mapping[int, str] | None = None,
                         include_hits: bool = True) -> LyricsResults:
    terms = sorted(index.document_weights[doc_id])
    return _ranked_results(index, terms, index.score_similar(doc_id, measure),
                           families, exclude_doc=doc_id, include_hits=include_hits)


@dataclass(frozen=True)
class TermContribution:
    term: str
    source_count: int
    target_count: int
    contribution: float
    share: float


@dataclass
class LyricsComparison:
    similarity: float
    terms: list[TermContribution]
    source_html: str
    target_html: str


def highlight_terms(lyrics: str, terms: set[str]) -> str:
    """Oznaci sve izvorne oblike trazenih stemova, uz ocuvan i escapovan tekst."""
    if not terms:
        return escape(lyrics)
    tokens, spans = TextProcessor(lyrics).tokenize_with_spans()
    parts = []
    cursor = 0
    for token, (start, end) in zip(tokens, spans):
        if token in terms:
            parts.extend((escape(lyrics[cursor:start]), "<mark>", escape(lyrics[start:end]), "</mark>"))
            cursor = end
    parts.append(escape(lyrics[cursor:]))
    return "".join(parts)


def compare_lyrics(index: PositionIndex, source_id: int, target_id: int,
                   source_lyrics: str, target_lyrics: str, limit: int = 10) -> LyricsComparison:
    contributions = index.similarity_contributions(source_id, target_id)
    total = math.fsum(contributions.values())
    terms = [
        TermContribution(term, index.directory[term][source_id], index.directory[term][target_id],
                         contributions[term], contributions[term] / total)
        for term in sorted(contributions, key=lambda term: (-contributions[term], term))[:limit]
    ]
    highlighted = {item.term for item in terms}
    return LyricsComparison(total, terms, highlight_terms(source_lyrics, highlighted),
                            highlight_terms(target_lyrics, highlighted))


def search_lyrics(index: PositionIndex, query: str, mode: str = "boolean",
                  operator: str = "AND", distance: int = 3,
                  measure: str = "tfidf",
                  families: Mapping[int, str] | None = None, *,
                  k1: float = 1.5, b: float = 0.75,
                  include_hits: bool = True) -> LyricsResults:
    terms = TextProcessor(query).text_tokenize()
    results = LyricsResults(terms, {})
    if not terms:
        return results
    if mode == "ranked":
        scores = (index.score_ranked(terms, measure, k1=k1, b=b) if measure == "bm25"
                  else index.score_ranked(terms, measure))
        return _ranked_results(index, terms, scores, families, include_hits=include_hits)
    if mode == "phrase":
        for doc_id, position in index.find_phrase(terms):
            results.hits.setdefault(doc_id, []).append((position, position + len(terms)))
    elif mode == "proximity":
        if len(terms) != 2:
            raise ValueError("Za blizinu unesite tacno dva termina nakon obrade teksta.")
        for doc_id, first, second in index.find_proximity(*terms, distance):
            results.hits.setdefault(doc_id, []).append((min(first, second), max(first, second) + 1))
    elif mode == "boolean":
        documents = index.find_boolean(terms, operator)
        matched_terms = terms[:1] if operator.upper() == "DIFFERENCE" else terms
        for doc_id in documents:
            hits: list[tuple[int, int]] = []
            for term in matched_terms:
                positions = index.index.get(term, {}).get(doc_id, [])
                hits.extend((position, position + 1) for position in positions)
            results.hits[doc_id] = sorted(hits)
    else:
        raise ValueError("Nepoznat rezim pretrage.")
    return results


def search_lyrics_qlm(model: QLM, query: str,
                      families: Mapping[int, str] | None = None) -> LyricsResults:
    """Rank remembered lyrics with Dirichlet smoothing, without an AND filter.

    Ignore collection-OOV terms for scoring: Dirichlet cannot give them a
    positive probability. Keep the full query for word-coverage explanations.
    With no known terms, return no arbitrary zero-score or -inf results.
    """
    terms = TextProcessor(query).text_tokenize()
    known_terms = [term for term in terms if model.pc.get(term, 0.0) > 0]
    if not known_terms:
        return LyricsResults(terms, {})
    scores = dict(model.rank(known_terms, mode="dir", topk=len(model.idx.doc_len)))
    return _ranked_results(model.idx, terms, scores, families, include_hits=False)


def search_lyrics_with_synonyms(model: QLM, query: str,
                                families: Mapping[int, str] | None = None) -> LyricsResults:
    """Try the original QLM query, then automatically expand poor coverage once."""
    from mrvectorbeat.search.ranking.wordnet_expansion import (
        FEEDBACK_DOCUMENTS, expand_with_wordnet, needs_query_expansion,
    )

    results = search_lyrics_qlm(model, query, families)
    if not needs_query_expansion(model.idx, results.terms, results.scores):
        return results
    weights = Counter(term for term in results.terms if model.pc.get(term, 0.0) > 0)
    try:
        weights, added_words = expand_with_wordnet(
            model.idx, query, weights, list(results.scores)[:FEEDBACK_DOCUMENTS])
    except LookupError:
        results.expansion_warning = (
            "WordNet rečnik nije dostupan; prikazani su početni rezultati. "
            "Instalirajte ga komandom: python -m nltk.downloader wordnet, "
            "pa ponovo pokrenite aplikaciju.")
        return results
    if not added_words:
        return results
    scores = dict(model.rank(weights, mode="dir", topk=len(model.idx.doc_len)))
    expanded = _ranked_results(model.idx, results.terms, scores, families, include_hits=False)
    expanded.added_words = added_words
    return expanded


def query_word_coverage(query: str, lyrics: str) -> tuple[list[str], list[str]]:
    """Return present/missing query words in input order, without stemming.

    Include stopwords, compare whole words ignoring case and punctuation, and
    show each word once. This is independent of the processed ranking tokens:
    for example, 'hope' is missing from lyrics containing only 'hopeless'.
    """
    query_processor = TextProcessor(query)
    words = dict.fromkeys(query_processor.extract_words(query.lower()))
    lyrics_words = set(TextProcessor(lyrics).extract_words(lyrics.lower()))
    return ([word for word in words if word in lyrics_words],
            [word for word in words if word not in lyrics_words])


def make_snippets(lyrics: str, hits: list[tuple[int, int]], context: int = 8,
                  limit: int = 3) -> list[str]:
    """Do limit isecaka sa oznacenim pogocima i bezbedno escapovanim HTML-om."""
    if not hits or limit <= 0:
        return []
    if context < 0:
        raise ValueError("Kontekst ne sme biti negativan.")
    _, spans = TextProcessor(lyrics).tokenize_with_spans()
    ranges: list[tuple[int, int]] = []
    windows: list[tuple[int, int]] = []
    for start, end in sorted(hits):
        if not 0 <= start < end <= len(spans):
            raise ValueError("Pozicije nisu uskladjene sa stihovima. Ponovo pripremite artefakte.")
        left, right = spans[start][0], spans[end - 1][1]
        if ranges and left <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(right, ranges[-1][1]))
        else:
            ranges.append((left, right))
        window_start = 0 if start <= context else spans[start - context][0]
        window_end = len(lyrics) if end + context >= len(spans) else spans[end + context - 1][1]
        # Ne spajamo sve susedne prozore: cest termin ne treba da prikaze celu pesmu.
        if not windows or left >= windows[-1][1]:
            if len(windows) < limit:
                windows.append((window_start, window_end))
    snippets: list[str] = []
    for left, right in windows:
        parts = ["... " if left else ""]
        cursor = left
        for start, end in ranges:
            if end <= left:
                continue
            if start >= right:
                break
            start, end = max(start, left), min(end, right)
            parts.extend((escape(lyrics[cursor:start]), "<mark>", escape(lyrics[start:end]), "</mark>"))
            cursor = end
        parts.extend((escape(lyrics[cursor:right]), " ..." if right < len(lyrics) else ""))
        snippets.append("".join(parts))
    return snippets
