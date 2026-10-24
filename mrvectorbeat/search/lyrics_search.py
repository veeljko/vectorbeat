"""Pretraga nad jednim indeksom i iseci originalnih stihova."""

from bisect import bisect_left
from dataclasses import dataclass
from html import escape

from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.textprocessing.text_processor import TextProcessor


@dataclass
class LyricsResults:
    terms: list[str]
    # katalog_id -> svi pogodjeni rasponi obradjenih tokena [pocetak, kraj).
    hits: dict[int, list[tuple[int, int]]]


def search_lyrics(index: PositionIndex, query: str, mode: str = "boolean",
                  operator: str = "AND", distance: int = 3) -> LyricsResults:
    terms = TextProcessor(query).text_tokenize()
    results = LyricsResults(terms, {})
    if not terms:
        return results
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
                postings = index.index.get(term, [])
                cursor = bisect_left(postings, doc_id, key=lambda posting: posting[0])
                if cursor < len(postings) and postings[cursor][0] == doc_id:
                    hits.extend((position, position + 1) for position in postings[cursor][1])
            results.hits[doc_id] = sorted(hits)
    else:
        raise ValueError("Nepoznat rezim pretrage.")
    return results


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
