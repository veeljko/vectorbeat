"""WordNet synonym candidates, selected with existing Rocchio feedback."""

from collections.abc import Iterable, Mapping
from functools import lru_cache
from itertools import islice
import math

from mrvectorbeat.textprocessing.text_processor import TextProcessor
from .rocchio.rocchio import rocchio


MIN_QUERY_COVERAGE = 0.8
FEEDBACK_DOCUMENTS = 5
SYNONYM_WEIGHT = 0.25
MAX_EXPANSION_TERMS = 5
MAX_SYNONYMS_PER_WORD = 2


def needs_query_expansion(index, terms: Iterable[str], ranked_ids: Iterable[int]) -> bool:
    """Expand if none of the first five songs covers 80% of unique query terms.

    Include collection-OOV terms in the denominator, but exclude stopwords via
    the query's usual preprocessing. This is a coverage heuristic, not a
    confidence probability derived from the uncalibrated QLM log score.
    """
    terms = set(terms)
    if not terms:
        return False
    coverage = max((len(terms.intersection(index.doc_terms.get(doc_id, ()))) / len(terms)
                    for doc_id in islice(ranked_ids, FEEDBACK_DOCUMENTS)), default=0.0)
    return coverage < MIN_QUERY_COVERAGE


@lru_cache(maxsize=4096)
def _wordnet_synonyms(word: str) -> tuple[str, ...]:
    """Look up the unstemmed word; WordNet also handles inflected forms.

    Limit to the first two senses per part of speech, including adjective
    satellites with adjectives. Read only lemmas of those synsets, without
    adding hypernyms, related words or antonyms.
    """
    from nltk.corpus import wordnet

    senses = {}
    synonyms = {}
    for synset in wordnet.synsets(word):
        pos = synset.pos()
        pos = "a" if pos == "s" else pos
        if senses.get(pos, 0) >= 2:
            continue
        senses[pos] = senses.get(pos, 0) + 1
        for name in synset.lemma_names():
            name = name.lower()
            # Single-word expansion keeps displayed words and scored terms in
            # one-to-one correspondence, instead of splitting WordNet phrases.
            if name.isalpha():
                synonyms.setdefault(name, None)
    return tuple(synonyms)


def expand_with_wordnet(index, query: str, original_weights: Mapping[str, float],
                        feedback_docs: Iterable[int] = (), *,
                        synonym_weight: float = SYNONYM_WEIGHT,
                        max_terms: int = MAX_EXPANSION_TERMS
                        ) -> tuple[dict[str, float], list[str]]:
    """Return (weighted processed query, added display words).

    Existing Rocchio orders only WordNet-approved candidates using up to five
    initial matching songs. It cannot introduce arbitrary feedback words.
    Original weights are preserved; new stems get weight 0.25 by default.
    Add at most two synonyms per source word and five in total, deduplicating
    both display forms and stems. Keep only terms present in the collection.
    """
    if (isinstance(synonym_weight, bool) or not isinstance(synonym_weight, (int, float))
            or not math.isfinite(synonym_weight) or not 0 < synonym_weight < 1):
        raise ValueError("synonym_weight must be between 0 and 1, exclusively.")
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 0:
        raise ValueError("max_terms must be a nonnegative integer.")
    weights = dict(original_weights)
    if not max_terms:
        return weights, []

    processor = TextProcessor(query)
    words = dict.fromkeys(processor.extract_words(query.lower()))
    original_terms = set(processor.text_tokenize())
    # term -> (display word, source words); insertion order retains WordNet's
    # sense order when feedback supplies no preference.
    candidates = {}
    for word in words:
        if not TextProcessor(word).text_tokenize():
            continue  # Do not expand stopwords removed from the scoring query.
        for synonym in _wordnet_synonyms(word):
            tokens = TextProcessor(synonym).text_tokenize()
            if len(tokens) != 1:
                continue
            term = tokens[0]
            if term in original_terms or term in weights or term not in index.directory:
                continue
            if term not in candidates:
                candidates[term] = (synonym, [])
            sources = candidates[term][1]
            if word not in sources:
                sources.append(word)
    if not candidates:
        return weights, []

    feedback_docs = [doc_id for doc_id in dict.fromkeys(feedback_docs)
                     if original_terms.intersection(index.doc_terms.get(doc_id, ()))]
    feedback = rocchio(index, weights, feedback_docs[:FEEDBACK_DOCUMENTS], [],
                       alpha=1.0, beta=0.75, gamma=0.0)
    # sorted() is stable: equal feedback weights keep WordNet/query order.
    ordered = sorted(candidates, key=lambda term: -feedback.get(term, 0.0))
    per_word = {}
    added_words = []
    for term in ordered:
        synonym, sources = candidates[term]
        source = next((word for word in sources
                       if per_word.get(word, 0) < MAX_SYNONYMS_PER_WORD), None)
        if source is None:
            continue
        weights[term] = float(synonym_weight)
        added_words.append(synonym)
        per_word[source] = per_word.get(source, 0) + 1
        if len(added_words) == max_terms:
            break
    return weights, added_words
