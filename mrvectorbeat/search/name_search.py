"""Usmeravanje predloga na prefiksno ili sufiksno trazenje."""

from collections.abc import Iterable

from mrvectorbeat.search.trie.trie import Trie, ReverseTrie
from mrvectorbeat.search.kgram.kgram import KGram


class NameSearch:
    def __init__(self, names: Iterable[tuple[str, str]]) -> None:
        self.forward = Trie()
        self.reverse = ReverseTrie()
        indexed_names: list[str] = []
        for name, _ in names:
            if not isinstance(name, str) or not name.strip():
                continue
            self.forward.insert(name)
            self.reverse.insert(name)
            indexed_names.append(name)
        self.kgram = KGram(indexed_names, k=3)

    def suggest(self, pattern: str, k: int = 10) -> list[str]:
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k mora biti nenegativan ceo broj")
        if not pattern or k == 0:
            return []
        if "*" not in pattern:
            matches = self.forward.find_words(pattern, k)
            if matches:
                return matches
            correction = self.kgram.correct(pattern)
            return [correction[0]] if correction else []
        if pattern.count("*") == 1:
            if pattern.endswith("*"):
                return self.forward.find_words(pattern, k)
            if pattern.startswith("*"):
                return self.reverse.find_words(pattern, k)
        matches = self.kgram.wildcard_pretraga(pattern)
        return sorted(matches, key=lambda name: (-self.forward.get_frequency(name),
                                                 name.casefold()))[:k]
