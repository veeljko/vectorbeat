"""Usmeravanje predloga na prefiksno ili sufiksno trazenje."""

from collections.abc import Iterable

from mrvectorbeat.trie.trie import Trie, ReverseTrie


class NameSearch:
    def __init__(self, names: Iterable[tuple[str, str]]) -> None:
        self.forward = Trie()
        self.reverse = ReverseTrie()
        self.types: dict[str, set[str]] = {}
        self.frequencies: dict[str, int] = {}
        for name, kind in names:
            if not isinstance(name, str) or not name.strip():
                continue
            key = name.casefold()
            self.forward.insert(name)
            self.reverse.insert(name)
            self.types.setdefault(key, set()).add(kind)
            self.frequencies[key] = self.frequencies.get(key, 0) + 1

    def suggest(self, pattern: str, k: int = 10) -> list[str]:
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k mora biti nenegativan ceo broj")
        if not pattern or k == 0:
            return []
        if pattern.count("*") > 1:
            pass
        elif "*" not in pattern or pattern.endswith("*"):
            return self.forward.find_words(pattern, k)
        elif pattern.startswith("*"):
            return self.reverse.find_words(pattern, k)
        else:
            # Zvezdica u sredini: implementirati kasnije.
            pass
        return []
