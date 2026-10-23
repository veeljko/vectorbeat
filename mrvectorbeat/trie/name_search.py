"""Usmeravanje predloga na prefiksno ili sufiksno trazenje."""

from collections.abc import Iterable

from mrvectorbeat.trie.trie import Trie, ReverseTrie


class NameSearch:
    def __init__(self, names: Iterable[tuple[str, str]]) -> None:
        self.forward = Trie()
        self.reverse = ReverseTrie()
        for name, _ in names:
            if not isinstance(name, str) or not name.strip():
                continue
            self.forward.insert(name)
            self.reverse.insert(name)

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
            left, right = pattern.split("*")
            # Presek mora ukljuciti sva poklapanja, ne samo top-k svake strane.
            prefix_matches = set(self.forward.find_words(left, k=None))
            suffix_matches = set(self.reverse.find_words("*" + right, k=None))
            matches = prefix_matches & suffix_matches
            # Prefiks i sufiks ne smeju da se preklapaju: ab*bc ne odgovara abc.
            minimum_length = len(left.casefold()) + len(right.casefold())
            matches = {name for name in matches if len(name.casefold()) >= minimum_length}
            return sorted(matches, key=lambda name: (-self.forward.get_frequency(name),
                                                     name.casefold()))[:k]
        return []
