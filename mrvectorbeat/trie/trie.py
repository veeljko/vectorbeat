"""Prefiksna i sufiksna dopuna rangirana po ucestanosti imena."""

from __future__ import annotations

import heapq
from itertools import count


class Node:
    def __init__(self) -> None:
        self.children: dict[str, Node] = {}
        self.endCount: int = 0
        self.original: str | None = None
        self.max_frequency: int = 0


class Trie:
    """Imena se porede bez obzira na velika slova; prikazuje se prvi zapis.

    Svaki insert povecava broj pojavljivanja. Kod jednakih ucestanosti
    redosled predloga nije abecedni. Prazan prefiks vraca globalni top-k.
    """

    def __init__(self) -> None:
        self.root = Node()

    def _key(self, word: str) -> str:
        return word.casefold()

    def insert(self, word: str) -> None:
        if not isinstance(word, str):
            raise TypeError("Ime mora biti string")
        if not word.strip():
            raise ValueError("Ime ne sme biti prazno")
        node = self.root
        path = [node]
        for char in self._key(word):
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]
            path.append(node)
        node.endCount += 1
        if node.original is None:
            node.original = word
        for ancestor in path:
            ancestor.max_frequency = max(ancestor.max_frequency, node.endCount)

    def _find_node(self, prefix: str) -> Node | None:
        node = self.root
        for char in self._key(prefix):
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def search(self, word: str) -> bool:
        """Proveri celo ime; za obrasce koristi find_words."""
        if "*" in word:
            raise ValueError("Za obrazac sa zvezdicom koristi find_words")
        node = self._find_node(word)
        return node is not None and node.endCount > 0

    def get_frequency(self, word: str) -> int:
        node = self._find_node(word)
        return 0 if node is None else node.endCount

    def _top_k(self, node: Node | None, k: int | None) -> list[str]:
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 0):
            raise ValueError("k mora biti nenegativan ceo broj")
        if node is None or k == 0 or node.max_frequency == 0:
            return []

        
        serial = count()
        # Grana ima gornju granicu ucestanosti, terminal tacnu ucestanost.
        pending: list[tuple[int, int, bool, Node]] = []
        heapq.heappush(pending, (-node.max_frequency, next(serial), False, node))
        result: list[str] = []
        while pending and (k is None or len(result) < k):
            _, _, terminal, current = heapq.heappop(pending)
            if terminal:
                assert current.original is not None
                result.append(current.original)
                continue
            if current.endCount:
                heapq.heappush(pending, (-current.endCount, next(serial), True, current))
            for child in current.children.values():
                heapq.heappush(pending, (-child.max_frequency, next(serial), False, child))
        # Svi preostali prioriteti su <= poslednje vracene ucestanosti.
        return result

    def find_words_with_prefix(self, prefix: str, k: int | None = 10) -> list[str]:
        """Vrati top-k za prefiks/prefiks*; k=None vraca sva poklapanja."""
        if "*" in prefix:
            if prefix.count("*") == 1 and prefix.endswith("*"):
                prefix = prefix[:-1]
            elif prefix.count("*") == 1 and prefix.startswith("*"):
                raise ValueError("Za *sufiks koristi ReverseTrie")
            else:
                raise ValueError("Za unutrasnje ili vise zvezdica koristi k-gramski indeks")
        return self._top_k(self._find_node(prefix), k)

    def find_words(self, pattern: str, k: int | None = 10) -> list[str]:
        return self.find_words_with_prefix(pattern, k)


class ReverseTrie(Trie):
    """Cuva obrnuta imena i obradjuje samo obrasce oblika *sufiks."""

    def _key(self, word: str) -> str:
        return word.casefold()[::-1]

    def find_words_with_prefix(self, prefix: str, k: int | None = 10) -> list[str]:
        raise ValueError("Za prefiks ili prefiks* koristi Trie; ReverseTrie koristi *sufiks")

    def find_words(self, pattern: str, k: int | None = 10) -> list[str]:
        if pattern.count("*") == 1 and pattern.startswith("*"):
            return self._top_k(self._find_node(pattern[1:]), k)
        if "*" not in pattern or (pattern.count("*") == 1 and pattern.endswith("*")):
            raise ValueError("Za prefiks ili prefiks* koristi Trie")
        raise ValueError("Za unutrasnje ili vise zvezdica koristi k-gramski indeks")