

class Node:
    def __init__(self):
        self.children = {}
        self.end = False

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word: str):
      node = self.root
      for c in word:
        if c not in node.children:
          node.children[c] = Node()
        node = node.children[c]
      node.end = True

    def search(self, word: str):
      node = self.root

      for c in word:
        if c not in node.children:
          return False

        node = node.children[c]

      return node.end

    def _find_node(self, prefix: str):
        """Vraca cvor na kraju prefiksa, ili None ako put ne postoji."""
        node = self.root
        for c in prefix:
            if c not in node.children:
                return None
            node = node.children[c]
        return node

    def _get_descendants(self, node: Node, current: str, result: str):
        """Rekurzivno skuplja sve reci ispod datog cvora."""

        if node.end:
            result.append(current)

        for c, child in node.children.items():
            self._get_descendants(child, current + c, result)

    def find_words_with_prefix(self, prefix: str):
        # Javna metoda - spaja dva koraka.

        # 1) Spusti se do cvora na kraju prefiksa.
        node = self._find_node(prefix)

        # Ako takav cvor ne postoji, nema ni rezultata.
        if node is None:
            return []

        # 2) Pokupi sve reci ispod njega.
        result = []
        self._get_descendants(node, prefix, result)
        return result