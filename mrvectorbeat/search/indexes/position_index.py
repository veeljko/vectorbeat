from collections import defaultdict


class PositionIndex:
    def __init__(self, tokens: list[list[str]], document_ids: list[int] | None = None):
        """Svaka unutrasnja lista sadrzi obradjene tokene jednog dokumenta.

        ID dokumenta je katalog_id, ili indeks u ulazu ako ID-evi nisu dati.
        Indeks je recnik: term -> {doc_id: lista pozicija}.
        """
        if document_ids is None:
            document_ids = list(range(len(tokens)))
        if len(document_ids) != len(tokens):
            raise ValueError("Svaki dokument mora imati svoj ID.")


        documents = sorted(zip(document_ids, tokens), key=lambda item: item[0])
        self.document_ids = [doc_id for doc_id, _ in documents]
        if any(a == b for a, b in zip(self.document_ids, self.document_ids[1:])):
            raise ValueError("ID-evi dokumenata moraju biti jedinstveni.")

        
        indeks = defaultdict(dict)
        for doc_id, tokens in documents:
            for pos, token in enumerate(tokens):
                indeks[token].setdefault(doc_id, []).append(pos)
        
        self.index: dict[str, dict[int, list[int]]] = dict(indeks)
 
    def get_documents(self, word: str) -> list[int]:
        #Vraca ID-jeve dokumenata gde se nalazi word
        return list(self.index.get(word, {}))

    def find_boolean(self, words: list[str], operator: str = "AND") -> list[int]:
        """AND: sve reci; OR: bar jedna; DIFFERENCE: prva bez svih ostalih."""
        operator = operator.upper()
        if operator not in ("AND", "OR", "DIFFERENCE"):
            raise ValueError("Operator mora biti AND, OR ili DIFFERENCE.")
        if not words:
            return []
        candidates = [self.get_documents(word) for word in words]
        if operator == "AND":
            candidates.sort(key=len)
        result = candidates[0]
        operation = {"AND": self.intersection, 
                     "OR": self.union,
                     "DIFFERENCE": self.difference
        }[operator]
                     
        for candidate in candidates[1:]:
            if not result and operator != "OR":
                break
            result = operation(result, candidate)
        return result

    def find_phrase(self, words: list[str]) -> list[tuple[int, int]]:
        """Vrati sve (doc_id, pocetna_pozicija) pogotke obradjene fraze."""
        if not words:
            return []

        
        postings = [self.index.get(word, {}) for word in words]
        candidates = [list(posting) for posting in postings]
        candidates.sort(key=len)

        documents = candidates[0]
        for candidate in candidates[1:]:
            if not documents:
                return []
            documents = self.intersection(documents, candidate)
            if not documents:
                return []
        
        result: list[tuple[int, int]] = []
        for doc_id in documents:
            positions = [self.index[word][doc_id] for word in words]

            for p in positions[0]:
                if all(p + i in positions[i] for i in range(1, len(words))):
                    result.append((doc_id, p))
        
        return result

    def find_proximity(self, a: str, b: str, distance: int) -> list[tuple[int, int, int]]:
        """Vrati (doc_id, pozicija_a, pozicija_b) za |pozicija_a - pozicija_b| < distance.

        Redosled reci nije bitan. Za istu rec vracamo svaki par razlicitih
        pojavljivanja jednom, sa manjom pozicijom prvom. Pozicije su u tokenima.
        """
        if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
            raise ValueError("distance mora biti nenegativan ceo broj")
        if not a or not b or distance <= 1:
            return []

        
        first = self.index.get(a, {})
        second = self.index.get(b, {})
        documents = self.intersection(
            list(first),
            list(second),
        )
        result: list[tuple[int, int, int]] = []
        for doc_id in documents:
            positions_a = first[doc_id]
            positions_b = second[doc_id]

            # Klizni prozor u sortiranoj listi pozicija druge reci.
            left = right = 0
            for px in positions_a:
                while left < len(positions_b) and positions_b[left] <= px - distance:
                    left += 1
                right = max(right, left)
                while right < len(positions_b) and positions_b[right] < px + distance:
                    right += 1
                for i in range(left, right):
                    py = positions_b[i]
                    if a != b or px < py:
                        result.append((doc_id, px, py))
        return result

    def intersection(self, l1: list[int], l2:list[int]) -> list[int]:
        p1 = 0
        p2 = 0
        res = []
        while p1 < len(l1) and p2 < len(l2):
            if l1[p1] == l2[p2]:
                res.append(l1[p1])
                p1+=1
                p2+=1
            elif l1[p1] < l2[p2]:
                p1+=1
            else:
                p2+=1
        return res

    def union(self, l1: list[int], l2:list[int]) -> list[int]:
        p1 = 0
        p2 = 0
        res = []
        while p2 < len(l2):
            while p1 < len(l1) and l1[p1] < l2[p2]:
                res.append(l1[p1])
                p1+=1
            if p1 < len(l1) and l1[p1] == l2[p2]:
                res.append(l1[p1])
                p1+=1
            else:
                res.append(l2[p2])
            p2+=1
        while p1 < len(l1):
            res.append(l1[p1])
            p1+=1

        return res

    def difference(self, l1: list[int], l2: list[int]) -> list[int]:
        p1 = 0
        p2 = 0
        l2 = self.intersection(l1, l2)
        res = []
        while p2 < len(l2):
            while p1 < len(l1) and l1[p1] < l2[p2]:
                res.append(l1[p1])
                p1+=1
            if p1 < len(l1) and l1[p1] == l2[p2]:
                p1+=1
            p2+=1
        while p1 < len(l1):
            res.append(l1[p1])
            p1+=1

        return res
