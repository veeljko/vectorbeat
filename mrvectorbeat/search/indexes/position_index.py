class PositionIndex:
    def __init__(self, tokens: list[list[str]]):
        """Svaka unutrasnja lista sadrzi obradjene tokene jednog dokumenta.

        ID dokumenta je njegov indeks u ulaznoj listi, pocevsi od nule.
        Postings su sortirane liste parova (doc_id, lista pozicija).
        """
        self.index: dict[str, list[tuple[int, list[int]]]] = {}
        for doc_id, document in enumerate(tokens):
            for pos, term in enumerate(document):
                postings = self.index.setdefault(term, [])
                if not postings or postings[-1][0] != doc_id:
                    postings.append((doc_id, []))
                postings[-1][1].append(pos)

    def find_phrase(self, words: list[str]) -> list[tuple[int, int]]:
        """Vrati sve (doc_id, pocetna_pozicija) pogotke obradjene fraze."""
        if not words:
            return []
        postings = [self.index.get(word, []) for word in words]
        candidates = [[doc_id for doc_id, _ in posting] for posting in postings]
        candidates.sort(key=len)
        documents = candidates[0]
        for candidate in candidates[1:]:
            documents = self.intersection(documents, candidate)
            if not documents:
                return []
        
        result: list[tuple[int, int]] = []
        cursors = [0] * len(words)
        for doc_id in documents:
            positions: list[list[int]] = []
            for i, posting in enumerate(postings):
                while posting[cursors[i]][0] < doc_id:
                    cursors[i] += 1
                positions.append(posting[cursors[i]][1])

            starts = positions[0]
            for offset, following in enumerate(positions[1:], start=1):
                # Uslov p + offset u following jednak je p u pomerenoj listi.
                # Npr. za drugu rec [3, 7] moguci poceci su [2, 6].
                shifted = [position - offset for position in following]
                starts = self.intersection(starts, shifted)
                if not starts:
                    break
            result.extend((doc_id, start) for start in starts)
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
        first = self.index.get(a, [])
        second = self.index.get(b, [])
        documents = self.intersection(
            [doc_id for doc_id, _ in first],
            [doc_id for doc_id, _ in second],
        )
        result: list[tuple[int, int, int]] = []
        first_cursor = second_cursor = 0
        for doc_id in documents:
            while first[first_cursor][0] < doc_id:
                first_cursor += 1
            while second[second_cursor][0] < doc_id:
                second_cursor += 1
            positions_a = first[first_cursor][1]
            positions_b = second[second_cursor][1]

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
