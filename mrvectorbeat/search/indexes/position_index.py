from collections import Counter, defaultdict
import math


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

        self.directory = defaultdict(dict)    # termin -> {doc_id: broj pojavljivanja}
        self.doc_terms = defaultdict(set)     # doc_id -> skup termina

        documents = sorted(zip(document_ids, tokens), key=lambda item: item[0])
        self.document_ids = [doc_id for doc_id, _ in documents]
        if any(a == b for a, b in zip(self.document_ids, self.document_ids[1:])):
            raise ValueError("ID-evi dokumenata moraju biti jedinstveni.")

        
        indeks = defaultdict(dict)
        for doc_id, tokens in documents:
            for pos, token in enumerate(tokens):
                indeks[token].setdefault(doc_id, []).append(pos)
                self.doc_terms[doc_id].add(token)
                self.directory[token][doc_id] = self.directory[token].get(doc_id, 0) + 1
        
        self.index: dict[str, dict[int, list[int]]] = dict(indeks)

        # Sve sto zavisi samo od korpusa gradi se i snima zajedno sa indeksom.
        n = len(self.document_ids)
        self.idf_values = {term: math.log10(n / len(posting))
                           for term, posting in self.directory.items()}
        self.document_weights: dict[int, dict[str, float]] = {
            doc_id: {} for doc_id in self.document_ids
        }
        for term, posting in self.directory.items():
            for doc_id, frequency in posting.items():
                self.document_weights[doc_id][term] = 1 + math.log10(frequency)
        self.document_norms = {
            doc_id: math.sqrt(sum(weight * weight for weight in weights.values()))
            for doc_id, weights in self.document_weights.items()
        }
        # Za slicne pesme izabrani dokument sluzi kao ltc upit.
        self.document_query_norms = {
            doc_id: math.sqrt(sum((weight * self.idf_values[term]) ** 2
                                  for term, weight in weights.items()))
            for doc_id, weights in self.document_weights.items()
        }
        self.ranking_version = 1
 
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

    def tf(self, term: str, doc_id: int) -> float:
        broj = self.directory.get(term, {}).get(doc_id)

        if not broj:
            return 0.0

        return 1 + math.log10(broj)

    def df(self, term : str) -> int:
        return len(self.directory.get(term,{}))

    def idf(self, term : str) -> float:
        return self.idf_values.get(term, 0.0)

    def tf_idf(self, term: str, doc_id: int) -> float:
        return self.document_weights.get(doc_id, {}).get(term, 0.0) * self.idf(term)

    def score_tf_idf(self, tokens: list[str]) -> dict[int, float]:
        """Suma log-TF * IDF tezina za svaki token upita (ponavljanja se broje)."""
        return self.score_ranked(tokens, measure="tfidf")

    def score_ranked(self, tokens: list[str], measure: str = "tfidf") -> dict[int, float]:
        """Ocenjuje samo uniju postings lista, bez obilaska celog kataloga."""
        return self._score_counts(Counter(tokens), measure)

    def _score_counts(self, counts: dict[str, int], measure: str) -> dict[int, float]:
        query_norm = 0.0
        if measure == "cosine":
            weights = {term: (1 + math.log10(count)) * self.idf(term)
                       for term, count in counts.items()}
            query_norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        else:
            weights = {term: count * self.idf(term) for term, count in counts.items()}
        return self._score_weights(weights, measure, query_norm)

    def _score_weights(self, weights: dict[str, float], measure: str,
                       query_norm: float = 0.0) -> dict[int, float]:
        if measure not in ("tfidf", "cosine"):
            raise ValueError("Mera mora biti tfidf ili cosine.")
        scores: dict[int, float] = {}
        for term, query_weight in weights.items():
            for doc_id in self.index.get(term, {}):
                weight = self.document_weights[doc_id][term]
                scores[doc_id] = scores.get(doc_id, 0.0) + query_weight * weight
        if measure == "cosine":
            for doc_id in scores:
                denominator = query_norm * self.document_norms[doc_id]
                scores[doc_id] = scores[doc_id] / denominator if denominator else 0.0
        return scores

    def score_similar(self, doc_id: int, measure: str = "cosine") -> dict[int, float]:
        """Izabrana pesma je upit; kandidati su samo pesme sa zajednickim recima."""
        if measure == "cosine":
            weights = {term: weight * self.idf(term)
                       for term, weight in self.document_weights[doc_id].items()}
        else:
            weights = {term: self.directory[term][doc_id] * self.idf(term)
                       for term in self.document_weights[doc_id]}
        scores = self._score_weights(weights, measure, self.document_query_norms[doc_id])
        scores.pop(doc_id, None)
        return scores

    def build_vec(self, tokens: list[str], use_idf: bool) -> tuple[dict[str, float], float]:
        """Vraca tezine po terminu i normu vektora."""
        tezine = {}
        for term, broj in Counter(tokens).items():
            w = 1 + math.log10(broj)
            if use_idf:
                w *= self.idf(term)
            tezine[term] = w

        norma = math.sqrt(sum(w * w for w in tezine.values()))
        return tezine, norma

    def cosine_lnc_ltc(self, query_tokens: list[str], doc_id: int) -> float:
        """Upit ltc i indeksirana pesma lnc, sa sacuvanom normom pesme."""
        wq, nq = self.build_vec(query_tokens, use_idf=True)     # ltc - upit ima idf
        wd = self.document_weights[doc_id]
        nd = self.document_norms[doc_id]

        if nq == 0 or nd == 0:
            return 0.0

        # skalarni proizvod samo po zajednickim terminima
        suma = sum(wq[t] * wd[t] for t in set(wq) & set(wd))
        return suma / (nq * nd)
