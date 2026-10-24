from collections import defaultdict
import re
from collections.abc import Iterable

class KGram:
    def __init__(self, dictionary: Iterable[str], k: int = 3):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k mora biti pozitivan ceo broj")
        self.index: dict[str, set[str]] = defaultdict(set)
        self.k = k
        self.originals: dict[str, str] = {}
        self.word_grams: dict[str, set[str]] = {}
        for word in dictionary:
            if not isinstance(word, str) or not word.strip():
                continue
            key = word.casefold()
            if key in self.originals:
                continue
            self.originals[key] = word
            grams = set(self.calc_kgram(key))
            self.word_grams[key] = grams
            for gram in grams:
                self.index[gram].add(key)

    def calc_kgram(self, word: str) -> list[str]:
        word = "$" + word.casefold() + "$"
        return [word[i:i + self.k] for i in range(len(word) - self.k + 1)]

    def search_kgrams(self, pattern: str) -> set[str] | None:
        """Vraca kandidate - reci koje sadrze sve trazene k-grame."""

        # 1) SECEMO UPIT NA DELOVE KOJE ZNAMO
        parts = pattern.casefold().split("*")

        # 2) OD SVAKOG DELA PRAVIMO K-GRAME
        required_grams = []
        for i, part in enumerate(parts):      # i = redni broj dela (0, 1, 2...)

            # Prazan deo (od zvezdice na pocetku/kraju) nema slova -> nema grama.
            if not part:
                continue

            # $ znaci "ivica reci" (pocetak ili kraj)
            marked_part = part
            if i == 0:
                marked_part = "$" + part
            if i == len(parts) - 1:
                marked_part = marked_part + "$"

            # Oba if-a mogu da vaze odjednom: upit bez zvezdice "beat" -> "$beat$".

            required_grams += [marked_part[j:j + self.k] for j in range(len(marked_part) - self.k + 1)]

        # 3) AKO NEMAMO NIJEDAN GRAM, INDEKS NE MOZE DA POMOGNE
        # Desava se kad su svi delovi kraci od k. Vracamo None, a NE praznu listu:
        #   []   bi znacilo "nijedna rec ne odgovara"  (to bi bilo pogresno!)
        #   None znaci "ne znam " (pozivalac onda skenira ceo recnik)
        if not required_grams:
            return None

        # 4) PRESEK: rec mora da ima SVE trazene grame

        postings = sorted(
            (self.index.get(gram, set()) for gram in set(required_grams)), key=len
        )
        # Kopija cuva originalni indeks od promena tokom presecanja.
        candidates = postings[0].copy()
        for posting in postings[1:]:
            if not candidates:
                break
            candidates.intersection_update(posting)
        return candidates

    def wildcard_pretraga(self, pattern: str) -> list[str]:
        candidates = self.search_kgrams(pattern)
        if candidates is None:
            candidates = self.originals.keys()
        # Samo * je specijalan znak; ? i uglaste zagrade su deo imena.
        expression = ".*".join(re.escape(part) for part in pattern.casefold().split("*"))
        matcher = re.compile(expression, re.DOTALL)
        return [self.originals[word] for word in sorted(candidates)
                if matcher.fullmatch(word)]

    def correct(self, query: str, threshold: float = 0.35) -> tuple[str, float] | None:
        """Najblize ime po Zakarovoj slicnosti, samo iz unije postings skupova.

        Prag 0.35 zahteva vise od 35% zajednickih grama u uniji. To je
        pocetna kompromisna vrednost: tolerise tipfelere, odbacuje slaba
        poklapanja. Treba je podesavati na stvarnim upitima i ocenama.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Prag mora biti izmedju 0 i 1")
        if not query or "*" in query:
            return None
        query_grams = set(self.calc_kgram(query))
        candidates: set[str] = set()
        for gram in query_grams:
            candidates.update(self.index.get(gram, set()))
        best_word: str | None = None
        best_score = threshold
        for word in sorted(candidates):
            grams = self.word_grams[word]
            score = len(query_grams & grams) / len(query_grams | grams)
            if score > best_score:
                best_word, best_score = word, score
        if best_word is None:
            return None
        return self.originals[best_word], best_score
