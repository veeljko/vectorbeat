import string
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag

# Ove reci cuvamo jer mogu menjati znacenje stiha ili naslova.
KEEP_WORDS = frozenset({
    "no", "not", "nor", "don't", "won't", "can't", "isn't", "aren't",
    "wasn't", "weren't", "didn't", "doesn't", "wouldn't", "couldn't",
    "shouldn't", "haven't", "hasn't", "hadn't", "mustn't", "needn't",
    "mightn't", "shan't", "up", "down", "out", "off", "on", "over", "under",
})
stop_words = stopwords.words("english")
stop_words = set([word for word in stop_words if word not in KEEP_WORDS])

class TextProcessor:
    _stem_cache = dict()
    _lem = WordNetLemmatizer()
    _stemmer = SnowballStemmer("english")

    def __init__(self, text: str | list[str]):
        self.text = text.copy() if isinstance(text, list) else text

    def to_lower(self, words: str | list[str]) -> str | list[str]:
        if isinstance(words, str):
            return words.lower()
        return [word.lower() for word in words]

    def remove_spaces(self, words: str | list[str]) -> str | list[str]:
        """Svedi uzastopne beline na jedan razmak"""
        if isinstance(words, str):
            return " ".join(words.split())
        return [" ".join(word.split()) for word in words]

    def remove_apostrophes(self, words: str | list[str]) -> str | list[str]:
        characters_to_remove = set("'`‘’ʼ")
        if isinstance(words, str):
            return "".join(char for char in words if char not in characters_to_remove)
        return [self.remove_apostrophes(word) for word in words]

    def remove_punctuation(self, words: str | list[str]) -> str | list[str]:
        """Kosu crtu i crtice zameni razmakom, ostalu interpunkciju obrisi."""
        separators = set(string.punctuation + "‐‑–—…“”‘’«»‹›")
        if isinstance(words, str):
            return "".join(
                " " if char in "/-‐‑–—"
                else "" if char in separators
                else char
                for char in words
            )
        return [self.remove_punctuation(word) for word in words]

    def remove_apostrophes_and_hyphens(self, words: str | list[str]) -> str | list[str]:
        """Kompatibilni poziv: apostrofe brise, crtice menja razmakom."""
        words = self.remove_apostrophes(words)
        if isinstance(words, str):
            return "".join(
                " " if char in "-‐‑–—" 
                else char 
                for char in words
            )
        return [self.remove_apostrophes_and_hyphens(word) for word in words]

    def remove_non_words(self, words: str | list[str]) -> str | list[str]:
        """Izdvoji alfanumericke delove bez gubljenja reci uz simbole."""
        if isinstance(words, str):
            return " ".join(
                "".join(char if char.isalnum() else " " for char in words).split()
            )
        return [
            token
            for word in words
            for token in self.remove_non_words(word).split()
        ]

    def remove_stop_words(self, tokens: list[str], stop_words: list[str] | set[str]) -> list[str]:
        excluded = set(self.extract_words(self.to_lower(list(stop_words))))
        return [word for word in tokens
                if self.remove_apostrophes(word.lower()) not in excluded]

    def word_stem(self, tokens: list[str]) -> list[str]:

        result = []
        for word in tokens:
            if word not in self._stem_cache:
                self._stem_cache[word] = self._stemmer.stem(word)
            result.append(self._stem_cache[word])
        return result

    def word_lem(self, tokens: list[str]) -> list[str]:
        def get_word_type(tag: str) -> str:
            if tag.startswith("V"):
                return "v"
            if tag.startswith("J"):
                return "a"
            if tag.startswith("R"):
                return "r"
            return "n"
        
        result = [self._lem.lemmatize(word, pos=get_word_type(tag))
                  for word, tag in pos_tag(tokens)]
        return result

    def extract_words(self, text: str | list[str]) -> list[str]:
        """Normalizuj apostrofe i separatore i izdvoji reci."""
        text = self.remove_apostrophes_and_hyphens(text)
        text = self.remove_punctuation(text)
        text = self.remove_spaces(text)
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = [token for part in text for token in part.split()]
        return self.remove_non_words(tokens)

    def text_tokenize(self, min_len: int = 1, method: str = "stem") -> list[str]:
        """Obradi self.text koristeci parametre ovog poziva."""
        global stop_words
        method = method.lower()
        if method not in {"stem", "lemma"}:
            raise ValueError("method mora biti 'stem' ili 'lemma'")
        if isinstance(min_len, bool) or not isinstance(min_len, int) or min_len < 1:
            raise ValueError("min_len mora biti pozitivan ceo broj")
        
        text = self.to_lower(self.text)
        tokens = self.extract_words(text)
        tokens = self.remove_stop_words(tokens, stop_words)

        tokens = [token for token in tokens if len(token) >= min_len]
        if method == "stem":
            return self.word_stem(tokens)
        return self.word_lem(tokens)

    def tokenize_with_spans(self) -> tuple[list[str], list[tuple[int, int]]]:
        """Default stem tokeni i [pocetak, kraj) svakog tokena u originalu.

        Pozicije indeksa broje obradjene tokene, pa za isecak moramo sacuvati
        vezu sa originalom, ukljucujuci izbacene stop-reci i interpunkciju.
        """
        if not isinstance(self.text, str):
            raise ValueError("Za pozicije u originalu potreban je string.")
        lower_offsets: list[int] = []
        for offset, char in enumerate(self.text):
            lower_offsets.extend([offset] * len(char.lower()))
        normalized: list[str] = []
        offsets: list[int] = []
        translations: dict[str, str] = {}
        for char, offset in zip(self.text.lower(), lower_offsets):
            if char not in translations:
                cleaned = self.remove_punctuation(self.remove_apostrophes_and_hyphens(char))
                translations[char] = "".join(c if c.isalnum() else " " for c in cleaned)
            for replacement in translations[char]:
                normalized.append(replacement)
                offsets.append(offset)
        excluded = set(self.extract_words(self.to_lower(list(stop_words))))
        words: list[str] = []
        spans: list[tuple[int, int]] = []
        for match in re.finditer(r"\S+", "".join(normalized)):
            if match.group() not in excluded:
                words.append(match.group())
                spans.append((offsets[match.start()], offsets[match.end() - 1] + 1))
        return self.word_stem(words), spans
