"""Resolve catalog artists to Wikipedia articles and save a two-column artifact."""

import json
import re
import time
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from mrvectorbeat.crawler.crawler import Crawler


API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_SEED_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "seed.parquet"
_MUSIC_ENTITY = re.compile(
    r"\b(?:bands?|musicians?|singers?|songwriters?|rappers?|composers?|"
    r"guitarists?|pianists?|vocalists?|drummers?|bassists?|saxophonists?|"
    r"violinists?|conductors?|orchestras?|duos?|trios?|quartets?|quintets?|"
    r"sextets?|djs?|deejays?|disc jockeys?|record producers?|recording artists?|(?:boy|girl) groups?|"
    r"(?:musical|music|pop|rock|dance|hip.hop|electronic).*? (?:groups?|acts?|projects?|collectives?))\b",
    re.IGNORECASE,
)
_OTHER_TOPIC = re.compile(
    r"\b(?:album|single|song|novel|television series|name list|surname|"
    r"given name|city|town|village|continent|river|municipality|"
    r"fictional character|video game)\b", re.IGNORECASE,
)
_VERSION_WORDS = r"\b(?:remix(?:ed)?|remaster(?:ed)?|live|version|deluxe|edition|expanded)\b"
_VERSION = re.compile(r"\s+[-–—]\s+.*" + _VERSION_WORDS + r".*$", re.I)
_BRACKETS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold().replace("&", " and "))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", text))


def _catalog_titles(values: Iterable[str]) -> set[str]:
    """Use distinctive catalog titles as evidence, excluding generic collections."""
    generic = {"greatest hits", "the greatest hits", "best of", "the best of",
               "singles", "unknown", "nan", "none"}
    titles = set()
    for value in values:
        if pd.isna(value):
            continue
        cleaned = _BRACKETS.sub(
            lambda match: " " if re.search(_VERSION_WORDS, match.group(), re.I) else match.group(),
            str(value),
        )
        title = _normalize(_VERSION.sub("", cleaned))
        if len(title) >= 6 and title not in generic:
            titles.add(title)
    return titles


class WikipediaSeedResolver:
    """Find music entities by name and disambiguate them using catalog titles.

    HTTP uses Crawler.fetch_page, including its User-Agent and configured request
    interval. Network/API failures raise instead of being saved as missing artists.
    A genuine missing or ambiguous match returns None.
    """

    def __init__(self):
        self._full_text: dict[int, str] = {}

    def _query(self, **parameters) -> dict:
        # Use the programmatic Action API under Wikimedia's API usage policy.
        # Article crawling and its robots.txt checks remain in focused_crawl.
        url = API_URL + "?" + urlencode({
            "action": "query", "format": "json", "formatversion": 2,
            "maxlag": 5, **parameters,
        })
        for attempt in range(3):
            # API responses may report maxlag/ratelimited even with HTTP 200.
            # Cache only downloaded pages, not these dynamic query responses.
            status, text = Crawler.fetch_page(url, use_cache=False)
            if status == 200 and text is not None:
                try:
                    data = json.loads(text)
                except (ValueError, TypeError) as error:
                    raise RuntimeError("Wikipedia returned an invalid JSON response.") from error
                error = data.get("error")
                if error is None:
                    return data.get("query", {})
                if error.get("code") not in {"maxlag", "ratelimited"}:
                    raise RuntimeError(f"Wikipedia API error: {error}")
            elif status not in {None, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"Wikipedia request failed (status={status}).")
            if attempt < 2:
                # Back off for throttling; the shared HTTP interval still applies.
                time.sleep(60 if status == 429 else 5 * (attempt + 1))
        raise RuntimeError("Wikipedia is unavailable after three attempts; retry preparation later.")

    def _pages(self, titles: list[str]) -> list[dict]:
        query = self._query(
            titles="\x1f" + "\x1f".join(titles), redirects=1,
            prop="info|pageprops|description|extracts",
            inprop="url", exintro=1, explaintext=1, exlimit=10,
        )
        pages = query.get("pages", [])
        by_title = {page["title"]: page for page in pages}
        redirects = {entry["from"]: entry["to"]
                     for kind in ("normalized", "converted", "redirects")
                     for entry in query.get(kind, [])}
        for requested in titles:
            title = requested
            chain = []
            while title in redirects and title not in chain:
                chain.append(title)
                title = redirects[title]
            page = by_title.get(title)
            if page is not None:
                # These names are verified by MediaWiki, not inferred from any
                # mention in the article. Keep aliases attached to their target.
                aliases = page.setdefault("title_aliases", [])
                aliases.extend(alias for alias in chain if alias not in aliases)
        return pages

    @staticmethod
    def _matches_artist_name(page: dict, artist: str) -> bool:
        """Match a canonical title or an API-confirmed title/redirect alias."""
        name = _normalize(artist)
        titles = [page["title"], *page.get("title_aliases", [])]
        return bool(name) and any(_normalize(title) == name for title in titles)

    def _search(self, query: str, *, exact_title: str | None = None) -> list[dict]:
        direct = self._pages([exact_title]) if exact_title else []
        # A direct title lookup also finds articles missing from search results.
        if any(self._candidate(page, exact_title)
               and self._matches_artist_name(page, exact_title) for page in direct):
            return direct
        results = self._query(list="search", srsearch=query, srnamespace=0,
                              srlimit=10).get("search", [])
        if not results:
            return direct
        return direct + self._pages([result["title"] for result in results])

    @staticmethod
    def _candidate(page: dict, artist: str) -> bool:
        if (page.get("ns") != 0 or "missing" in page or "invalid" in page
                or "disambiguation" in page.get("pageprops", {})):
            return False
        if re.match(r"lists? of\b", page["title"], re.I):
            return False
        description = page.get("description") or page.get("pageprops", {}).get("wikibase-shortdesc", "")
        if _OTHER_TOPIC.search(description):
            return False
        # Only the subject's introduction establishes that this is an artist.
        # Music mentioned elsewhere on a city/album page is not enough.
        introduction = page.get("extract", "").strip().split("\n", 1)[0]
        if not _MUSIC_ENTITY.search(description + " " + introduction[:500]):
            return False
        if WikipediaSeedResolver._matches_artist_name(page, artist):
            return True
        name = _normalize(artist)
        title = _normalize(re.sub(r"\s*\([^)]*\)\s*$", "", page["title"]))
        if name == title:
            return True
        # Require an explicit alias: a name merely mentioned in a biography, or
        # the word Rainbow in Rainbow Kitten Surprise, is not the same identity.
        lead_name = _normalize(introduction[:300])
        alias = r"\b(?:known (?:professionally )?as|stylized as|abbreviated as) "
        return bool(name and re.search(alias + re.escape(name) + r"(?: |$)", lead_name))

    def _choose(self, artist: str, pages: list[dict], albums: set[str],
                songs: set[str]) -> str | None:
        candidates = {page["pageid"]: page for page in pages if self._candidate(page, artist)}
        if not candidates:
            return None
        # An exact title or verified redirect describing a musician is sufficient
        # for a single candidate. This covers stage names and stylized spellings
        # such as A$AP Mob -> ASAP Mob without guessing character substitutions.
        # Wikipedia discographies need not contain every catalog song.
        exact_match = (len(candidates) == 1 and
                       self._matches_artist_name(next(iter(candidates.values())), artist))
        if exact_match:
            page = next(iter(candidates.values()))
        elif albums or songs:
            # Qualified titles, aliases, or multiple artists with the same name
            # still need evidence from the catalog to select a clear match.
            scores = []
            for page_id in candidates:
                if page_id not in self._full_text:
                    # TextExtracts permits only one full (non-intro) extract per call.
                    full = self._query(pageids=page_id, prop="extracts",
                                       explaintext=1, exlimit=1)
                    pages = full.get("pages", [])
                    self._full_text[page_id] = pages[0].get("extract", "") if pages else ""
                text = " " + _normalize(self._full_text[page_id]) + " "
                score = (4 * sum(f" {title} " in text for title in albums)
                         + 3 * sum(f" {title} " in text for title in songs))
                scores.append((score, page_id))
            scores.sort(reverse=True)
            if not scores or scores[0][0] == 0 or (len(scores) > 1 and scores[0][0] == scores[1][0]):
                return None
            page = candidates[scores[0][1]]
        elif len(candidates) > 1:
            return None
        else:
            page = next(iter(candidates.values()))
        url = page.get("canonicalurl") or page.get("fullurl")
        if not url or not Crawler.is_article(url):
            return None
        return Crawler._normalize(url)

    def resolve_artist_page(self, artist_name: str, *, albums: Iterable[str] = (),
                            songs: Iterable[str] = ()) -> str | None:
        """Return a canonical artist article URL, or None when no clear match exists."""
        if not isinstance(artist_name, str) or not artist_name.strip():
            raise ValueError("artist_name must be a non-empty string.")
        # Strip title-forbidden brackets, but retain the original name in the artifact.
        query = re.sub(r"[\[\]{}|<>]", " ", artist_name).strip()
        album_titles, song_titles = _catalog_titles(albums), _catalog_titles(songs)
        pages = self._search(query, exact_title=artist_name.strip())
        seed = self._choose(artist_name, pages, album_titles, song_titles)
        if seed is not None:
            return seed
        # Restrict title matches while looking for music subjects, so a common
        # name is not crowded out by unrelated results in the first results.
        title_query = query.replace('"', " ")
        pages += self._search(f'intitle:"{title_query}" (band OR musician OR singer OR rapper OR DJ)')
        seed = self._choose(artist_name, pages, album_titles, song_titles)
        if seed is not None:
            return seed
        evidence = sorted(song_titles or album_titles, key=lambda title: (-len(title), title))
        if evidence:
            # A catalog title helps recover artists outside the first search results.
            extra = self._search(query + ' "' + evidence[0] + '"')
            pages += extra
            seed = self._choose(artist_name, pages, album_titles, song_titles)
            if seed is not None:
                return seed
        return None


def generate_seed(catalog: pd.DataFrame, path: Path = DEFAULT_SEED_PATH, *,
                  existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resolve every distinct catalog artist and atomically save artist_name, seed.

    Unresolved names remain in the artifact with a null seed. No guessed URL is
    substituted. Preparation aborts on HTTP/API failures without replacing a
    previous artifact. Pass existing to keep previously resolved URLs and retry
    only unresolved/new artists.
    """
    resolver = WikipediaSeedResolver()
    previous = {} if existing is None else dict(existing.itertuples(index=False, name=None))
    rows = []
    artists = catalog.dropna(subset=["track_artist"]).groupby("track_artist", sort=True)
    total = artists.ngroups
    for index, (artist, tracks) in enumerate(artists, 1):
        seed = previous.get(artist)
        if not isinstance(seed, str) or not seed.strip():
            seed = resolver.resolve_artist_page(
                artist, albums=tracks.get("track_album_name", []), songs=tracks.get("track_name", []),
            )
        rows.append({"artist_name": artist, "seed": seed})
        print(f"Seed {index}/{total}: {artist} -> {seed or 'unresolved'}", flush=True)
    table = pd.DataFrame(rows, columns=["artist_name", "seed"]).astype("string")
    path = Path(path)
    if existing is not None and path.exists() and table.equals(existing.astype("string")):
        return table
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)
    return table
