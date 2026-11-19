from collections.abc import Hashable, Iterable, Mapping
from contextlib import contextmanager
from hashlib import sha256
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from sys import intern
from tempfile import NamedTemporaryFile
from threading import Lock, RLock, local
import json
import math
import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup, SoupStrainer
from urllib.parse import quote, unquote, urljoin, urlsplit

from mrvectorbeat.search.ranking.harvest_rate import harvest_rate


WIKI = "https://en.wikipedia.org/wiki/"
PAGE_CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "crawler_pages"
REQUEST_INTERVAL_SECONDS = 1.0
NON_ARTICLE_PREFIXES = (
    "Special:", "Media:", "File:", "File_talk:", "Image:", "Image_talk:",
    "Category:", "Category_talk:", "Help:", "Help_talk:", "Talk:",
    "Template:", "Template_talk:", "Wikipedia:", "Wikipedia_talk:",
    "Project:", "Project_talk:", "WP:", "WT:", "MediaWiki:", "MediaWiki_talk:",
    "Portal:", "Portal_talk:", "User:", "User_talk:", "Draft:", "Draft_talk:",
    "Module:", "Module_talk:", "Book:", "Book_talk:", "Main_Page",
)
KEYWORDS = [
    "music", "musical", "musician", "singer", "songwriter", "album", "concert",
    "composer", "guitar", "orchestra", "band", "vocalist", "rapper", "drummer",
    "bassist", "guitarist", "pianist", "keyboardist", "violinist", "saxophonist",
    "trumpeter", "conductor", "instrumentalist", "discography", "song",
    "soundtrack", "symphony", "opera", "choir", "melody", "rhythm", "percussion",
    "record label", "record producer", "recording studio", "music video",
    "music festival", "recording artist", "jazz", "blues", "reggae", "hip hop",
    "hip-hop", "heavy metal", "punk rock", "rock music", "pop music",
    "electronic music", "classical music", "country music", "rhythm and blues",
]

class Crawler:
    # Protect metadata separately from the serialized HTTP transport.
    _request_lock = RLock()
    _cache_file_lock = RLock()
    _http_lock = Lock()
    _page_locks = {}
    _sessions = local()
    _last_request_finished: float | None = None
    _alias_cache_signature = None
    _alias_cache = None

    @staticmethod
    def _get_session() -> requests.Session:
        """Reuse connections within a thread, without sharing mutable sessions."""
        if not hasattr(Crawler._sessions, "session"):
            Crawler._sessions.session = requests.Session()
        return Crawler._sessions.session

    @staticmethod
    @contextmanager
    def _page_lock(url: str):
        """Deduplicate simultaneous fetches of one URL without blocking others."""
        with Crawler._request_lock:
            entry = Crawler._page_locks.setdefault(url, [Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with Crawler._request_lock:
                entry[1] -= 1
                if entry[1] == 0:
                    del Crawler._page_locks[url]

    @staticmethod
    def _http_get(url: str):
        # Serialize every HTTP request, including calls from other instances or
        # threads. Wait from the previous request's completion, not its start.
        with Crawler._http_lock:
            while Crawler._last_request_finished is not None:
                remaining = REQUEST_INTERVAL_SECONDS - (time.monotonic() - Crawler._last_request_finished)
                if remaining <= 0:
                    break
                time.sleep(remaining)
            try:
                return Crawler._get_session().get(
                    url, headers={"User-Agent": "UPI-Crawler/1.0 (contact: vmladenovic1524rn@raf.rs)"},
                    timeout=10, allow_redirects=False,
                )
            finally:
                Crawler._last_request_finished = time.monotonic()

    @staticmethod
    def _cache_path(url: str) -> Path:
        """Use a normalized URL hash as a portable filename."""
        key = sha256(Crawler._normalize(url).encode("utf-8")).hexdigest()
        return PAGE_CACHE_DIR / (key + ".json")

    @staticmethod
    def _cache_file_operation(operation):
        """Serialize file access and retry temporary Windows/OneDrive locks.

        HTTP uses a separate lock. Never delete the old destination to force a
        replacement; persistent permission errors must still reach the caller.
        """
        with Crawler._cache_file_lock:
            for attempt in range(10):
                try:
                    return operation()
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(min(0.05 * 2 ** attempt, 1.0))

    @staticmethod
    def _read_cache_json(path: Path) -> dict:
        """Release the encoded JSON buffer before returning the decoded page."""
        def read():
            with path.open(encoding="utf-8") as source:
                return json.load(source)

        return Crawler._cache_file_operation(read)

    @staticmethod
    def _read_cached_page(url: str) -> tuple[str, str] | None:
        try:
            cached = Crawler._read_cache_json(Crawler._cache_path(url))
        except (FileNotFoundError, UnicodeError, ValueError):
            return None
        if (isinstance(cached, dict) and cached.get("url") == Crawler._normalize(url)
                and isinstance(cached.get("text"), str)):
            return cached["text"], Crawler._cached_final_url(cached)
        return None

    @staticmethod
    def _cached_final_url(cached: dict) -> str:
        final_url = cached.get("final_url")
        if not isinstance(final_url, str):
            final_url = cached["url"]
            # Old cache entries did not store the redirect destination. Wikipedia
            # HTML normally contains its canonical address, allowing offline migration.
            if Crawler.is_article(final_url):
                soup = BeautifulSoup(cached["text"], "html.parser", parse_only=SoupStrainer("link"))
                try:
                    canonical = soup.find("link", rel="canonical", href=True)
                    if canonical is not None:
                        candidate = urljoin(final_url, canonical["href"])
                        if Crawler.is_article(candidate):
                            final_url = candidate
                finally:
                    soup.decompose()
        return Crawler._normalize(final_url)

    @staticmethod
    def _write_cache_json(path: Path, value: dict) -> None:
        """Replace a cache entry atomically, preserving complete UTF-8 content."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                    suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                json.dump(value, output, ensure_ascii=False)
            Crawler._cache_file_operation(lambda: temporary.replace(path))
        finally:
            if temporary is not None:
                try:
                    Crawler._cache_file_operation(lambda: temporary.unlink(missing_ok=True))
                except OSError:
                    # Cleanup must not hide the original write/replace error.
                    # A locked leftover .tmp is never read as a cached page.
                    pass

    @staticmethod
    def _save_cached_page(url: str, text: str, final_url: str | None = None) -> None:
        Crawler._write_cache_json(Crawler._cache_path(url), {
            "url": Crawler._normalize(url), "text": text,
            "final_url": Crawler._normalize(final_url or url),
        })

    @staticmethod
    def _load_alias_index() -> dict:
        """Called under the shared lock; cache the small persistent alias index."""
        path = PAGE_CACHE_DIR / "url_aliases.json"
        stamp = path.stat().st_mtime_ns if path.exists() else None
        signature = (path, stamp)
        if Crawler._alias_cache_signature != signature:
            try:
                state = Crawler._read_cache_json(path)
            except (FileNotFoundError, UnicodeError, ValueError):
                state = {}
            if not isinstance(state, dict) or not isinstance(state.get("aliases"), dict):
                state = {"aliases": {}, "legacy_cache_scanned": False}
            Crawler._alias_cache = state
            Crawler._alias_cache_signature = signature
        return Crawler._alias_cache

    @staticmethod
    def _save_alias_index(state: dict) -> None:
        path = PAGE_CACHE_DIR / "url_aliases.json"
        Crawler._write_cache_json(path, state)
        Crawler._alias_cache = state
        Crawler._alias_cache_signature = (path, path.stat().st_mtime_ns)

    @staticmethod
    def _remember_aliases(aliases: Iterable[str], final_url: str) -> None:
        if not Crawler.is_article(final_url):
            return
        state = Crawler._load_alias_index()
        known = dict(state["aliases"])
        final_url = Crawler._normalize(final_url)
        for alias in aliases:
            alias = Crawler._normalize(alias)
            if alias != final_url and Crawler.is_article(alias):
                known[alias] = final_url
        if known != state["aliases"]:
            Crawler._save_alias_index({**state, "aliases": known})

    @staticmethod
    def canonical_url(url: str, aliases: Mapping[str, str] | None = None) -> str:
        """Resolve already confirmed redirects without any HTTP requests."""
        if aliases is None:
            with Crawler._request_lock:
                aliases = Crawler._load_alias_index()["aliases"]
        url = Crawler._normalize(url)
        seen = set()
        while url in aliases and url not in seen:
            seen.add(url)
            url = Crawler._normalize(aliases[url])
        return url

    @staticmethod
    def migrate_cached_aliases() -> None:
        """Recover redirect identities from old cached HTML once, without fetching."""
        with Crawler._request_lock:
            state = Crawler._load_alias_index()
            if state.get("legacy_cache_scanned") or not PAGE_CACHE_DIR.exists():
                return
            aliases = dict(state["aliases"])
            for path in PAGE_CACHE_DIR.glob("*.json"):
                if path.name == "url_aliases.json":
                    continue
                try:
                    cached = Crawler._read_cache_json(path)
                except (UnicodeError, ValueError):
                    continue
                if (not isinstance(cached, dict) or not isinstance(cached.get("url"), str)
                        or not isinstance(cached.get("text"), str)):
                    del cached
                    continue
                source = Crawler._normalize(cached["url"])
                target = Crawler._cached_final_url(cached)
                if source != target and Crawler.is_article(source) and Crawler.is_article(target):
                    aliases[source] = target
                del cached
            Crawler._save_alias_index({"aliases": aliases, "legacy_cache_scanned": True})

    @staticmethod
    def canonicalize_graph(graph: dict[str, list[str]]) -> bool:
        """Merge confirmed URL aliases in place, including incoming edges.

        Preserve all outgoing edges, remove duplicates and self-links produced
        by merging, and return whether the graph changed.
        """
        with Crawler._request_lock:
            aliases = dict(Crawler._load_alias_index()["aliases"])
        merged = {}
        seen = {}
        for source, targets in graph.items():
            source = Crawler.canonical_url(source, aliases)
            neighbors = merged.setdefault(source, [])
            known = seen.setdefault(source, set())
            for target in targets:
                target = Crawler.canonical_url(target, aliases)
                if target != source and target not in known:
                    neighbors.append(target)
                    known.add(target)
        for targets in tuple(merged.values()):
            for target in targets:
                merged.setdefault(target, [])
        if graph == merged:
            return False
        graph.clear()
        graph.update(merged)
        return True

    @staticmethod
    def fetch_page(url: str, *, use_cache: bool = True, return_url: bool = False
                   ) -> tuple[int | None, str | None] | tuple[int | None, str | None, str]:
        """Return (status, text), reading successful responses from disk first.

        HTTP 200 responses are saved in artifacts/crawler_pages, including
        robots.txt and successful redirect aliases. Failures are not cached.
        return_url=True adds the final URL as a third return value. The default
        two-value interface remains compatible with existing callers.
        Set use_cache=False for dynamic API calls, whose HTTP 200 body may
        contain temporary API errors that must be retried.

        HTTP requests are serialized, with at least REQUEST_INTERVAL_SECONDS
        (one second) from completion of a request to the start of the next.
        The first request starts immediately; no delay follows the last request.
        The interval is shared by all instances and applies to redirects too.
        Request failures print the exception type and return (None, None).
        """
        url = Crawler._normalize(url)
        redirects = 0
        aliases = []
        def result(status, text, final_url):
            return (status, text, final_url) if return_url else (status, text)

        try:
            while True:
                # Hold only this hop's URL lock. Release it before following a
                # redirect, avoiding deadlocks between crossing redirect chains.
                with Crawler._page_lock(url):
                    if use_cache:
                        cached = Crawler._read_cached_page(url)
                        if cached is not None:
                            text, final_url = cached
                            final_url = Crawler.canonical_url(final_url)
                            if final_url != url and not Crawler._cache_path(final_url).exists():
                                Crawler._save_cached_page(final_url, text, final_url)
                            for alias in [url, *aliases]:
                                if alias != final_url:
                                    Crawler._save_cached_page(alias, text, final_url)
                            with Crawler._request_lock:
                                Crawler._remember_aliases([url, *aliases], final_url)
                            return result(200, text, final_url)
                        known_url = Crawler.canonical_url(url)
                        if known_url != url:
                            aliases.append(url)
                            url = known_url
                            continue
                    response = Crawler._http_get(url)
                    status = response.status_code
                    if status in (301, 302, 303, 307, 308) and response.headers.get("Location"):
                        next_url = urljoin(url, response.headers["Location"])
                        response.close()
                        if redirects >= 30:
                            raise requests.TooManyRedirects("Exceeded 30 redirects.")
                        redirects += 1
                        aliases.append(url)
                        url = Crawler._normalize(next_url)
                        continue
                    data = response.text if status == 200 else None
                    response.close()
                    if status == 200 and Crawler.is_article(url):
                        # MediaWiki can render a redirect target with HTTP 200.
                        canonical = Crawler._cached_final_url({"url": url, "text": data})
                        if canonical != url:
                            aliases.append(url)
                            url = canonical
                    if use_cache and status == 200:
                        Crawler._save_cached_page(url, data)
                        for alias in aliases:
                            Crawler._save_cached_page(alias, data, url)
                        with Crawler._request_lock:
                            Crawler._remember_aliases(aliases, url)
                    return result(status, data, url)
        except requests.RequestException as error:
            print("Request failed:", type(error).__name__)
            return result(None, None, url)

    @staticmethod
    def _raw_links(html: str, web: str) -> list[str]:
        """Svi linkovi sa stranice, dopunjeni do apsolutnog URL-a."""
        soup = BeautifulSoup(html, "html.parser", parse_only=SoupStrainer("a", href=True))
        try:
            return [urljoin(web, anchor["href"]) for anchor in soup.find_all("a", href=True)]
        finally:
            soup.decompose()

    @staticmethod
    def _normalize(web: str) -> str:
        """Unify encoded Wikipedia titles and discard article queries and fragments."""
        parsed = urlsplit(web)
        if (parsed.scheme == "https" and parsed.netloc.lower() == "en.wikipedia.org"
                and parsed.path.startswith("/wiki/")):
            title = unicodedata.normalize("NFC", unquote(parsed.path[len("/wiki/"):]))
            title = title.replace(" ", "_")
            return WIKI + quote(title, safe="/():")
        # Query parameters may identify different resources on other websites.
        return web.split("#", 1)[0]


    def get_links(self, html: str, web: str) -> list[str]:
        """Linkovi iz webs[web], bez fragmenata; nepoznata stranica daje praznu listu."""
        links = []
        for link in self._raw_links(html, web):
            links.append(self._normalize(link))
        return links


    @staticmethod
    def get_text(html: str) -> str:
        """Tekst stranice iz recnika URL -> HTML; nepoznat URL daje prazan tekst."""
        soup = BeautifulSoup(html, "html.parser")
        try:
            for symbol in soup(["script", "style"]):
                symbol.decompose()
            return " ".join(soup.get_text(separator=" ", strip=True).split())
        finally:
            soup.decompose()

    @staticmethod
    def parse_robots(text: str) -> list[str]:
        """Return non-empty Disallow path prefixes from User-agent: * groups."""
        blocked_paths = []
        applies_to_all = False
        has_rules = False
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            directive, separator, value = line.partition(":")
            if not separator:
                continue
            directive = directive.strip().lower()
            value = value.strip()
            if directive == "user-agent":
                # Consecutive user-agent lines belong to the same group.
                if has_rules:
                    applies_to_all = False
                    has_rules = False
                applies_to_all = applies_to_all or value == "*"
            elif directive in ("disallow", "allow"):
                has_rules = True
                if directive == "disallow" and applies_to_all and value:
                    blocked_paths.append(value)
        return blocked_paths

    @staticmethod
    def get_path(url: str) -> str:
        """Return the URL path and query, for example /calendar?month=3."""
        parsed = urlsplit(url)
        path = parsed.path or "/"
        return path + "?" + parsed.query if parsed.query else path

    @staticmethod
    def is_allowed(url: str, blocked_paths: Iterable[str]) -> bool:
        """Return False if the path and query start with a blocked prefix."""
        path = Crawler.get_path(url)
        for blocked_path in blocked_paths:
            if path.startswith(blocked_path):
                return False
        return True

    @staticmethod
    def is_article(url: str, current_url: str | None = None) -> bool:
        """Check for an English Wikipedia article, excluding namespaces and self-links."""
        url = Crawler._normalize(url)
        if not url.startswith(WIKI):
            return False
        parsed = urlsplit(url)
        if current_url is not None:
            current = urlsplit(current_url)
            # Fragments and query parameters still refer to the same article.
            if parsed.netloc == current.netloc and unquote(parsed.path) == unquote(current.path):
                return False
        title = unquote(parsed.path[len("/wiki/"):])
        if not title:
            return False
        for prefix in NON_ARTICLE_PREFIXES:
            if title.casefold().startswith(prefix.casefold()):
                return False
        return True

    def get_article_links(self, html: str,  web: str, *,
                          verbose: bool = False) -> list[str]:
        """Return sorted unique article links; optionally print counts and eight titles."""
        all_links = self.get_links(html, web)
        unique_links = set(all_links)
        articles = []
        for url in sorted(unique_links):
            if self.is_article(url, web):
                articles.append(url)
        if verbose:
            print(f"Links in HTML : {len(all_links)}")
            print(f"Unique links  : {len(unique_links)}")
            print(f"Article links : {len(articles)}")
            print()
            for url in articles[:8]:
                print("  ", url[len(WIKI):])
        return articles

    def calculate_score(self, html):
        """Count whole music terms and simple plurals, avoiding substring matches."""
        text = self.get_text(html).lower()
        score = 0
        for keyword in KEYWORDS:
            pattern = r"(?<!\w)" + re.escape(keyword) + r"(?:s|es)?(?!\w)"
            score += len(re.findall(pattern, text))
        return score

    @staticmethod
    def author_seed(author_name: str) -> str:
        """Build a candidate English Wikipedia URL from an artist's catalog name."""
        if not isinstance(author_name, str) or not author_name.strip():
            raise ValueError("author_name must be a non-empty string.")
        title = "_".join(author_name.split())
        return WIKI + quote(title, safe="/()")

    @staticmethod
    def _artist_seeds() -> dict[str, str | None]:
        """Read prepared seeds only; crawling never starts a new artist search."""
        from mrvectorbeat.pipeline import ucitaj_seed

        table = ucitaj_seed()
        return {artist: seed if isinstance(seed, str) and seed else None
                for artist, seed in table.itertuples(index=False, name=None)}

    def focused_crawl(self,
                      author_name: str,
                      max_pages: int = 10,
                      threshold: float = 2, *,
                      seed: str | None = None) -> dict[str, dict[str, str]]:
        """Fetch pages by descending parent score, expanding only on-topic pages.

        calculate_score(html) evaluates each downloaded page.
        Successful off-topic pages are retained but their
        links are not followed. Robots rules are checked before each request.
        Return author_name -> URL -> HTML, with at most max_pages pages.
        Use the supplied seed or read the artist's resolved seed artifact.
        Skip unresolved artists instead of guessing an article URL.
        Print the successful download count and music-related harvest rate.
        """
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 0:
            raise ValueError("max_pages must be a nonnegative integer.")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite.")

        if seed is None and max_pages:
            seed = self._artist_seeds().get(author_name)
        if seed is None or max_pages == 0:
            print(f"No pages to crawl for {author_name}: unresolved seed or zero page limit.")
            print("Downloaded pages: 0")
            print("Music-related share (harvest rate): 0.00%")
            return {author_name: {}}
        if not self.is_article(seed):
            raise ValueError("seed must be an English Wikipedia article URL.")
        sequence = count()
        queue = [(-1000.0, next(sequence), self._normalize(seed))]
        visited = set()
        pages = {}
        relevant_urls = set()

        blocked_paths = []
        if max_pages:
            robots_status, robots_text = self.fetch_page("https://en.wikipedia.org/robots.txt")
            if robots_status != 200 or robots_text is None:
                raise RuntimeError(f"Could not fetch robots.txt (status={robots_status}).")
            blocked_paths = self.parse_robots(robots_text)

        while queue and len(pages) < max_pages:
            _, _, url = heappop(queue)
            url = self.canonical_url(url)
            if url in visited:
                continue
            visited.add(url)
            if not self.is_allowed(url, blocked_paths):
                continue

            # fetch_page returns status first, followed by the response body.
            status, html, final_url = self.fetch_page(url, return_url=True)
            if status != 200 or html is None:
                continue
            final_url = self.canonical_url(final_url)
            visited.add(final_url)
            if not self.is_article(final_url) or final_url in pages:
                continue
            pages[final_url] = html

            page_score = self.calculate_score(html)
            if page_score < threshold:
                continue
            relevant_urls.add(final_url)
            with Crawler._request_lock:
                aliases = Crawler._load_alias_index()["aliases"]
            for link in self.get_article_links(html, final_url):
                link = self.canonical_url(link, aliases)
                if link not in visited and link != final_url:
                    # Negative scores give a max-priority queue; ties use discovery order.
                    heappush(queue, (-page_score, next(sequence), link))
        rate = harvest_rate(pages, relevant_urls)
        print(f"Downloaded pages: {len(pages)}")
        print(f"Music-related share (harvest rate): {rate:.2%}")
        return {author_name: pages}

    def crawl_into_graph(self, author_name: str, graph: dict[str, list[str]],
                         max_pages: int = 10, threshold: float = 2, *,
                         seed: str | None = None, canonicalize: bool = True) -> dict[str, list[str]]:
        """Crawl one artist and merge discovered article links into graph in place.

        graph is a shared URL -> list[URL] adjacency dictionary. Existing nodes
        and edges are retained. New destinations get an empty adjacency list
        until their own links are collected. Return the same graph object.
        This is a temporary discovery graph; build_graph_from_cache creates
        the final graph containing only downloaded articles and internal links.
        """
        options = {"max_pages": max_pages, "threshold": threshold}
        if seed is not None:
            options["seed"] = seed
        pages = self.focused_crawl(author_name, **options)[author_name]
        with Crawler._request_lock:
            aliases = Crawler._load_alias_index()["aliases"]
        for url, html in pages.items():
            source = self.canonical_url(url, aliases)
            neighbors = graph.setdefault(source, [])
            known = set(neighbors)
            for link in self.get_article_links(html, source):
                target = self.canonical_url(link, aliases)
                if target == source:
                    continue
                graph.setdefault(target, [])
                if target not in known:
                    neighbors.append(target)
                    known.add(target)
        if canonicalize:
            self.canonicalize_graph(graph)
        return graph

    @staticmethod
    def build_graph_from_cache() -> dict[str, list[str]]:
        """Build the graph of downloaded articles in two offline passes.

        First collect canonical URLs of valid saved English Wikipedia pages.
        Then retain only links between those pages. Discovered URLs without a
        cached article never become nodes. Keep downloaded isolated pages,
        merge aliases and duplicate edges, and read one HTML page at a time.
        Intern shared URLs and never crawl or keep a collection of HTML bodies.
        """
        if not PAGE_CACHE_DIR.is_dir():
            raise FileNotFoundError(f"HTML cache directory does not exist: {PAGE_CACHE_DIR}")
        with Crawler._request_lock:
            aliases = dict(Crawler._load_alias_index()["aliases"])
        crawler = Crawler()
        graph = {}
        records = []
        print("Graf iz kesa: popis preuzetih clanaka...", flush=True)
        for path in PAGE_CACHE_DIR.glob("*.json"):
            if path.name == "url_aliases.json":
                continue
            try:
                cached = Crawler._read_cache_json(path)
            except (FileNotFoundError, UnicodeError, ValueError):
                continue
            if (not isinstance(cached, dict) or not isinstance(cached.get("url"), str)
                    or not isinstance(cached.get("text"), str)):
                del cached
                continue
            if not Crawler.is_article(cached["url"]):
                del cached
                continue
            requested = Crawler._normalize(cached["url"])
            source = Crawler._cached_final_url(cached)
            if not Crawler.is_article(source):
                del cached
                continue
            if requested != source:
                aliases[requested] = source
            del cached
            records.append((path, source))
            if len(records) % 250 == 0:
                print(f"Graf iz kesa: provereno {len(records)} zapisa.", flush=True)
        if not records:
            raise ValueError("HTML cache contains no valid downloaded Wikipedia articles.")

        # Resolve after all metadata has been read: a later record may identify
        # the destination of an earlier alias. Artist labels use the same map.
        for position, (path, source) in enumerate(records):
            source = intern(Crawler.canonical_url(source, aliases))
            records[position] = (path, source)
            graph.setdefault(source, [])
        with Crawler._request_lock:
            state = Crawler._load_alias_index()
            merged_aliases = {**state["aliases"], **aliases}
            if merged_aliases != state["aliases"] or not state.get("legacy_cache_scanned"):
                Crawler._save_alias_index({"aliases": merged_aliases, "legacy_cache_scanned": True})

        print(f"Graf iz kesa: {len(graph)} preuzetih clanaka; izdvajanje unutrasnjih veza...", flush=True)
        for processed, (path, source) in enumerate(records, 1):
            try:
                cached = Crawler._read_cache_json(path)
            except (FileNotFoundError, UnicodeError, ValueError):
                continue
            if not isinstance(cached, dict) or not isinstance(cached.get("text"), str):
                del cached
                continue
            links = crawler.get_article_links(cached["text"], source)
            del cached
            neighbors = graph[source]
            known = set(neighbors)
            for link in links:
                target = Crawler.canonical_url(link, aliases)
                if target in graph and target != source and target not in known:
                    neighbors.append(intern(target))
                    known.add(target)
            if processed % 250 == 0:
                print(f"Graf iz kesa: obradjeno {processed}/{len(records)} zapisa, {len(graph)} cvorova.", flush=True)
        return graph

    @staticmethod
    def build_artist_nodes(artist_names: Iterable[str],
                           graph: Mapping[str, Iterable[str]], *,
                           artist_seeds: Mapping[str, str | None] | None = None) -> dict[str, str]:
        """Map catalog names to graph nodes using resolved, prepared seed URLs.

        Omit names whose seed is absent. This only builds a lookup dictionary;
        it performs no crawling and does not modify graph.
        """
        artist_names = list(artist_names)
        if not artist_names or not graph:
            return {}
        if artist_seeds is None:
            artist_seeds = Crawler._artist_seeds()
        with Crawler._request_lock:
            aliases = Crawler._load_alias_index()["aliases"]
        artist_nodes = {}
        for artist in artist_names:
            seed = artist_seeds.get(artist)
            if not isinstance(seed, str) or not seed:
                continue
            seed = Crawler.canonical_url(seed, aliases)
            if seed in graph:
                artist_nodes[artist] = seed
        return artist_nodes

    @staticmethod
    def build_graph(links: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
        """Keep unique internal edges, excluding self-links and preserving dangling nodes."""
        graph = {}
        for url, targets in links.items():
            graph[url] = []
            seen = set()
            for target in targets:
                if target in links and target != url and target not in seen:
                    graph[url].append(target)
                    seen.add(target)
        return graph

if __name__ == "__main__":
    # Example usage outside the class. HTML and robots rules are local fixtures.
    crawler = Crawler()
    current_url = WIKI + "David_Bowie"
    webs = crawler.fetch_page(current_url)[1]
    links = crawler.get_links(webs, current_url)

    robots_text = crawler.fetch_page("https://en.wikipedia.org/robots.txt")[1]
    article_links = crawler.get_article_links(webs, current_url, verbose=True)
    blocked_paths = crawler.parse_robots(robots_text)
    print(blocked_paths)
