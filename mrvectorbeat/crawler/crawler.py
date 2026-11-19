from collections.abc import Hashable, Iterable, Mapping
from heapq import heappop, heappush
from itertools import count
from threading import Lock
import math
import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, unquote, urljoin, urlsplit

from mrvectorbeat.search.ranking.harvest_rate import harvest_rate


WIKI = "https://en.wikipedia.org/wiki/"
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
    _request_lock = Lock()
    _last_request_finished: float | None = None

    @staticmethod
    def fetch_page(url: str) -> tuple[int | None, str | None]:
        """Return (status, response text) for HTTP 200, or (status, None) otherwise.

        Leave at least one second between the end of one request and the next.
        The first request starts immediately; no delay follows the last request.
        The interval is shared by all instances and applies to redirects too.
        Request failures print the exception type and return (None, None).
        """
        request_headers = {"User-Agent": "UPI-Crawler/1.0 (contact: vmladenovic1524rn@raf.rs)"}
        redirects = 0
        try:
            while True:
                with Crawler._request_lock:
                    while Crawler._last_request_finished is not None:
                        remaining = 1 - (time.monotonic() - Crawler._last_request_finished)
                        if remaining <= 0:
                            break
                        time.sleep(remaining)
                    try:
                        # Follow redirects here so every HTTP hop observes the same interval.
                        response = requests.get(url, headers=request_headers, allow_redirects=False)
                    finally:
                        Crawler._last_request_finished = time.monotonic()
                status = response.status_code
                if status in (301, 302, 303, 307, 308) and response.headers.get("Location"):
                    next_url = urljoin(url, response.headers["Location"])
                    response.close()
                    if redirects >= 30:
                        raise requests.TooManyRedirects("Exceeded 30 redirects.")
                    redirects += 1
                    url = next_url
                    continue
                data = response.text if status == 200 else None
                return status, data
        except requests.RequestException as error:
            print("Request failed:", type(error).__name__)
            return None, None

    @staticmethod
    def _raw_links(html: str, web: str) -> list[str]:
        """Svi linkovi sa stranice, dopunjeni do apsolutnog URL-a."""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            links.append(urljoin(web, a["href"]))
        return links

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
        for symbol in soup(["script", "style"]):
            symbol.decompose()
        return " ".join(soup.get_text(separator=" ", strip=True).split())

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

    def focused_crawl(self,
                      author_name: str,
                      max_pages: int = 50,
                      threshold: float = 2, *,
                      seed: str | None = None) -> dict[str, dict[str, str]]:
        """Fetch pages by descending parent score, expanding only on-topic pages.

        calculate_score(html) evaluates each downloaded page.
        Successful off-topic pages are retained but their
        links are not followed. Robots rules are checked before each request.
        Return author_name -> URL -> HTML, with at most max_pages pages.
        Use the supplied seed or derive it from author_name.
        Print the successful download count and music-related harvest rate.
        """
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 0:
            raise ValueError("max_pages must be a nonnegative integer.")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite.")

        seed = self.author_seed(author_name) if seed is None else seed
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
            if url in visited:
                continue
            visited.add(url)
            if not self.is_allowed(url, blocked_paths):
                continue

            # fetch_page returns status first, followed by the response body.
            status, html = self.fetch_page(url)
            if status != 200 or html is None:
                continue
            pages[url] = html

            page_score = self.calculate_score(html)
            if page_score < threshold:
                continue
            relevant_urls.add(url)
            for link in self.get_article_links(html, url):
                link = self._normalize(link)
                if link not in visited:
                    # Negative scores give a max-priority queue; ties use discovery order.
                    heappush(queue, (-page_score, next(sequence), link))
        rate = harvest_rate(pages, relevant_urls)
        print(f"Downloaded pages: {len(pages)}")
        print(f"Music-related share (harvest rate): {rate:.2%}")
        return {author_name: pages}

    def crawl_into_graph(self, author_name: str, graph: dict[str, list[str]],
                         max_pages: int = 50, threshold: float = 2) -> dict[str, list[str]]:
        """Crawl one artist and merge discovered article links into graph in place.

        graph is a shared URL -> list[URL] adjacency dictionary. Existing nodes
        and edges are retained. New destinations get an empty adjacency list
        until their own links are collected. Return the same graph object.
        """
        pages = self.focused_crawl(author_name, max_pages=max_pages,
                                   threshold=threshold)[author_name]
        for url, html in pages.items():
            source = self._normalize(url)
            neighbors = graph.setdefault(source, [])
            known = set(neighbors)
            for link in self.get_article_links(html, source):
                target = self._normalize(link)
                if target == source:
                    continue
                graph.setdefault(target, [])
                if target not in known:
                    neighbors.append(target)
                    known.add(target)
        return graph

    @staticmethod
    def build_artist_nodes(artist_names: Iterable[str],
                           graph: Mapping[str, Iterable[str]]) -> dict[str, str]:
        """Map catalog artist names to their existing seed URL nodes in graph.

        Omit names whose seed is absent. This only builds a lookup dictionary;
        it performs no crawling and does not modify graph.
        """
        artist_nodes = {}
        for artist in artist_names:
            seed = Crawler._normalize(Crawler.author_seed(artist))
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
    webs = crawler.fetch_page(current_url, headers=None, timeout=1000)[1]
    links = crawler.get_links(webs, current_url)

    robots_text = crawler.fetch_page("https://en.wikipedia.org/robots.txt")[1]
    article_links = crawler.get_article_links(webs, current_url, verbose=True)
    blocked_paths = crawler.parse_robots(robots_text)
    print(blocked_paths)
