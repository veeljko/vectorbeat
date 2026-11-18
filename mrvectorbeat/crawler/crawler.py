from collections.abc import Iterable, Mapping
from heapq import heappop, heappush
from itertools import count
import math

import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote, urljoin, urlsplit

from mrvectorbeat.search.ranking.harvest_rate import harvest_rate


WIKI = "https://en.wikipedia.org/wiki/"
NON_ARTICLE_PREFIXES = (
    "Special:", "File:", "Category:", "Help:", "Talk:", "Template:", "Template_talk:",
    "Wikipedia:", "Portal:", "User:", "User_talk:", "Draft:", "Module:", "Main_Page",
)
KEYWORDS = ["music", "singer", "songwriter", "album",
             "concert", "composer", "guitar", "orchestra"]

class Crawler:
    @staticmethod
    def fetch_page(url: str, headers: Mapping[str, str] | None = None,
                   timeout: float = 10) -> tuple[int | None, str | None]:
        """Return (status, response text) for HTTP 200, or (status, None) otherwise.

        Send a crawler User-Agent by default; custom headers can override it.
        Request failures print the exception type and return (None, None).
        """
        request_headers = {"User-Agent": "UPI-Crawler/1.0 (contact: vmladenovic1524rn@raf.rs)"}
        if headers is not None:
            request_headers.update(headers)
        try:
            response = requests.get(url, headers=request_headers, timeout=timeout)
            status = response.status_code
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
        return web.split("#")[0]        # fragment ne menja stranicu


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
            if title.startswith(prefix):
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
        text = self.get_text(html).lower()
        score = 0
        for rec in KEYWORDS:
            score += text.count(rec)
        return score

    def focused_crawl(self, seed: str, blocked_paths: Iterable[str],
                      max_pages: int = 50, threshold: float = 2) -> dict[str, str]:
        """Fetch pages by descending parent score, expanding only on-topic pages.

        calculate_score(html) evaluates each downloaded page.
        Successful off-topic pages are retained but their
        links are not followed. Robots rules are checked before each request.
        Return a URL -> HTML dictionary containing at most max_pages pages.
        Print the successful download count and music-related harvest rate.
        """
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 0:
            raise ValueError("max_pages must be a nonnegative integer.")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite.")

        sequence = count()
        queue = [(-1000.0, next(sequence), self._normalize(seed))]
        visited = set()
        pages = {}
        relevant_urls = set()
        blocked_paths = tuple(blocked_paths)

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
        return pages

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

    robots_text = crawler.fetch_page("https://en.wikipedia.org/robots.txt", headers=None, timeout=1000)[1]
    article_links = crawler.get_article_links(webs, current_url, verbose=True)
    blocked_paths = crawler.parse_robots(robots_text)
    print(blocked_paths)
