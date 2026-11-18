from collections.abc import Iterable, Mapping

from bs4 import BeautifulSoup
from urllib.parse import unquote, urljoin, urlsplit


WIKI = "https://en.wikipedia.org/wiki/"
NON_ARTICLE_PREFIXES = (
    "Special:", "File:", "Category:", "Help:", "Talk:", "Template:", "Template_talk:",
    "Wikipedia:", "Portal:", "User:", "User_talk:", "Draft:", "Module:", "Main_Page",
)

class Crawler:
    @staticmethod
    def _raw_links(webs: Mapping[str, str | bytes], web: str) -> list[str]:
        """Svi linkovi sa stranice, dopunjeni do apsolutnog URL-a."""
        html = webs.get(Crawler._normalize(web), "")
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            links.append(urljoin(web, a["href"]))
        return links

    @staticmethod
    def _normalize(web: str) -> str:
        return web.split("#")[0]        # fragment ne menja stranicu


    def get_links(self, webs: Mapping[str, str | bytes], web: str) -> list[str]:
        """Linkovi iz webs[web], bez fragmenata; nepoznata stranica daje praznu listu."""
        links = []
        for link in self._raw_links(webs, web):
            links.append(self._normalize(link))
        return links


    @staticmethod
    def get_text(webs: Mapping[str, str | bytes], web: str) -> str:
        """Tekst stranice iz recnika URL -> HTML; nepoznat URL daje prazan tekst."""
        html = webs.get(Crawler._normalize(web), "")
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

    def get_article_links(self, webs: Mapping[str, str | bytes], web: str, *,
                          verbose: bool = False) -> list[str]:
        """Return sorted unique article links; optionally print counts and eight titles."""
        all_links = self.get_links(webs, web)
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
