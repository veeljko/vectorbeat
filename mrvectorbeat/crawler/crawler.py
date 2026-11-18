from collections.abc import Mapping

from bs4 import BeautifulSoup
from urllib.parse import urljoin


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
