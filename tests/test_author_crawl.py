import contextlib
import io
import unittest
from unittest.mock import call, patch

from mrvectorbeat.crawler.crawler import Crawler, WIKI


ROBOTS_URL = "https://en.wikipedia.org/robots.txt"


class AuthorCrawlTests(unittest.TestCase):
    def test_seed_uses_catalog_name_and_encodes_special_characters(self):
        cases = {
            "David Bowie": WIKI + "David_Bowie",
            "  Bell   Biv DeVoe ": WIKI + "Bell_Biv_DeVoe",
            "AC/DC": WIKI + "AC/DC",
            "P!nk": WIKI + "P%21nk",
            "Róisín Murphy": WIKI + "R%C3%B3is%C3%ADn_Murphy",
            "Artist #1?": WIKI + "Artist_%231%3F",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(Crawler.author_seed(name), expected)

    def test_wraps_graph_in_supplied_author_id_and_preserves_all_downloaded_nodes(self):
        name = "David Bowie"
        seed, peer, isolated = WIKI + "David_Bowie", WIKI + "ABBA", WIKI + "Isolated"
        pages = {
            seed: '<a href="/wiki/ABBA">A</a><a href="/wiki/ABBA#History">A again</a>'
                  '<a href="/wiki/Outside">Outside</a><a href="#top">Self</a>',
            peer: '<a href="/wiki/David_Bowie">Source</a>',
            isolated: "<p>Unrelated page with no links.</p>",
        }
        expected_graph = {seed: [peer], peer: [seed], isolated: []}
        for author_id in (42, "artist-42"):
            with self.subTest(author_id=author_id):
                crawler = Crawler()
                with patch.object(crawler, "focused_crawl", return_value={name: pages}) as focused:
                    result = crawler.crawl_author(name, author_id, max_pages=20, threshold=3)
                self.assertEqual(result, {author_id: expected_graph})
                focused.assert_called_once_with(name, max_pages=20, threshold=3, seed=seed)

    def test_actual_focused_crawl_fetches_robots_first_without_global_instance(self):
        name, author_id = "David Bowie", "artist-bowie"
        seed, peer, blocked = WIKI + "David_Bowie", WIKI + "ABBA", WIKI + "Blocked"
        responses = {
            ROBOTS_URL: (200, "User-agent: *\nDisallow: /wiki/Blocked\n"),
            seed: (200, '<p>music singer</p><a href="/wiki/ABBA">ABBA</a>'
                        '<a href="/wiki/Blocked">Blocked</a>'),
            peer: (200, '<p>Weather.</p><a href="/wiki/David_Bowie">Source</a>'),
        }
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=lambda url: responses[url]) as fetch, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = crawler.crawl_author(name, author_id)
        self.assertEqual(result, {author_id: {seed: [peer], peer: [seed]}})
        self.assertEqual(fetch.call_args_list, [call(ROBOTS_URL), call(seed), call(peer)])
        self.assertNotIn(call(blocked), fetch.call_args_list)
        self.assertIn("Downloaded pages: 2", output.getvalue())
        self.assertIn("harvest rate): 50.00%", output.getvalue())

    def test_empty_crawl_keeps_author_id(self):
        crawler = Crawler()
        with patch.object(crawler, "focused_crawl", return_value={"Artist": {}}):
            self.assertEqual(crawler.crawl_author("Artist", 17), {17: {}})

    def test_zero_page_limit_does_not_send_http_requests(self):
        crawler = Crawler()
        with patch.object(crawler, "fetch_page") as fetch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(crawler.crawl_author("Artist", 17, max_pages=0), {17: {}})
        fetch.assert_not_called()

    def test_robots_failure_is_reported_before_any_article_is_requested(self):
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", return_value=(None, None)) as fetch:
            with self.assertRaisesRegex(RuntimeError, "robots.txt"):
                crawler.crawl_author("Artist", 17)
        fetch.assert_called_once_with(ROBOTS_URL)

    def test_invalid_name_is_rejected_before_crawling(self):
        crawler = Crawler()
        with patch.object(crawler, "focused_crawl") as focused:
            for name in ("", "  ", None):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        crawler.crawl_author(name, 17)
        focused.assert_not_called()


if __name__ == "__main__":
    unittest.main()
