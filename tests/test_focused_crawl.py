import contextlib
import io
import unittest
from copy import deepcopy
from unittest.mock import call, patch

from mrvectorbeat.crawler.crawler import Crawler, WIKI
from mrvectorbeat.search.ranking import harvest_rate


def page(*targets, text=""):
    return "<p>" + text + "</p>" + "".join(f'<a href="{url}">Link</a>' for url in targets)


class FocusedCrawlTests(unittest.TestCase):
    def test_prioritizes_links_from_higher_scoring_parents(self):
        seed, low, high, low_child, high_child = [
            WIKI + name for name in ("Seed", "A_low", "B_high", "Low_child", "High_child")]
        html = {seed: page(low, high), low: page(low_child), high: page(high_child),
                low_child: page(seed), high_child: page(seed)}
        scores = {seed: 10, low: 3, high: 8, low_child: 2, high_child: 2}
        scores_by_html = {html[url]: scores[url] for url in html}
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=lambda url: (200, html[url])) as fetch, \
                patch.object(crawler, "calculate_score", side_effect=lambda body: scores_by_html[body]) as score:
            pages = crawler.focused_crawl(seed, [])
        expected = [seed, low, high, high_child, low_child]
        self.assertEqual(fetch.call_args_list, [call(url) for url in expected])
        self.assertEqual(list(pages), expected)
        for url in expected:
            self.assertIn(call(html[url]), score.call_args_list)

    def test_skips_blocked_failed_and_missing_bodies_and_stops_at_low_scores(self):
        seed, blocked, failed, no_body, low, child = [
            WIKI + name for name in ("Seed", "Blocked", "Failed", "No_body", "Low", "Child")]
        responses = {seed: (200, page(blocked, failed, no_body, low)),
                     failed: (404, None), no_body: (200, None), low: (200, page(child))}
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=lambda url: responses[url]) as fetch, \
                patch.object(crawler, "calculate_score",
                             side_effect=lambda body: 1 if body == responses[low][1] else 10):
            pages = crawler.focused_crawl(seed, iter(["/wiki/Blocked"]))
        self.assertEqual(set(pages), {seed, low})
        self.assertNotIn(call(blocked), fetch.call_args_list)
        self.assertNotIn(call(child), fetch.call_args_list)

    def test_successful_page_limit_and_duplicate_links(self):
        seed, first, second = [WIKI + name for name in ("Seed", "A_first", "B_second")]
        html = {seed: page(first, first + "#History", second), first: page(seed), second: ""}
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=lambda url: (200, html[url])) as fetch, \
                patch.object(crawler, "calculate_score", return_value=5):
            pages = crawler.focused_crawl(seed + "#top", [], max_pages=2)
        self.assertEqual(list(pages), [seed, first])
        self.assertEqual(fetch.call_args_list, [call(seed), call(first)])

    def test_uses_existing_score_by_default(self):
        seed, child, ignored = [WIKI + name for name in ("Seed", "Child", "Ignored")]
        html = {seed: page(child, text="music singer"), child: page(ignored, text="unrelated")}
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=lambda url: (200, html[url])) as fetch:
            pages = crawler.focused_crawl(seed, [])
        self.assertEqual(list(pages), [seed, child])
        self.assertEqual(fetch.call_args_list, [call(seed), call(child)])

    def test_does_not_retry_a_failed_url_discovered_again(self):
        seed, failed, other = [WIKI + name for name in ("Seed", "A_failed", "B_other")]
        responses = {seed: (200, page(failed, other)), failed: (None, None),
                     other: (200, page(failed))}
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=lambda url: responses[url]) as fetch, \
                patch.object(crawler, "calculate_score", return_value=5):
            pages = crawler.focused_crawl(seed, [])
        self.assertEqual(set(pages), {seed, other})
        self.assertEqual(fetch.call_args_list.count(call(failed)), 1)

    def test_zero_limit_blocked_seed_and_invalid_configuration(self):
        crawler = Crawler()
        with patch.object(crawler, "fetch_page") as fetch:
            self.assertEqual(crawler.focused_crawl(WIKI + "Seed", [], max_pages=0), {})
            self.assertEqual(crawler.focused_crawl(WIKI + "Seed", ["/"]), {})
            fetch.assert_not_called()
        for parameters in ({"max_pages": -1}, {"max_pages": True}, {"threshold": float("nan")}):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    crawler.focused_crawl(WIKI + "Seed", [], **parameters)

    def test_build_graph_keeps_unique_internal_edges_and_dangling_nodes(self):
        links = {"A": ["A", "B", "B", "outside"], "B": ["A", "C"], "C": []}
        original = deepcopy(links)
        expected = {"A": ["B"], "B": ["A", "C"], "C": []}
        self.assertEqual(Crawler.build_graph(links), expected)
        self.assertEqual(Crawler().build_graph(links), expected)
        self.assertEqual(links, original)
        self.assertEqual(Crawler.build_graph({}), {})

    def test_prints_download_count_and_content_based_harvest_rate(self):
        seed, music, weather, failed, blocked = [
            WIKI + name for name in ("Seed", "Music", "Weather", "Failed", "Blocked")]
        responses = {
            seed: (200, page(music, weather, failed, blocked, text="music singer")),
            music: (200, page(text="album concert")),
            weather: (200, page(text="weather forecast")),
            failed: (500, None),
        }
        output = io.StringIO()
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", side_effect=lambda url: responses[url]), \
                patch("mrvectorbeat.crawler.crawler.harvest_rate", wraps=harvest_rate) as metric, \
                contextlib.redirect_stdout(output):
            pages = crawler.focused_crawl(seed, ["/wiki/Blocked"])
        self.assertEqual(set(pages), {seed, music, weather})
        metric.assert_called_once_with(pages, {seed, music})
        self.assertIn("Downloaded pages: 3", output.getvalue())
        self.assertIn("Music-related share (harvest rate): 66.67%", output.getvalue())

    def test_prints_zero_harvest_rate_when_no_pages_are_downloaded(self):
        output = io.StringIO()
        crawler = Crawler()
        with patch.object(crawler, "fetch_page", return_value=(500, None)), \
                contextlib.redirect_stdout(output):
            self.assertEqual(crawler.focused_crawl(WIKI + "Seed", []), {})
        self.assertIn("Downloaded pages: 0", output.getvalue())
        self.assertIn("Music-related share (harvest rate): 0.00%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
