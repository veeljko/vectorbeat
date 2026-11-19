import unittest
from unittest.mock import patch

from mrvectorbeat.crawler.crawler import Crawler, WIKI


class WikiFilterTests(unittest.TestCase):
    def test_non_article_namespaces_and_encoded_aliases_are_rejected(self):
        for title in ("File_talk:Music", "Category_talk:Music", "Help_talk:Music",
                      "Wikipedia_talk:Music", "Portal_talk:Music", "Draft_talk:Music",
                      "Module_talk:Music", "MediaWiki_talk:Music", "Image:Cover.jpg",
                      "Image_talk:Cover.jpg", "File%20talk%3AMusic", "file_talk:Music",
                      "Project:Rules", "WP:Rules"):
            with self.subTest(title=title):
                self.assertFalse(Crawler.is_article(WIKI + title))
        self.assertTrue(Crawler.is_article(WIKI + "Star_Trek:_The_Next_Generation"))
        self.assertFalse(Crawler.is_article("https://de.wikipedia.org/wiki/Music"))

    def test_equivalent_wikipedia_urls_normalize_to_one_article(self):
        variants = [WIKI + "ABBA", WIKI + "ABB%41", WIKI + "ABBA?oldid=123",
                    WIKI + "ABBA#History", WIKI + "ABBA?oldid=123#History"]
        self.assertEqual({Crawler._normalize(url) for url in variants}, {WIKI + "ABBA"})
        self.assertEqual(Crawler._normalize(WIKI + "David%20Bowie"), WIKI + "David_Bowie")
        self.assertEqual(Crawler._normalize(WIKI + "AC%2FDC"), WIKI + "AC/DC")

    def test_normalization_preserves_other_site_queries_and_is_idempotent(self):
        self.assertEqual(Crawler._normalize("https://example.com/calendar?month=3#top"),
                         "https://example.com/calendar?month=3")
        url = WIKI + "Bj%C3%B6rk?oldid=42#Albums"
        normalized = Crawler._normalize(url)
        self.assertEqual(normalized, WIKI + "Bj%C3%B6rk")
        self.assertEqual(Crawler._normalize(normalized), normalized)

    def test_extracted_article_links_are_deduplicated_and_filtered(self):
        html = '''<a href="/wiki/ABBA">A</a><a href="/wiki/ABB%41">B</a>
                  <a href="/wiki/ABBA?oldid=123">C</a><a href="/wiki/ABBA#History">D</a>
                  <a href="/wiki/File_talk:Music">Talk</a>
                  <a href="/wiki/Image:Cover.jpg">Image</a>'''
        self.assertEqual(Crawler().get_article_links(html, WIKI + "Source"), [WIKI + "ABBA"])

    def test_graph_merges_url_variants_into_one_node(self):
        seed = WIKI + "Source"
        pages = {seed: '<a href="/wiki/ABBA">A</a><a href="/wiki/ABB%41">B</a>'
                       '<a href="/wiki/ABBA?oldid=123">C</a>'}
        graph = {}
        crawler = Crawler()
        with patch.object(crawler, "focused_crawl", return_value={"Source": pages}):
            crawler.crawl_into_graph("Source", graph)
        self.assertEqual(graph, {seed: [WIKI + "ABBA"], WIKI + "ABBA": []})

    def test_expanded_keywords_match_words_and_simple_plurals(self):
        crawler = Crawler()
        self.assertGreaterEqual(crawler.calculate_score("<p>The drummer and vocalist joined a jazz band.</p>"), 4)
        self.assertEqual(crawler.calculate_score("<p>album albums concert concerts</p>"), 4)
        self.assertEqual(crawler.calculate_score("<p>albumin in an abandoned warehouse</p>"), 0)


if __name__ == "__main__":
    unittest.main()
