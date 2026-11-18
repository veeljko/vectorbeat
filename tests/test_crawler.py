import contextlib
import io
import unittest

from mrvectorbeat.crawler.crawler import Crawler, NON_ARTICLE_PREFIXES, WIKI


class CrawlerTests(unittest.TestCase):
    def test_instance_returns_all_absolute_links_without_fragments(self):
        html = '''<a href="../Artist#Career">Artist</a>
                  <a href="?page=2#Top">Next</a>
                  <a href="https://other.example/Music#History">Other</a>
                  <a>No href</a>'''
        url = "https://example.com/wiki/Page"
        self.assertEqual(Crawler().get_links({url: html}, url), [
            "https://example.com/Artist",
            "https://example.com/wiki/Page?page=2",
            "https://other.example/Music",
        ])

    def test_empty_page_and_page_without_links(self):
        crawler = Crawler()
        for html in ("", "<p>Text</p>", "<a>Name only</a>"):
            with self.subTest(html=html):
                url = "https://example.com/"
                self.assertEqual(crawler.get_links({url: html}, url), [])

    def test_helpers_work_on_instance_and_class(self):
        html = '<a href="/Music#History">Music</a>'
        url = "https://example.com/"
        for target in (Crawler, Crawler()):
            self.assertEqual(target._raw_links({url: html}, url),
                             ["https://example.com/Music#History"])
            self.assertEqual(target._normalize("https://example.com/?page=2#Part"),
                             "https://example.com/?page=2")

    def test_text_separates_elements_and_excludes_scripts_and_styles(self):
        html = '''<style>hidden style</style><script>hidden script</script>
                  <p>First</p><p>Second <b>word</b>.</p><div> More\n text &amp; music </div>'''
        expected = "First Second word . More text & music"
        url = "https://example.com/"
        self.assertEqual(Crawler().get_text({url: html}, url), expected)
        self.assertEqual(Crawler.get_text({url: html}, url), expected)
        self.assertEqual(Crawler().get_text({url: ""}, url), "")

    def test_bytes_html(self):
        html = b'<p>Hello</p><a href="/music">Music</a>'
        url = "https://example.com/"
        self.assertEqual(Crawler().get_text({url: html}, url), "Hello Music")
        self.assertEqual(Crawler().get_links({url: html}, url),
                         ["https://example.com/music"])

    def test_dictionary_selects_the_requested_page_and_ignores_comments(self):
        webs = {
            "http://mini.ir/index.html": '''<h1>Handbook</h1>
                <!-- <a href="promo.html">Old ad</a> -->
                <a href="ranking.html">Ranking</a>''',
            "http://mini.ir/ranking.html": '''<h1>Ranking</h1>
                <a href="bm25.html#parameters">BM25</a>''',
        }
        crawler = Crawler()
        self.assertEqual(crawler.get_links(webs, "http://mini.ir/index.html"),
                         ["http://mini.ir/ranking.html"])
        self.assertEqual(crawler.get_links(webs, "http://mini.ir/ranking.html#top"),
                         ["http://mini.ir/bm25.html"])
        self.assertEqual(crawler.get_text(webs, "http://mini.ir/ranking.html#top"), "Ranking BM25")
        self.assertNotIn("Old ad", crawler.get_text(webs, "http://mini.ir/index.html"))

    def test_unknown_page_has_no_links_or_text(self):
        webs = {"http://mini.ir/index.html": "<p>Home</p>"}
        crawler = Crawler()
        self.assertEqual(crawler.get_links(webs, "http://mini.ir/calendar?month=1"), [])
        self.assertEqual(crawler.get_text(webs, "http://mini.ir/calendar?month=1"), "")
        self.assertEqual(crawler.get_links({}, "http://mini.ir/index.html"), [])

    def test_parse_robots_uses_wildcard_groups_and_ignores_empty_rules(self):
        text = """# Site rules
            User-agent: SpecificBot
            Disallow: /specific-only

            USER-AGENT: *
            Disallow: /calendar # Infinite calendar pages
            disallow : /private/
            Disallow:

            User-agent: AnotherBot
            Disallow: /

            User-agent: *
            Disallow: /drafts
        """
        for target in (Crawler, Crawler()):
            self.assertEqual(target.parse_robots(text), ["/calendar", "/private/", "/drafts"])
            self.assertEqual(target.parse_robots(""), [])

    def test_parse_robots_supports_multiple_user_agents_in_one_group(self):
        text = """User-agent: *
            User-agent: ExampleBot
            Disallow: /private

            User-agent: OtherBot
            Disallow: /other
        """
        self.assertEqual(Crawler.parse_robots(text), ["/private"])

    def test_get_path_preserves_query_and_excludes_fragment(self):
        cases = {
            "http://mini.ir/calendar?month=3": "/calendar?month=3",
            "https://mini.ir:8443/bm25.html#parameters": "/bm25.html",
            "http://mini.ir": "/",
            "http://mini.ir/": "/",
            "http://mini.ir?month=3#top": "/?month=3",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(Crawler().get_path(url), expected)

    def test_is_allowed_matches_path_prefixes(self):
        blocked = Crawler.parse_robots("User-agent: *\nDisallow: /calendar\nDisallow: /private/")
        for target in (Crawler, Crawler()):
            self.assertFalse(target.is_allowed("http://mini.ir/calendar?month=3", blocked))
            self.assertFalse(target.is_allowed("http://mini.ir/private/page.html#top", blocked))
            self.assertTrue(target.is_allowed("http://mini.ir/ranking.html", blocked))
            self.assertTrue(target.is_allowed("http://mini.ir/page?next=/calendar", blocked))
            self.assertTrue(target.is_allowed("http://mini.ir/", []))
            self.assertFalse(target.is_allowed("http://mini.ir/anything", ["/"]))

    def test_is_article_accepts_only_english_wikipedia_articles(self):
        for target in (Crawler, Crawler()):
            self.assertTrue(target.is_article(WIKI + "Music"))
            self.assertTrue(target.is_article(WIKI + "AC/DC"))
            self.assertTrue(target.is_article(WIKI + "Star_Trek:_The_Next_Generation"))
            for prefix in NON_ARTICLE_PREFIXES:
                with self.subTest(prefix=prefix):
                    self.assertFalse(target.is_article(WIKI + prefix + "Example"))
            for url in ("https://de.wikipedia.org/wiki/Music", "https://example.com/wiki/Music",
                        "https://en.wikipedia.org.evil.example/wiki/Music",
                        "http://en.wikipedia.org/wiki/Music", WIKI, WIKI + "#top",
                        WIKI + "File%3ACover.jpg"):
                with self.subTest(url=url):
                    self.assertFalse(target.is_article(url))

    def test_is_article_excludes_self_links_with_fragments_and_queries(self):
        current = WIKI + "David_Bowie"
        for suffix in ("", "#Career", "?oldid=123", "?oldid=123#Career"):
            with self.subTest(suffix=suffix):
                self.assertFalse(Crawler.is_article(current + suffix, current))
        self.assertTrue(Crawler.is_article(WIKI + "ABBA", current))

    def test_get_article_links_filters_sorts_deduplicates_and_prints_counts(self):
        current = WIKI + "David_Bowie"
        webs = {current: '''
            <a href="/wiki/Ziggy_Stardust#Biography">Ziggy</a>
            <a href="/wiki/Ziggy_Stardust#Discography">Ziggy again</a>
            <a href="/wiki/ABBA">ABBA</a>
            <a href="#top">Self</a><a href="?oldid=123">Previous version</a>
            <a href="/wiki/File:Cover.jpg">Cover</a>
            <a href="https://de.wikipedia.org/wiki/Music">Other language</a>
            <a href="/wiki/Main_Page">Home</a>
            <!-- <a href="/wiki/Commented_out">Ignored</a> -->
        '''}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            articles = Crawler().get_article_links(webs, current, verbose=True)
        self.assertEqual(articles, [WIKI + "ABBA", WIKI + "Ziggy_Stardust"])
        self.assertIn("Links in HTML : 8", output.getvalue())
        self.assertIn("Unique links  : 7", output.getvalue())
        self.assertIn("Article links : 2", output.getvalue())
        self.assertIn("ABBA", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as silent:
            self.assertEqual(Crawler().get_article_links(webs, current), articles)
        self.assertEqual(silent.getvalue(), "")

    def test_article_preview_prints_only_eight_titles_without_truncating_results(self):
        current = WIKI + "Source"
        html = "".join(f'<a href="/wiki/Artist_{i:02}">{i}</a>' for i in range(10))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            articles = Crawler().get_article_links({current: html}, current, verbose=True)
        self.assertEqual(len(articles), 10)
        self.assertEqual(len([line for line in output.getvalue().splitlines() if line.startswith("   ")]), 8)
        self.assertEqual(Crawler().get_article_links({}, current), [])


if __name__ == "__main__":
    unittest.main()
