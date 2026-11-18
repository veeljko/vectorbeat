import unittest

from mrvectorbeat.crawler.crawler import Crawler


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


if __name__ == "__main__":
    unittest.main()
