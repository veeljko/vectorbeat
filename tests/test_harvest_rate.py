import unittest

from mrvectorbeat.search.ranking import harvest_rate


class HarvestRateTests(unittest.TestCase):
    def test_fraction_of_relevant_urls(self):
        self.assertAlmostEqual(harvest_rate(["a", "b", "c"], {"a", "c"}), 2 / 3)
        self.assertEqual(harvest_rate(["a", "b"], {"a", "b"}), 1)
        self.assertEqual(harvest_rate(["a", "b"], {"other"}), 0)

    def test_empty_download_collection(self):
        self.assertEqual(harvest_rate([], {"a"}), 0.0)
        self.assertEqual(harvest_rate({}, set()), 0.0)

    def test_page_mapping_and_iterable_relevant_urls(self):
        pages = {"a": "<p>Music</p>", "b": "<p>Weather</p>"}
        self.assertEqual(harvest_rate(pages, iter(["a"])), 0.5)

    def test_relevant_urls_not_downloaded_do_not_increase_rate(self):
        self.assertEqual(harvest_rate(["a", "b"], {"a", "missing"}), 0.5)


if __name__ == "__main__":
    unittest.main()
