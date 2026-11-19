import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from mrvectorbeat import pipeline
from mrvectorbeat.crawler.crawler import Crawler, WIKI
from mrvectorbeat.crawler.generate_seed import WikipediaSeedResolver, generate_seed


def article(page_id, title, description, text=""):
    return {"pageid": page_id, "ns": 0, "title": title, "description": description,
            "extract": text, "canonicalurl": WIKI + title.replace(" ", "_")}


class SeedResolverTests(unittest.TestCase):
    def test_selects_band_instead_of_city_and_returns_canonical_url(self):
        pages = [article(1, "Boston", "City in Massachusetts", "Boston has music and bands."),
                 article(2, "Boston (band)", "American rock band")]
        resolver = WikipediaSeedResolver()
        with patch.object(resolver, "_search", return_value=pages):
            self.assertEqual(resolver.resolve_artist_page("Boston"), WIKI + "Boston_(band)")

    def test_rejects_disambiguation_albums_and_other_topics(self):
        disambiguation = article(1, "Queen", "")
        disambiguation["pageprops"] = {"disambiguation": ""}
        pages = [disambiguation,
                 article(2, "Queen", "Album by an English rock band", "Queen is an album by a band."),
                 article(3, "Queen", "City", "Queen is a city with many singers."),
                 {"ns": 0, "title": "Queen", "missing": True},
                 {"ns": 0, "title": "Queen", "invalid": True}]
        resolver = WikipediaSeedResolver()
        with patch.object(resolver, "_search", return_value=pages):
            self.assertIsNone(resolver.resolve_artist_page("Queen"))

    def test_catalog_evidence_distinguishes_bands_with_the_same_name(self):
        pages = [article(1, "Rainbow (rock band)", "British rock band"),
                 article(2, "Rainbow (girl group)", "South Korean girl group",
                         "Rainbow is a South Korean girl group.")]
        resolver = WikipediaSeedResolver()
        with patch.object(resolver, "_search", return_value=pages), \
                patch.object(resolver, "_query", side_effect=[
                    {"pages": [{"pageid": 1, "extract": "Their album Rising includes Stargazer."}]},
                    {"pages": [{"pageid": 2, "extract": "The group released Rainbow Syndrome."}]},
                ]) as query:
            self.assertEqual(resolver.resolve_artist_page("Rainbow", albums=["Rising"],
                                                         songs=["Stargazer"]),
                             WIKI + "Rainbow_(rock_band)")
        self.assertTrue(all(call.kwargs["exlimit"] == 1 for call in query.call_args_list))

    def test_ambiguous_matches_stay_unresolved_and_full_text_is_cached(self):
        pages = [article(1, "Example (band)", "Rock band"),
                 article(2, "Example (singer)", "Pop singer")]
        resolver = WikipediaSeedResolver()
        with patch.object(resolver, "_search", return_value=pages), \
                patch.object(resolver, "_query", side_effect=[
                    {"pages": [{"pageid": 1, "extract": "A different album."}]},
                    {"pages": [{"pageid": 2, "extract": "A different album."}]},
                ]) as query:
            self.assertIsNone(resolver.resolve_artist_page("Example", songs=["Missing Song"]))
        self.assertEqual(query.call_count, 2)

    def test_strips_forbidden_brackets_for_search_preserving_artist_identity(self):
        resolver = WikipediaSeedResolver()
        with patch.object(resolver, "_search", return_value=[
            article(1, "SITD", "German electronic music band"),
        ]) as search:
            self.assertEqual(resolver.resolve_artist_page("[:SITD:]"), WIKI + "SITD")
        self.assertNotIn("[", search.call_args.args[0])

    def test_country_singer_is_an_artist_and_prefix_is_not_an_alias(self):
        self.assertTrue(WikipediaSeedResolver._candidate(
            article(1, "Johnny Cash", "American country singer"), "Johnny Cash"))
        self.assertFalse(WikipediaSeedResolver._candidate(
            article(2, "Rainbow Kitten Surprise", "American alternative rock band",
                    "Rainbow Kitten Surprise is an American rock band."), "Rainbow"))

    def test_single_musician_without_matching_catalog_evidence_stays_unresolved(self):
        resolver = WikipediaSeedResolver()
        pages = [article(1, "Example (singer)", "American singer")]
        with patch.object(resolver, "_search", return_value=pages), \
                patch.object(resolver, "_query", return_value={
                    "pages": [{"pageid": 1, "extract": "A different artist's discography."}],
                }):
            self.assertIsNone(resolver.resolve_artist_page("Example", songs=["Catalog Song"]))

    def test_album_versions_are_removed_before_matching_catalog_evidence(self):
        resolver = WikipediaSeedResolver()
        pages = [article(1, "Rainbow (rock band)", "British rock band")]
        with patch.object(resolver, "_search", return_value=pages), \
                patch.object(resolver, "_query", return_value={
                    "pages": [{"pageid": 1, "extract": "Rainbow released Rising in 1976."}],
                }):
            self.assertEqual(resolver.resolve_artist_page("Rainbow", albums=["Rising (Remastered)"]),
                             WIKI + "Rainbow_(rock_band)")

    def test_title_search_recovers_an_artist_crowded_out_by_other_topics(self):
        resolver = WikipediaSeedResolver()
        with patch.object(resolver, "_search", side_effect=[
            [article(1, "Rainbow", "Meteorological phenomenon")],
            [article(2, "Rainbow (rock band)", "British rock band")],
        ]) as search, patch.object(resolver, "_query", return_value={
            "pages": [{"pageid": 2, "extract": "The band recorded Since You Been Gone."}],
        }):
            self.assertEqual(resolver.resolve_artist_page("Rainbow", songs=["Since You Been Gone"]),
                             WIKI + "Rainbow_(rock_band)")
        self.assertIn('intitle:"Rainbow"', search.call_args.args[0])

    def test_search_limits_namespace_and_resolves_redirects(self):
        resolver = WikipediaSeedResolver()
        with patch.object(Crawler, "fetch_page", side_effect=[
            (200, json.dumps({"query": {"search": [{"title": "KARD"}]}})),
            (200, json.dumps({"query": {"pages": [article(7, "Kard (group)", "South Korean pop group")]}})),
        ]) as fetch:
            self.assertEqual(resolver.resolve_artist_page("KARD"), WIKI + "Kard_(group)")
        search = parse_qs(urlsplit(fetch.call_args_list[0].args[0]).query)
        details = parse_qs(urlsplit(fetch.call_args_list[1].args[0]).query)
        self.assertEqual(search["srnamespace"], ["0"])
        self.assertEqual(details["redirects"], ["1"])

    def test_network_failure_aborts_without_saving_false_missing_results(self):
        catalog = pd.DataFrame({"track_artist": ["Example"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.parquet"
            with patch.object(WikipediaSeedResolver, "resolve_artist_page",
                              side_effect=RuntimeError("Network unavailable")):
                with self.assertRaises(RuntimeError):
                    generate_seed(catalog, path)
            self.assertFalse(path.exists())

    def test_generation_deduplicates_artists_and_saves_only_two_columns(self):
        catalog = pd.DataFrame({"track_artist": ["Boston", "Boston", "Unknown"],
                                "track_name": ["More Than a Feeling", "Peace of Mind", "Unknown Song"],
                                "track_album_name": ["Boston", "Boston", "Unknown Album"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.parquet"
            with patch.object(WikipediaSeedResolver, "resolve_artist_page",
                              side_effect=[WIKI + "Boston_(band)", None]) as resolve, \
                    contextlib.redirect_stdout(io.StringIO()):
                built = generate_seed(catalog, path)
            loaded = pd.read_parquet(path)
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(list(loaded.columns), ["artist_name", "seed"])
        pd.testing.assert_frame_equal(loaded, built)
        self.assertTrue(pd.isna(loaded.loc[1, "seed"]))


class SeedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        paths = patch.multiple(pipeline, SEED_PATH=root / "seed.parquet",
                               GRAF_PATH=root / "graf.pkl", ARTIST_NODES_PATH=root / "artist_nodes.pkl")
        paths.start()
        self.addCleanup(paths.stop)
        self.seeds = pd.DataFrame({"artist_name": ["Boston", "Missing"],
                                   "seed": [WIKI + "Boston_(band)", None]}).astype("string")

    def test_existing_seeds_are_loaded_without_catalog_or_network_access(self):
        self.seeds.to_parquet(pipeline.SEED_PATH, index=False)
        with patch.object(pipeline, "ucitaj_katalog", side_effect=AssertionError("Catalog read")), \
                patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP request")):
            pd.testing.assert_frame_equal(pipeline.napravi_seed(), self.seeds)

    def test_missing_seeds_are_generated_and_subsequently_reused(self):
        catalog = pd.DataFrame({"track_artist": ["Boston"]})
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", return_value=WIKI + "Boston_(band)"), \
                contextlib.redirect_stdout(io.StringIO()):
            built = pipeline.napravi_seed()
        self.assertTrue(pipeline.SEED_PATH.exists())
        with patch.object(WikipediaSeedResolver, "resolve_artist_page", side_effect=AssertionError("Search")):
            pd.testing.assert_frame_equal(pipeline.napravi_seed(), built)

    def test_graph_build_uses_resolved_seeds_skips_nulls_and_then_only_reads(self):
        self.seeds.to_parquet(pipeline.SEED_PATH, index=False)
        seed = WIKI + "Boston_(band)"

        def crawl(artist, graph, **options):
            self.assertEqual(options["seed"], seed)
            graph[seed] = []
            return graph

        with patch.object(Crawler, "crawl_into_graph", side_effect=crawl) as crawl_mock, \
                contextlib.redirect_stdout(io.StringIO()):
            built = pipeline.napravi_graf(max_pages=3)
        self.assertEqual(crawl_mock.call_count, 1)
        self.assertEqual(built, ({seed: []}, {"Boston": seed}))
        self.assertTrue(pipeline.GRAF_PATH.exists())
        self.assertTrue(pipeline.ARTIST_NODES_PATH.exists())
        with patch.object(pipeline, "napravi_seed", side_effect=AssertionError("Seeds rebuilt")), \
                patch.object(Crawler, "crawl_into_graph", side_effect=AssertionError("Crawl")):
            self.assertEqual(pipeline.ucitaj_graf(), built)

    def test_missing_artist_lookup_is_rebuilt_without_recrawling_graph(self):
        self.seeds.to_parquet(pipeline.SEED_PATH, index=False)
        seed = WIKI + "Boston_(band)"
        pipeline._snimi_pickle({seed: []}, pipeline.GRAF_PATH)
        with patch.object(Crawler, "crawl_into_graph", side_effect=AssertionError("Crawl")):
            self.assertEqual(pipeline.napravi_graf(), ({seed: []}, {"Boston": seed}))

    def test_loading_missing_graph_raises_without_starting_preparation(self):
        with patch.object(pipeline, "napravi_seed", side_effect=AssertionError("Seeds rebuilt")), \
                patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP request")):
            with self.assertRaisesRegex(FileNotFoundError, "Graf nije pripremljen.*python -m scripts.pripremi"):
                pipeline.ucitaj_graf()
        self.assertFalse(pipeline.GRAF_PATH.exists())

    def test_loading_missing_artist_lookup_raises_without_rebuilding_it(self):
        pipeline._snimi_pickle({WIKI + "Boston_(band)": []}, pipeline.GRAF_PATH)
        with patch.object(pipeline, "napravi_seed", side_effect=AssertionError("Seeds rebuilt")), \
                patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP request")):
            with self.assertRaisesRegex(FileNotFoundError, "Mapa izvodjaca nije pripremljena.*python -m scripts.pripremi"):
                pipeline.ucitaj_graf()
        self.assertFalse(pipeline.ARTIST_NODES_PATH.exists())

    def test_loading_missing_seeds_shows_preparation_command(self):
        with patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP request")):
            with self.assertRaisesRegex(FileNotFoundError, "Seedovi nisu pripremljeni.*python -m scripts.pripremi"):
                pipeline.ucitaj_seed()

    def test_crawler_reads_prepared_seed_and_never_guesses_for_unresolved_artist(self):
        self.seeds.to_parquet(pipeline.SEED_PATH, index=False)
        seed = WIKI + "Boston_(band)"
        crawler = Crawler()
        with patch.object(Crawler, "fetch_page", side_effect=[
            (200, "User-agent: *"), (200, "<p>Rock band music.</p>"),
        ]) as fetch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(crawler.focused_crawl("Boston", max_pages=1),
                             {"Boston": {seed: "<p>Rock band music.</p>"}})
        self.assertEqual(fetch.call_args_list[1].args[0], seed)
        with patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP request")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(crawler.focused_crawl("Missing"), {"Missing": {}})
        self.assertEqual(crawler.build_artist_nodes(["Boston", "Missing"], {seed: []}), {"Boston": seed})

    def test_zero_page_limit_needs_neither_artifact_nor_http(self):
        with patch.object(Crawler, "_artist_seeds", side_effect=AssertionError("Artifact read")), \
                patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP request")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(Crawler().focused_crawl("Boston", max_pages=0), {"Boston": {}})


if __name__ == "__main__":
    unittest.main()
