import contextlib
import io
import json
import os
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
    def test_stylized_names_use_verified_redirects_without_discography_matches(self):
        for original, canonical, description in (
            ("A$AP Ferg", "ASAP Ferg", "American rapper"),
            ("A$AP Mob", "ASAP Mob", "American hip hop collective"),
            ("Ke$ha", "Kesha", "American singer"),
            ("P!nk", "Pink", "American singer"),
        ):
            with self.subTest(artist=original):
                page = article(1, canonical, description)
                payload = {"query": {"redirects": [{"from": original, "to": canonical}],
                                     "pages": [page]}}
                with patch.object(Crawler, "fetch_page", return_value=(200, json.dumps(payload))) as fetch:
                    seed = WikipediaSeedResolver().resolve_artist_page(
                        original, songs=["Song not listed in the article"],
                    )
                self.assertEqual(seed, WIKI + canonical.replace(" ", "_"))
                fetch.assert_called_once()

    def test_normalization_and_redirect_chains_keep_aliases_with_their_own_target(self):
        ferg = article(1, "ASAP Ferg", "American rapper")
        mob = article(2, "ASAP Mob", "American hip hop collective")
        payload = {
            "normalized": [{"from": "A$AP_Ferg", "to": "A$AP Ferg"}],
            "redirects": [{"from": "A$AP Ferg", "to": "ASAP Ferg"},
                          {"from": "A$AP Mob", "to": "ASAP Mob"}],
            "pages": [ferg, mob],
        }
        resolver = WikipediaSeedResolver()
        with patch.object(resolver, "_query", return_value=payload):
            pages = resolver._pages(["A$AP_Ferg", "A$AP Mob"])
        self.assertTrue(resolver._candidate(pages[0], "A$AP Ferg"))
        self.assertTrue(resolver._candidate(pages[1], "A$AP Mob"))
        self.assertFalse(resolver._candidate(pages[0], "A$AP Mob"))
        self.assertFalse(resolver._candidate(pages[1], "A$AP Ferg"))

    def test_verified_redirect_still_requires_an_artist_article(self):
        for page in (
            article(1, "Boston", "City in Massachusetts"),
            article(2, "List of musicians", "List of musicians"),
        ):
            page["title_aliases"] = ["Catalog Artist"]
            self.assertFalse(WikipediaSeedResolver._candidate(page, "Catalog Artist"))

    def test_industry_association_or_label_cannot_be_an_artist_even_when_it_mentions_performers(self):
        for title, description, introduction in (
            ("Australian Recording Industry Association", "Australian recording industry association",
             "The Australian Recording Industry Association (ARIA) is a trade association representing musicians and bands."),
            ("Australian Recording Industry Association", "",
             "The Australian Recording Industry Association (ARIA) is a trade association representing musicians and bands."),
            ("Example Records", "American record label",
             "Example Records is a record label founded by a musician and representing singers."),
            ("Aria", "Musical form",
             "An aria is a musical composition performed by singers."),
        ):
            with self.subTest(title=title, description=description):
                page = article(1, title, description, introduction)
                page["title_aliases"] = ["ARIA"]
                self.assertFalse(WikipediaSeedResolver._candidate(page, "ARIA"))

    def test_artist_who_owns_a_label_and_band_named_record_company_remain_valid(self):
        pages = [
            article(1, "Example", "American singer and record label owner",
                    "Example is an American singer who founded a record label."),
            article(2, "The Record Company", "American rock band",
                    "The Record Company is an American rock band."),
            article(3, "The Record Company", "",
                    "The Record Company is an American rock band."),
        ]
        for page in pages:
            with self.subTest(page=page):
                self.assertTrue(WikipediaSeedResolver._candidate(page, page["title"]))

    def test_aria_association_redirect_cannot_resolve_the_catalog_recording_artist(self):
        page = article(1, "Australian Recording Industry Association", "Australian trade association",
                       "ARIA is a trade association representing musicians and record producers.")
        page["title_aliases"] = ["ARIA"]
        resolver = WikipediaSeedResolver()
        with patch.object(resolver, "_search", return_value=[page]), \
                patch.object(resolver, "_query", side_effect=AssertionError("Non-artist discography inspected")):
            self.assertIsNone(resolver.resolve_artist_page(
                "ARIA", albums=["Bleu Chanel (Na-No Remix)"], songs=["Bleu Chanel (Na-No Remix)"],
            ))

    def test_exact_artist_article_does_not_require_complete_discography(self):
        page = article(83151929, "Anja Kotar", "Slovenian singer-songwriter",
                       "Anja Kotar is a Slovenian singer-songwriter based in California.")
        with patch.object(Crawler, "fetch_page", return_value=(
            200, json.dumps({"query": {"pages": [page]}}),
        )) as fetch:
            seed = WikipediaSeedResolver().resolve_artist_page(
                "Anja Kotar", albums=["House Party"], songs=["House Party"],
            )
        self.assertEqual(seed, WIKI + "Anja_Kotar")
        fetch.assert_called_once()
        parameters = parse_qs(urlsplit(fetch.call_args.args[0]).query)
        self.assertIn("Anja Kotar", parameters["titles"][0])
        self.assertNotIn("list", parameters)

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
            (200, json.dumps({"query": {"pages": [{"ns": 0, "title": "KARD", "missing": True}]}})),
            (200, json.dumps({"query": {"search": [{"title": "KARD"}]}})),
            (200, json.dumps({"query": {"pages": [article(7, "Kard (group)", "South Korean pop group")]}})),
        ]) as fetch:
            self.assertEqual(resolver.resolve_artist_page("KARD"), WIKI + "Kard_(group)")
        search = parse_qs(urlsplit(fetch.call_args_list[1].args[0]).query)
        details = parse_qs(urlsplit(fetch.call_args_list[2].args[0]).query)
        self.assertEqual(search["srnamespace"], ["0"])
        self.assertEqual(details["redirects"], ["1"])

    def test_network_failure_aborts_without_saving_false_missing_results(self):
        catalog = pd.DataFrame({"track_artist": ["Example"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.parquet"
            with patch.object(WikipediaSeedResolver, "resolve_artist_page",
                              side_effect=RuntimeError("Network unavailable")), \
                    patch.object(WikipediaSeedResolver, "prefetch_titles"):
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
                    patch.object(WikipediaSeedResolver, "prefetch_titles"), \
                    contextlib.redirect_stdout(io.StringIO()):
                built = generate_seed(catalog, path)
            loaded = pd.read_parquet(path)
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(list(loaded.columns), ["artist_name", "seed"])
        pd.testing.assert_frame_equal(loaded, built)
        self.assertTrue(pd.isna(loaded.loc[1, "seed"]))


class SeedPipelineTests(unittest.TestCase):
    def setUp(self):
        prefetch = patch.object(WikipediaSeedResolver, "prefetch_titles")
        prefetch.start()
        self.addCleanup(prefetch.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        paths = patch.multiple(pipeline, SEED_PATH=root / "seed.parquet",
                               GRAF_PATH=root / "graf.pkl", ARTIST_NODES_PATH=root / "artist_nodes.pkl")
        paths.start()
        self.addCleanup(paths.stop)
        cache = patch("mrvectorbeat.crawler.crawler.PAGE_CACHE_DIR", root / "pages")
        cache.start()
        self.addCleanup(cache.stop)
        self.seeds = pd.DataFrame({"artist_name": ["Boston", "Missing"],
                                   "seed": [WIKI + "Boston_(band)", None]}).astype("string")

    def test_existing_seeds_are_loaded_without_catalog_or_network_access(self):
        self.seeds.to_parquet(pipeline.SEED_PATH, index=False)
        with patch.object(pipeline, "ucitaj_katalog", side_effect=AssertionError("Catalog read")), \
                patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP request")):
            pd.testing.assert_frame_equal(pipeline.napravi_seed(), self.seeds)

    def test_interrupted_retry_resumes_even_without_retry_flag(self):
        seeds = pd.DataFrame({"artist_name": ["Anja Kotar", "Boston", "Missing"],
                              "seed": [None, WIKI + "Boston_(band)", None]}).astype("string")
        seeds.to_parquet(pipeline.SEED_PATH, index=False)
        progress = seeds[seeds["artist_name"].ne("Anja Kotar")]
        progress.to_parquet(pipeline.SEED_PATH.with_name("seed.progress.parquet"), index=False)
        catalog = pd.DataFrame({"track_artist": seeds["artist_name"]})
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", return_value=WIKI + "Anja_Kotar") as resolve, \
                contextlib.redirect_stdout(io.StringIO()):
            result = pipeline.napravi_seed()
        self.assertEqual(resolve.call_args.args, ("Anja Kotar",))
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(result["seed"].iloc[0], WIKI + "Anja_Kotar")

    def test_missing_seeds_are_generated_and_subsequently_reused(self):
        catalog = pd.DataFrame({"track_artist": ["Boston"]})
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", return_value=WIKI + "Boston_(band)"), \
                contextlib.redirect_stdout(io.StringIO()):
            built = pipeline.napravi_seed()
        self.assertTrue(pipeline.SEED_PATH.exists())
        with patch.object(WikipediaSeedResolver, "resolve_artist_page", side_effect=AssertionError("Search")):
            pd.testing.assert_frame_equal(pipeline.napravi_seed(), built)

    def test_retry_unresolved_preserves_resolved_artists_and_two_column_format(self):
        seeds = pd.DataFrame({"artist_name": ["Anja Kotar", "Boston"],
                              "seed": [None, WIKI + "Boston_(band)"]}).astype("string")
        seeds.to_parquet(pipeline.SEED_PATH, index=False)
        catalog = pd.DataFrame({"track_artist": ["Anja Kotar", "Boston"],
                                "track_name": ["House Party", "Peace of Mind"]})
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(WikipediaSeedResolver, "resolve_artist_page",
                             return_value=WIKI + "Anja_Kotar") as resolve, \
                contextlib.redirect_stdout(io.StringIO()):
            updated = pipeline.napravi_seed(retry_unresolved=True)
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(resolve.call_args.args, ("Anja Kotar",))
        self.assertEqual(updated["seed"].tolist(), [WIKI + "Anja_Kotar", WIKI + "Boston_(band)"])
        self.assertEqual(list(updated.columns), ["artist_name", "seed"])
        pd.testing.assert_frame_equal(pipeline.ucitaj_seed(), updated)

    def test_unchanged_unresolved_seeds_do_not_invalidate_graph_artifact(self):
        self.seeds.to_parquet(pipeline.SEED_PATH, index=False)
        before = pipeline.SEED_PATH.stat().st_mtime_ns
        catalog = pd.DataFrame({"track_artist": ["Boston", "Missing"]})
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.napravi_seed(retry_unresolved=True)
        self.assertEqual(pipeline.SEED_PATH.stat().st_mtime_ns, before)

    def test_updated_seeds_extend_existing_graph_instead_of_returning_stale_lookup(self):
        old = WIKI + "Boston_(band)"
        new = WIKI + "Anja_Kotar"
        seeds = pd.DataFrame({"artist_name": ["Anja Kotar", "Boston"],
                              "seed": [new, old]}).astype("string")
        pipeline._snimi_pickle({old: []}, pipeline.GRAF_PATH)
        pipeline._snimi_pickle({"Boston": old}, pipeline.ARTIST_NODES_PATH)
        seeds.to_parquet(pipeline.SEED_PATH, index=False)
        # A deterministic newer seed timestamp avoids filesystem clock granularity.
        outdated = pipeline.SEED_PATH.stat().st_mtime_ns - 1_000_000_000
        for path in (pipeline.GRAF_PATH, pipeline.ARTIST_NODES_PATH):
            os.utime(path, ns=(outdated, outdated))

        def crawl(artist, graph, **options):
            graph.setdefault(options["seed"], [])
            Crawler._save_cached_page(options["seed"], "<p>Music artist</p>")
            return graph

        with patch.object(Crawler, "crawl_into_graph", side_effect=crawl), \
                contextlib.redirect_stdout(io.StringIO()):
            graph, nodes = pipeline.napravi_graf()
        self.assertEqual(graph, {old: [], new: []})
        self.assertEqual(nodes, {"Anja Kotar": new, "Boston": old})

    def test_graph_build_uses_resolved_seeds_skips_nulls_and_then_only_reads(self):
        self.seeds.to_parquet(pipeline.SEED_PATH, index=False)
        seed = WIKI + "Boston_(band)"

        def crawl(artist, graph, **options):
            self.assertEqual(options["seed"], seed)
            graph[seed] = []
            Crawler._save_cached_page(seed, "<p>Music artist</p>")
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
            (200, "User-agent: *"), (200, "<p>Rock band music.</p>", seed),
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
