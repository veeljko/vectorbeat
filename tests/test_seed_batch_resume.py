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

from mrvectorbeat.crawler.crawler import Crawler, WIKI
from mrvectorbeat.crawler.generate_seed import WikipediaSeedResolver, generate_seed


def article(number, title, **extra):
    return {"pageid": number, "ns": 0, "title": title,
            "description": "American singer", "extract": "",
            "canonicalurl": WIKI + title.replace(" ", "_"), **extra}


class SeedBatchResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "seed.parquet"
        self.progress_path = self.path.with_name("seed.progress.parquet")
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_twenty_three_exact_artists_need_only_three_api_requests(self):
        names = [f"Artist {number:02}" for number in range(23)]

        def fetch(url, **options):
            self.assertFalse(options["use_cache"])
            params = parse_qs(urlsplit(url).query)
            titles = params["titles"][0].strip("\x1f").split("\x1f")
            self.assertLessEqual(len(titles), 10)
            self.assertEqual(params["exintro"], ["1"])
            return 200, json.dumps({"query": {"pages": [
                article(index, title) for index, title in enumerate(titles)
            ]}})

        catalog = pd.DataFrame({"track_artist": names + names[:2]})
        with patch.object(Crawler, "fetch_page", side_effect=fetch) as fetch_mock:
            result = generate_seed(catalog, self.path)
        self.assertEqual(fetch_mock.call_count, 3)
        self.assertEqual(result["artist_name"].tolist(), names)
        self.assertEqual(result["seed"].tolist(), [WIKI + name.replace(" ", "_") for name in names])
        self.assertEqual(list(pd.read_parquet(self.path).columns), ["artist_name", "seed"])
        self.assertFalse(self.progress_path.exists())

    def test_batch_preserves_redirect_identity_and_does_not_repeat_missing_title_lookup(self):
        resolver = WikipediaSeedResolver()
        payload = {
            "redirects": [{"from": "A$AP Mob", "to": "ASAP Mob"},
                          {"from": "A$AP Ferg", "to": "ASAP Ferg"}],
            "pages": [article(1, "ASAP Mob"), article(2, "ASAP Ferg"),
                      {"ns": 0, "title": "Missing", "missing": True}],
        }
        with patch.object(resolver, "_query", side_effect=[payload, {"search": []}, {"search": []}]) as query:
            resolver.prefetch_titles(["A$AP Mob", "A$AP Ferg", "Missing"])
            self.assertEqual(resolver.resolve_artist_page("A$AP Mob"), WIKI + "ASAP_Mob")
            self.assertEqual(resolver.resolve_artist_page("A$AP Ferg"), WIKI + "ASAP_Ferg")
            self.assertIsNone(resolver.resolve_artist_page("Missing"))
        self.assertEqual(query.call_count, 3)
        self.assertTrue(all(call.kwargs.get("list") == "search" for call in query.call_args_list[1:]))

    def test_failure_preserves_final_file_and_resumes_successful_and_missing_artists(self):
        catalog = pd.DataFrame({"track_artist": ["Alpha", "Bravo", "Charlie", "Known"]})
        existing = pd.DataFrame({"artist_name": catalog["track_artist"],
                                 "seed": [None, None, None, WIKI + "Known"]}).astype("string")
        existing.to_parquet(self.path, index=False)
        original = self.path.read_bytes()
        with patch.object(WikipediaSeedResolver, "prefetch_titles"), \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", side_effect=[
                    WIKI + "Alpha", None, RuntimeError("Offline"),
                ]):
            with self.assertRaisesRegex(RuntimeError, "Offline"):
                generate_seed(catalog, self.path, existing=existing)
        self.assertEqual(self.path.read_bytes(), original)
        progress = pd.read_parquet(self.progress_path).set_index("artist_name")["seed"]
        self.assertEqual(progress["Alpha"], WIKI + "Alpha")
        self.assertTrue(pd.isna(progress["Bravo"]))
        self.assertNotIn("Charlie", progress)
        with patch.object(WikipediaSeedResolver, "prefetch_titles") as prefetch, \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", return_value=WIKI + "Charlie") as resolve:
            result = generate_seed(catalog, self.path, existing=existing)
        prefetch.assert_called_once_with(["Charlie"])
        self.assertEqual(resolve.call_args.args, ("Charlie",))
        self.assertEqual(result["seed"].dropna().tolist(), [WIKI + "Alpha", WIKI + "Charlie", WIKI + "Known"])
        self.assertFalse(self.progress_path.exists())

    def test_ctrl_c_saves_completed_artists_without_creating_partial_final_artifact(self):
        catalog = pd.DataFrame({"track_artist": ["Alpha", "Bravo"]})
        with patch.object(WikipediaSeedResolver, "prefetch_titles"), \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", side_effect=[WIKI + "Alpha", KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                generate_seed(catalog, self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(pd.read_parquet(self.progress_path)["artist_name"].tolist(), ["Alpha"])
        with patch.object(WikipediaSeedResolver, "prefetch_titles") as prefetch, \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", return_value=WIKI + "Bravo"):
            generate_seed(catalog, self.path)
        prefetch.assert_called_once_with(["Bravo"])
        self.assertTrue(self.path.exists())

    def test_completed_batch_is_on_disk_before_starting_next_api_batch(self):
        names = [f"Artist {number:02}" for number in range(11)]

        def prefetch(batch):
            if names[-1] in batch:
                self.assertEqual(len(pd.read_parquet(self.progress_path)), 10)
                raise RuntimeError("Second batch unavailable")

        with patch.object(WikipediaSeedResolver, "prefetch_titles", side_effect=prefetch), \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", side_effect=lambda name, **kwargs: WIKI + name):
            with self.assertRaises(RuntimeError):
                generate_seed(pd.DataFrame({"track_artist": names}), self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(len(pd.read_parquet(self.progress_path)), 10)

    def test_older_progress_does_not_override_newer_final_artifact(self):
        existing = pd.DataFrame({"artist_name": ["Known", "Missing"],
                                 "seed": [WIKI + "Known", None]}).astype("string")
        existing.to_parquet(self.path, index=False)
        before = self.path.stat().st_mtime_ns
        stale = existing.copy()
        stale["seed"] = [WIKI + "Wrong", WIKI + "Wrong_2"]
        stale.to_parquet(self.progress_path, index=False)
        os.utime(self.progress_path, ns=(before - 1_000_000_000, before - 1_000_000_000))
        with patch.object(WikipediaSeedResolver, "prefetch_titles") as prefetch, \
                patch.object(WikipediaSeedResolver, "resolve_artist_page", return_value=None):
            result = generate_seed(pd.DataFrame({"track_artist": ["Known", "Missing"]}), self.path, existing=existing)
        prefetch.assert_called_once_with(["Missing"])
        pd.testing.assert_frame_equal(result, existing)
        self.assertEqual(self.path.stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()
