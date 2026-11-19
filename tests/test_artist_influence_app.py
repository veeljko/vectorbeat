import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from mrvectorbeat import pipeline
from mrvectorbeat.search.ranking.artist_influence import build_artist_influence


class ArtistInfluenceAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.addCleanup(st.cache_data.clear)
        self.addCleanup(st.cache_resource.clear)

    def app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15)
        app.session_state["strana"] = "Izvodjaci"
        return app

    def test_artists_page_reads_prepared_scores_and_displays_incoming_counts(self):
        artifact = build_artist_influence(
            {"a": [], "b": [], "genre": ["a"], "album": ["genre"], "other": ["b"]},
            {"A$AP Alpha": "a", "Beta": "b", "Not catalog": "genre"}, ["A$AP Alpha", "Beta"])
        with patch.object(pipeline, "ucitaj_rangove", return_value=artifact) as load, \
                patch.object(pipeline, "napravi_rangove", side_effect=AssertionError("Build")), \
                patch.object(pipeline, "ucitaj_katalog", side_effect=AssertionError("Catalog")), \
                patch.object(pipeline, "ucitaj_graf", side_effect=AssertionError("Graph")), \
                patch("mrvectorbeat.search.ranking.artist_influence.pagerank", side_effect=AssertionError("PageRank")):
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, "Najuticajniji izvođači")
            table = app.dataframe[0].value
            self.assertEqual(table["Izvođač"].tolist(), ["A$AP Alpha", "Beta"])
            self.assertEqual(table["Ulazne veze"].tolist(), [1, 1])
            self.assertEqual(table["Uticaj (PageRank)"].tolist(), artifact["artists"]["pagerank"].tolist())
            app.text_input(key="uticaj_izvodjac").set_value("$").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value["Izvođač"].tolist(), ["A$AP Alpha"])
            load.assert_called_once()

    def test_missing_artifact_displays_error_without_building(self):
        with patch.object(pipeline, "ucitaj_rangove", side_effect=FileNotFoundError(
            "Uticaj izvodjaca nije pripremljen. Pokreni: python -m scripts.pripremi --samo-rangovi")), \
                patch.object(pipeline, "napravi_rangove", side_effect=AssertionError("Build")):
            app = self.app().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("--samo-rangovi" in entry.value for entry in app.info))


if __name__ == "__main__":
    unittest.main()
