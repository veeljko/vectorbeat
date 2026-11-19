import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from mrvectorbeat import pipeline
from mrvectorbeat.crawler.crawler import Crawler
from mrvectorbeat.search.ranking.similar_artists import build_similarity_index


WIKI = "https://en.wikipedia.org/wiki/"


class SimilarArtistsAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.addCleanup(st.cache_data.clear)
        self.addCleanup(st.cache_resource.clear)
        patcher = patch.object(pipeline, "ucitaj_rangove", side_effect=FileNotFoundError("PageRank not prepared"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15)
        app.session_state["strana"] = "Izvodjaci"
        return app

    def test_selection_displays_ordered_similar_artists_with_common_page_links(self):
        a, b, c, d = [WIKI + title for title in ("Alpha", "Beta", "Gamma", "Isolated")]
        genre, label = WIKI + "Hip_hop_music", WIKI + "Def_Jam_Recordings"
        index = build_similarity_index({genre: [a, b, c], label: [a, b], d: []},
                                       {"Alpha": a, "Alpha alias": a, "Beta": b, "Gamma": c, "Isolated": d})
        with patch.object(pipeline, "ucitaj_slicne_izvodjace", return_value=index) as load, \
                patch.object(pipeline, "ucitaj_graf", side_effect=AssertionError("Graph read")), \
                patch.object(pipeline, "napravi_slicne_izvodjace", side_effect=AssertionError("Build")), \
                patch.object(Crawler, "fetch_page", side_effect=AssertionError("HTTP")):
            app = self.app().run()
            self.assertFalse(app.exception)
            app.selectbox(key="slicni_izvodjac").select("Alpha alias").run()
            self.assertFalse(app.exception)
            results = [item.value for item in app.markdown if item.value.startswith("**")]
            self.assertEqual(results, ["**1. Beta**", "**2. Gamma**"])
            self.assertTrue(any("0.833333" in item.value for item in app.caption))
            links = [item.value for item in app.markdown if item.value.startswith("Oboje se pominju u:")]
            self.assertEqual(len(links), 2)
            self.assertIn(f"[Hip hop music](<{genre}>)", links[0])
            self.assertIn(f"[Def Jam Recordings](<{label}>)", links[0])
            app.selectbox(key="slicni_izvodjac").select("Isolated").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Nema drugih izvođača" in item.value for item in app.info))
            load.assert_called_once()

    def test_missing_similarity_artifact_shows_preparation_command(self):
        with patch.object(pipeline, "ucitaj_slicne_izvodjace", side_effect=FileNotFoundError(
            "Indeks nije pripremljen. Pokreni: python -m scripts.pripremi --samo-slicni-izvodjaci")), \
                patch.object(pipeline, "napravi_slicne_izvodjace", side_effect=AssertionError("Build")):
            app = self.app().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("--samo-slicni-izvodjaci" in item.value for item in app.info))

    def test_empty_similarity_index_is_handled(self):
        with patch.object(pipeline, "ucitaj_slicne_izvodjace", return_value=build_similarity_index({}, {})):
            app = self.app().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("U grafu još nema" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
