from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from mrvectorbeat import pipeline
from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.search.ranking.qlm import QLM
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class SongFeedbackAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.addCleanup(st.cache_data.clear)
        self.addCleanup(st.cache_resource.clear)

    def test_marks_are_exclusive_removable_and_survive_pagination_and_query_changes(self):
        ids = list(range(100, 112))
        lyrics = ["love night music"] * len(ids)
        index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics], ids)
        catalog = pd.DataFrame({
            "katalog_id": ids, "track_name": [f"Song {doc_id}" for doc_id in ids],
            "track_artist": ["Artist"] * len(ids), "lyrics": lyrics,
            "porodica": [f"song_{doc_id}" for doc_id in ids],
            "godina": [2020] * len(ids), "playlist_genre": ["pop"] * len(ids),
        })
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(pipeline, "ucitaj_indeks", return_value=index), \
                patch.object(pipeline, "ucitaj_trie", return_value=None), \
                patch.object(QLM, "rank", autospec=True, side_effect=QLM.rank) as rank:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
            app.session_state["strana"] = "Jezicki model"
            app.run()
            app.text_input(key="jezicki_upit").set_value("love").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.pills), 10)
            self.assertEqual(app.pills[0].options, ["Dopada mi se", "Ne dopada mi se"])
            self.assertTrue(all(pill.value is None for pill in app.pills))
            rank.assert_called_once()

            app.pills[0].set_value("like").run()
            app.pills[1].set_value("dislike").run()
            app.pills[2].set_value("like").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["ocene_pesama"][("jezicki", "love")],
                             {100: "like", 101: "dislike", 102: "like"})
            # Switching the same song replaces its previous mark.
            app.pills[0].set_value("dislike").run()
            self.assertEqual(app.session_state["ocene_pesama"][("jezicki", "love")][100], "dislike")
            # Clearing either kind of mark removes the stored judgment.
            app.pills[0].set_value(None).run()
            app.pills[2].set_value(None).run()
            self.assertEqual(app.session_state["ocene_pesama"][("jezicki", "love")], {101: "dislike"})

            app.number_input(key="jezicki_strana").set_value(2).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.pills), 2)
            app.pills[0].set_value("like").run()
            app.number_input(key="jezicki_strana").set_value(1).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.pills[1].value, "dislike")
            self.assertIsNone(app.pills[0].value)
            app.selectbox(key="jezicki_velicina").set_value(20).run()
            self.assertEqual(len(app.pills), 12)
            self.assertEqual(app.pills[10].value, "like")
            self.assertEqual(app.session_state["ocene_pesama"][("jezicki", "love")],
                             {101: "dislike", 110: "like"})
            rank.assert_called_once()  # Feedback does not recompute or alter ranking.

            app.text_input(key="jezicki_upit").set_value("night").run()
            self.assertFalse(app.exception)
            self.assertTrue(all(pill.value is None for pill in app.pills))
            app.pills[0].set_value("like").run()
            self.assertEqual(app.session_state["ocene_pesama"][("jezicki", "night")], {100: "like"})
            app.text_input(key="jezicki_upit").set_value("").run()
            self.assertEqual(len(app.pills), 0)
            app.text_input(key="jezicki_upit").set_value("love").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.pills[1].value, "dislike")
            app.selectbox(key="jezicki_velicina").set_value(20).run()
            self.assertEqual(app.pills[10].value, "like")

            app.sidebar.radio[0].set_value("Stihovi").run()
            app.text_input(key="stihovi_upit").set_value("love").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.pills), 10)
            self.assertTrue(all(pill.value is None for pill in app.pills))
            app.pills[0].set_value("like").run()
            app.pills[0].set_value(None).run()
            self.assertFalse(app.exception)
            stihovi_context = ("stihovi", "love", "boolean", "AND", 3, "tfidf", 1.5, 0.75)
            self.assertEqual(app.session_state["ocene_pesama"][stihovi_context], {})
            self.assertEqual(app.session_state["ocene_pesama"][("jezicki", "love")],
                             {101: "dislike", 110: "like"})


if __name__ == "__main__":
    unittest.main()
