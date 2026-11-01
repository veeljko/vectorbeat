import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from mrvectorbeat import pipeline
from mrvectorbeat.search.indexes.position_index import PositionIndex
from mrvectorbeat.textprocessing.text_processor import TextProcessor


class SimilarSongsAppTests(unittest.TestCase):
    def test_similar_comparison_persists_and_uses_cosine_with_tfidf_search(self):
        lyrics = ["Love love night dream.", "Love live night.", "Love night night dream.",
                  "Dream dream light love.", "Alone running.", "the and"]
        ids = [2, 7, 13, 19, 23, 29]
        index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics], ids)
        catalog = pd.DataFrame({
            "katalog_id": ids, "track_name": ["Source", "Source Live", "Other", "Third", "Alone", "Empty"],
            "track_artist": ["Artist"] * 6, "lyrics": lyrics,
            "porodica": ["source", "source", "other", "third", "alone", "empty"],
            "godina": [2020] * 6, "playlist_genre": ["pop"] * 6,
        })
        st.cache_data.clear()
        st.cache_resource.clear()
        try:
            with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                    patch.object(pipeline, "ucitaj_indeks", return_value=index), \
                    patch.object(pipeline, "ucitaj_trie", return_value=None), \
                    patch.object(index, "score_similar", wraps=index.score_similar) as scoring:
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                        default_timeout=20).run()
                app.sidebar.radio[0].set_value("Stihovi").run()
                next(r for r in app.radio if r.label == "Rezim pretrage").set_value("Rangirano").run()
                app.text_input[0].set_value("love").run()
                self.assertFalse(app.exception)
                self.assertEqual(next(s for s in app.selectbox if s.label == "Mera slicnosti").value,
                                 "TF-IDF")
                next(b for b in app.button if b.key == "slicne_2").click().run()
                self.assertFalse(app.exception)
                scoring.assert_called_once_with(2, "cosine")
                self.assertFalse(any(b.key == "uporedi_2_7" for b in app.button))
                next(b for b in app.button if b.key == "uporedi_2_13").click().run()
                self.assertFalse(app.exception)
                # Izbor para ne gubi listu slicnih i ne pokrece njeno novo racunanje.
                scoring.assert_called_once()
                self.assertTrue(any(b.key == "uporedi_2_19" for b in app.button))
                self.assertEqual(len(app.dataframe), 1)
                table = app.dataframe[0].value
                self.assertEqual(set(table["Reč / osnova"]), {"love", "night", "dream"})
                self.assertAlmostEqual(table["Udeo ukupne sličnosti (%)"].sum(), 100)
                self.assertGreaterEqual(sum("<mark>" in m.value for m in app.markdown), 2)
                self.assertEqual(len(app.get("column")), 2)
                next(b for b in app.button if b.key == "uporedi_2_19").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["stihovi_poredjenje"], 19)
                next(b for b in app.button if b.key == "slicne_13").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["stihovi_slicne_izvor"], 13)
                self.assertEqual(len(app.dataframe), 0)
                app.text_input[0].set_value("zzzznomatch").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Nema pesama" in info.value for info in app.info))
                self.assertFalse(any(s.value == "Slične pesme" for s in app.subheader))
                app.text_input[0].set_value("running").run()
                next(b for b in app.button if b.key == "slicne_23").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Nema drugih pesama" in info.value for info in app.info))
        finally:
            st.cache_data.clear()
            st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
