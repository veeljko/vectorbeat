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


class PlaylistAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.addCleanup(st.cache_data.clear)
        self.addCleanup(st.cache_resource.clear)
        lyrics = ["love guitar guitar stage", "love beach ocean sun"]
        lyrics += ["love guitar"] * 8 + ["guitar stage riff"] * 13
        lyrics += ["beach ocean sun"] * 12 + ["unrelated ambient"]
        ids = list(range(len(lyrics)))
        families = [f"song_{doc}" for doc in ids]
        families[10] = families[0]
        families[25] = families[24]
        self.index = PositionIndex([TextProcessor(text).text_tokenize() for text in lyrics], ids)
        self.catalog = pd.DataFrame({
            "katalog_id": ids, "track_name": [f"Song {doc}" for doc in ids],
            "track_artist": ["Artist"] * len(ids), "lyrics": lyrics, "porodica": families,
            "godina": [2020] * len(ids), "playlist_genre": ["pop"] * len(ids),
        })
        self.enterContext(patch.object(pipeline, "ucitaj_katalog", return_value=self.catalog))
        self.enterContext(patch.object(pipeline, "ucitaj_indeks", return_value=self.index))
        self.enterContext(patch.object(pipeline, "ucitaj_trie", return_value=None))

    def app(self, page):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
        app.session_state["strana"] = page
        app.run()
        key = "jezicki_upit" if page == "Jezicki model" else "stihovi_upit"
        app.text_input(key=key).set_value("love").run()
        self.assertFalse(app.exception)
        return app

    def mark(self, app, doc, value):
        next(pill for pill in app.pills if pill.key.endswith(f"_{doc}")).set_value(value).run()
        self.assertFalse(app.exception)

    def test_explicit_labels_refresh_playlist_and_explanations_without_another_button_click(self):
        app = self.app("Stihovi")
        context = ("stihovi", "love", "boolean", "AND", 3, "tfidf", 1.5, 0.75)
        self.assertFalse(any(sub.value == "Tvoja plejlista" for sub in app.subheader))
        self.mark(app, 0, "like")
        self.mark(app, 1, "dislike")
        next(button for button in app.button if button.label == "Napravi plejlistu").click().run()
        self.assertFalse(app.exception)
        rock = app.session_state["plejliste"][context]
        self.assertEqual(len(rock.songs), 10)
        self.assertEqual(rock.relevant, [0])
        self.assertEqual(rock.non_relevant, [1])
        self.assertFalse(rock.assumed)
        self.assertNotIn(10, [doc for doc, _ in rock.songs])  # Hidden live version of Song 0.
        self.assertTrue(any("guitar" in text.value and text.value.startswith("Reči iz pesama")
                            for text in app.text))
        all_text = [text.value for text in app.text]
        words_position = next(i for i, text in enumerate(all_text) if text.startswith("Reči iz pesama"))
        song_position = next(i for i, text in enumerate(all_text) if text.startswith("1. Song"))
        self.assertLess(words_position, song_position)

        self.mark(app, 0, None)
        self.mark(app, 1, "like")
        beach = app.session_state["plejliste"][context]
        self.assertEqual(beach.relevant, [1])
        self.assertEqual(beach.non_relevant, [])
        self.assertEqual(len(beach.songs), 10)
        self.assertNotEqual(rock.songs, beach.songs)
        self.assertNotEqual([item.term for item in rock.terms], [item.term for item in beach.terms])
        self.assertTrue(any("beach" in text.value and text.value.startswith("Reči iz pesama")
                            for text in app.text))
        self.assertEqual(len({self.catalog.loc[doc, "porodica"] for doc, _ in beach.songs}), 10)
        self.mark(app, 1, None)
        assumed = app.session_state["plejliste"][context]
        self.assertTrue(assumed.assumed)
        self.assertEqual(assumed.relevant, [0, 1, 2, 3, 4])
        self.assertTrue(any("Nema oznaka" in info.value for info in app.info))

    def test_qlm_fallback_does_not_exclude_the_entire_catalog_and_remembers_seen_pages(self):
        with patch.object(QLM, "rank", autospec=True, side_effect=QLM.rank) as rank:
            app = self.app("Jezicki model")
            context = ("jezicki", "love")
            next(button for button in app.button if button.label == "Napravi plejlistu").click().run()
            self.assertFalse(app.exception)
            first = app.session_state["plejliste"][context]
            self.assertTrue(first.assumed)
            self.assertEqual(len(first.relevant), 5)
            self.assertEqual(len(first.songs), 10)
            rank.assert_called_once()
            shown = app.session_state["prikazane_pesme"][context]
            self.assertEqual(shown, set(range(10)))
            self.assertFalse(shown.intersection(doc for doc, _ in first.songs))

            app.number_input(key="jezicki_strana").set_value(2).run()
            self.assertFalse(app.exception)
            second = app.session_state["plejliste"][context]
            shown = app.session_state["prikazane_pesme"][context]
            self.assertEqual(len(shown), 20)
            shown_families = {self.catalog.loc[doc, "porodica"] for doc in shown}
            self.assertFalse(shown_families.intersection(self.catalog.loc[doc, "porodica"]
                                                       for doc, _ in second.songs))
            self.assertEqual(second.relevant, first.relevant)
            app.number_input(key="jezicki_strana").set_value(1).run()
            self.assertEqual(app.session_state["prikazane_pesme"][context], shown)
            rank.assert_called_once()
            app.text_input(key="jezicki_upit").set_value("beach").run()
            self.assertFalse(app.exception)
            self.assertFalse(any(sub.value == "Tvoja plejlista" for sub in app.subheader))


if __name__ == "__main__":
    unittest.main()
