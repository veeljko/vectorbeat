import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mrvectorbeat import pipeline


class IndexPipelineTests(unittest.TestCase):
    def test_build_and_reload_artifact_preserves_catalog_ids(self):
        catalog = pd.DataFrame({"katalog_id": [8, 2, 5], "track_id": ["same", "same", "empty"]})
        tokens = pd.DataFrame({"katalog_id": [2, 5, 8], "track_id": ["same", "empty", "same"],
                               "tokens": [["night", "love"], [], ["love", "love"]]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(pipeline, INDEKS_PATH=root / "index.pkl",
                                KATALOG_PATH=root / "catalog.parquet", TOKENI_PATH=root / "tokens.parquet"):
                with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                        patch.object(pipeline, "ucitaj_tokene", return_value=tokens):
                    built = pipeline.napravi_indeks()
                with patch.object(pipeline, "napravi_indeks", side_effect=AssertionError("Rebuild")):
                    loaded = pipeline.ucitaj_indeks()
                self.assertEqual(loaded.index, built.index)
                self.assertEqual(loaded.index["love"], [(2, [1]), (8, [0, 1])])
                self.assertEqual(loaded.document_ids, [2, 5, 8])
                self.assertEqual(loaded.find_boolean(["love", "night"]), [2])
                self.assertEqual(loaded.find_phrase(["night", "love"]), [(2, 0)])

    def test_mismatched_catalog_and_tokens_rejected(self):
        catalog = pd.DataFrame({"katalog_id": [2], "track_id": ["a"]})
        tokens = pd.DataFrame({"katalog_id": [9], "track_id": ["a"], "tokens": [["love"]]})
        with patch.object(pipeline, "ucitaj_katalog", return_value=catalog), \
                patch.object(pipeline, "ucitaj_tokene", return_value=tokens):
            with self.assertRaises(ValueError):
                pipeline.napravi_indeks()


if __name__ == "__main__":
    unittest.main()
