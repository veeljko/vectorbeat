import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from scripts import pripremi


class PreparationCommandTests(unittest.TestCase):
    def test_ranking_only_mode_uses_existing_graph_without_crawling(self):
        pipeline = Mock()
        pipeline.napravi_rangove.return_value = {"artists": pd.DataFrame(), "d": 0.85}
        with patch.object(pripremi, "pipeline", pipeline), \
                patch("sys.argv", ["pripremi", "--samo-rangovi"]), \
                contextlib.redirect_stdout(io.StringIO()):
            pripremi.main()
        self.assertEqual([call[0] for call in pipeline.mock_calls], ["napravi_rangove"])

    def test_graph_seed_mode_only_prepares_seeds_then_graph(self):
        pipeline = Mock()
        pipeline.napravi_seed.return_value = pd.DataFrame({
            "artist_name": ["Anja Kotar"], "seed": ["https://en.wikipedia.org/wiki/Anja_Kotar"],
        })
        pipeline.napravi_graf.return_value = ({"url": []}, {"Anja Kotar": "url"})
        with patch.object(pripremi, "pipeline", pipeline), \
                patch("sys.argv", ["pripremi", "--samo-graf-seed"]), \
                contextlib.redirect_stdout(io.StringIO()):
            pripremi.main()
        self.assertEqual([call[0] for call in pipeline.mock_calls], ["napravi_seed", "napravi_graf"])
        pipeline.napravi_seed.assert_called_once_with(retry_unresolved=True)
        pipeline.napravi_graf.assert_called_once_with()

    def test_evaluation_only_mode_does_not_prepare_seeds_or_graph(self):
        pipeline = Mock()
        with patch.object(pripremi, "pipeline", pipeline), \
                patch("sys.argv", ["pripremi", "--samo-evaluacija"]), \
                contextlib.redirect_stdout(io.StringIO()):
            pripremi.main()
        self.assertEqual([call[0] for call in pipeline.mock_calls], ["napravi_evaluaciju"])

    def test_modes_cannot_be_combined(self):
        with patch("sys.argv", ["pripremi", "--samo-evaluacija", "--samo-graf-seed"]), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            pripremi.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
