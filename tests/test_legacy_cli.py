from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy_cli import parse_seed_range


class LegacyCliTests(unittest.TestCase):
    def test_parse_seed_range_reads_cli_flags(self) -> None:
        with patch.object(sys, "argv", ["prog", "--start", "2", "--end", "5"]):
            start_index, end_index = parse_seed_range()

        self.assertEqual((start_index, end_index), (2, 5))


if __name__ == "__main__":
    unittest.main()
