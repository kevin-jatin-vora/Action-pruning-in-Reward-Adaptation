from __future__ import annotations

import unittest
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from qmanip.aggregation import aggregate_csv_group
from qmanip.experiments import AggregateSpec


class AggregationTests(unittest.TestCase):
    def test_aggregate_csv_group_concatenates_seed_outputs(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        tmp_dir = tests_dir / "_tmp_aggregation_case"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        try:
            for seed in range(3):
                seed_dir = tmp_dir / str(seed)
                seed_dir.mkdir()
                pd.DataFrame({"reward": [seed, seed + 1]}).to_csv(seed_dir / "ours.csv", index=False)

            output_path = aggregate_csv_group(
                source_dir=tmp_dir,
                output_dir=tmp_dir,
                spec=AggregateSpec(output_name="aggregate.csv", input_template="{seed}/ours.csv"),
                start=0,
                end=3,
            )

            aggregated = pd.read_csv(output_path)
            self.assertEqual(list(aggregated["reward"]), [0, 1, 1, 2, 2, 3])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
