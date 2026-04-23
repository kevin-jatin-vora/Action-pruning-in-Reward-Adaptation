from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmanip.experiments import REPO_ROOT, list_experiments


class RegistryTests(unittest.TestCase):
    def test_registered_scripts_exist(self) -> None:
        experiments = list_experiments()
        self.assertGreaterEqual(len(experiments), 6)
        for experiment in experiments:
            self.assertTrue(experiment.workdir_path.exists(), msg=str(experiment.workdir_path))
            for step in experiment.collect_steps:
                step_workdir = REPO_ROOT / step.workdir if step.workdir else experiment.workdir_path
                script_path = step_workdir / step.script_name
                self.assertTrue(script_path.exists(), msg=str(script_path))
            for spec in experiment.aggregate_specs:
                source_dir = REPO_ROOT / spec.source_workdir if spec.source_workdir else experiment.workdir_path
                self.assertTrue(source_dir.exists(), msg=str(source_dir))
            if experiment.plot_script is not None:
                plot_path = experiment.workdir_path / experiment.plot_script
                self.assertTrue(plot_path.exists(), msg=str(plot_path))


if __name__ == "__main__":
    unittest.main()
