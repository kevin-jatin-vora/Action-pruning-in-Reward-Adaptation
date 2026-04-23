from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .aggregation import aggregate_experiment
from .experiments import ExperimentConfig, LegacyCommand, REPO_ROOT


class ExperimentRunner:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def _resolve_workdir(self, experiment: ExperimentConfig, step: LegacyCommand | None = None) -> Path:
        if step is not None and step.workdir is not None:
            return REPO_ROOT / step.workdir
        return experiment.workdir_path

    def run(self, experiment: ExperimentConfig, stage: str, start: int, end: int) -> None:
        if start < 0 or end <= start:
            raise ValueError("Expected a non-empty seed range where start >= 0 and end > start.")

        if stage in {"collect", "all"}:
            self._run_collect(experiment=experiment, start=start, end=end)
        if stage in {"aggregate", "all"}:
            if self.dry_run:
                for spec in experiment.aggregate_specs:
                    source_dir = REPO_ROOT / spec.source_workdir if spec.source_workdir else experiment.workdir_path
                    output_dir = REPO_ROOT / spec.output_workdir if spec.output_workdir else experiment.workdir_path
                    print(
                        "[aggregate] "
                        f"{output_dir / spec.output_name} "
                        f"<- {source_dir / spec.input_template}"
                    )
            else:
                outputs = aggregate_experiment(experiment=experiment, start=start, end=end)
                for output_path in outputs:
                    print(f"[aggregate] wrote {output_path}")
        if stage in {"plot", "all"}:
            self._run_plot(experiment=experiment)

    def _run_collect(self, experiment: ExperimentConfig, start: int, end: int) -> None:
        for step in experiment.collect_steps:
            step_workdir = self._resolve_workdir(experiment, step)
            if step.per_seed:
                for seed in range(start, end):
                    seed_dir = step_workdir / str(seed)
                    if step.create_seed_dirs:
                        seed_dir.mkdir(parents=True, exist_ok=True)
                    self._run_script(
                        step=step,
                        script_path=step_workdir / step.script_name,
                        cwd=seed_dir,
                        stdin_text=None,
                        extra_args=[],
                    )
                continue

            stdin_text = None
            if step.stdin_template is not None:
                stdin_text = step.stdin_template.format(start=start, end=end)

            self._run_script(
                step=step,
                script_path=step_workdir / step.script_name,
                cwd=step_workdir,
                stdin_text=stdin_text,
                extra_args=["--start", str(start), "--end", str(end)],
            )

    def _run_plot(self, experiment: ExperimentConfig) -> None:
        if experiment.plot_script is None:
            print(f"[plot] no plot script is registered for {experiment.slug}")
            return
        plot_path = experiment.workdir_path / experiment.plot_script
        self._run_script(
            step=LegacyCommand(name="plot", script_name=experiment.plot_script),
            script_path=plot_path,
            cwd=experiment.workdir_path,
            stdin_text=None,
            extra_args=[],
        )

    def _run_script(
        self,
        step: LegacyCommand,
        script_path: Path,
        cwd: Path,
        stdin_text: str | None,
        extra_args: list[str],
    ) -> None:
        if not script_path.exists():
            raise FileNotFoundError(f"Expected script does not exist: {script_path}")

        command = [sys.executable, str(script_path), *extra_args]
        print(f"[run] {step.name}: {' '.join(command)} (cwd={cwd})")

        if self.dry_run:
            return

        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(
            command,
            cwd=str(cwd),
            input=stdin_text,
            text=True,
            check=True,
            env=env,
        )
