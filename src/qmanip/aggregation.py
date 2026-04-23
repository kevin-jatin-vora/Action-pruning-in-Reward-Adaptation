from __future__ import annotations

from pathlib import Path

import pandas as pd

from .experiments import AggregateSpec, ExperimentConfig


def aggregate_csv_group(base_dir: Path, spec: AggregateSpec, start: int, end: int) -> Path:
    frames: list[pd.DataFrame] = []
    missing_files: list[Path] = []

    for seed in range(start, end):
        input_path = base_dir / spec.input_template.format(seed=seed)
        if not input_path.exists():
            missing_files.append(input_path)
            continue
        frames.append(pd.read_csv(input_path))

    if missing_files:
        sample = "\n".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_files)} required CSV files while aggregating "
            f"{spec.output_name}.\nExamples:\n{sample}"
        )

    if not frames:
        raise FileNotFoundError(f"No inputs were found for aggregate {spec.output_name}.")

    output_path = base_dir / spec.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    return output_path


def aggregate_experiment(
    experiment: ExperimentConfig,
    start: int,
    end: int,
) -> list[Path]:
    base_dir = experiment.workdir_path
    outputs: list[Path] = []

    for spec in experiment.aggregate_specs:
        outputs.append(aggregate_csv_group(base_dir=base_dir, spec=spec, start=start, end=end))

    return outputs
