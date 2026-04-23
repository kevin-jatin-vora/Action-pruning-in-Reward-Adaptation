from __future__ import annotations

from pathlib import Path

import pandas as pd

from .experiments import AggregateSpec, ExperimentConfig, REPO_ROOT


def _resolve_dir(directory: str | None, default: Path) -> Path:
    if directory is None:
        return default
    return REPO_ROOT / directory


def aggregate_csv_group(
    source_dir: Path,
    output_dir: Path,
    spec: AggregateSpec,
    start: int,
    end: int,
) -> Path:
    frames: list[pd.DataFrame] = []
    missing_files: list[Path] = []

    for seed in range(start, end):
        input_path = source_dir / spec.input_template.format(seed=seed)
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

    combined = pd.concat(frames, ignore_index=True)

    output_path = output_dir / spec.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    legacy_output_name = spec.legacy_output_name or Path(spec.input_template.format(seed=start)).name
    legacy_output_path = source_dir / legacy_output_name
    if legacy_output_path != output_path:
        combined.to_csv(legacy_output_path, index=False)

    return output_path


def aggregate_experiment(
    experiment: ExperimentConfig,
    start: int,
    end: int,
) -> list[Path]:
    outputs: list[Path] = []

    for spec in experiment.aggregate_specs:
        source_dir = _resolve_dir(spec.source_workdir, experiment.workdir_path)
        output_dir = _resolve_dir(spec.output_workdir, experiment.workdir_path)
        outputs.append(
            aggregate_csv_group(
                source_dir=source_dir,
                output_dir=output_dir,
                spec=spec,
                start=start,
                end=end,
            )
        )

    return outputs
