from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LegacyCommand:
    name: str
    script_name: str
    stdin_template: str | None = None
    per_seed: bool = False
    create_seed_dirs: bool = False


@dataclass(frozen=True)
class AggregateSpec:
    output_name: str
    input_template: str


@dataclass(frozen=True)
class ExperimentConfig:
    slug: str
    title: str
    paper_section: str
    description: str
    workdir: str
    collect_steps: tuple[LegacyCommand, ...] = field(default_factory=tuple)
    aggregate_specs: tuple[AggregateSpec, ...] = field(default_factory=tuple)
    plot_script: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def workdir_path(self) -> Path:
        return REPO_ROOT / self.workdir


def _aggregate_family(
    outputs: list[str],
    input_prefix: str | None = None,
) -> tuple[AggregateSpec, ...]:
    prefix = input_prefix
    specs: list[AggregateSpec] = []
    for output_name in outputs:
        input_name = output_name if prefix is None else output_name.replace(prefix.split("_")[0], input_prefix.split("_")[0], 1)
        specs.append(AggregateSpec(output_name=output_name, input_template=f"{{seed}}/{input_name}"))
    return tuple(specs)


def _specs(names: list[tuple[str, str]]) -> tuple[AggregateSpec, ...]:
    return tuple(
        AggregateSpec(output_name=output_name, input_template=f"{{seed}}/{input_name}")
        for output_name, input_name in names
    )


EXPERIMENTS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        slug="dollar-euro",
        title="Dollar-Euro",
        paper_section="Section 4.3",
        description="Fixed gridworld reward-adaptation experiment with linear reward combination.",
        workdir=r"Exp 1 Fixed MDP FIxed R\Dollar-Euro",
        collect_steps=(
            LegacyCommand(name="QL baseline", script_name="QL_RL.py"),
            LegacyCommand(
                name="SFQL baseline",
                script_name="SFQL_try1.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M / M-Q-M legacy run",
                script_name="DE_correction_inR.py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=_specs(
            [
                ("ours_0_1.csv", "ours_0_1.csv"),
                ("ours_0_2.csv", "ours_0_2.csv"),
                ("ours_0_3.csv", "ours_0_3.csv"),
                ("ours_0_4.csv", "ours_0_4.csv"),
                ("QL_0_1.csv", "QL_0_1.csv"),
                ("QL_0_2.csv", "QL_0_2.csv"),
                ("QL_0_3.csv", "QL_0_3.csv"),
                ("QL_0_4.csv", "QL_0_4.csv"),
                ("sfql_0_1.csv", "sfql_0_1.csv"),
                ("sfql_0_2.csv", "sfql_0_2.csv"),
                ("sfql_0_3.csv", "sfql_0_3.csv"),
                ("sfql_0_4.csv", "sfql_0_4.csv"),
            ]
        ),
        plot_script="plot_DE_all.py",
    ),
    ExperimentConfig(
        slug="racetrack",
        title="Racetrack",
        paper_section="Section 4.3",
        description="Racetrack gridworld reward-adaptation experiment with fixed dynamics.",
        workdir=r"Exp 1 Fixed MDP FIxed R\Racetrack",
        collect_steps=(
            LegacyCommand(
                name="QL baseline",
                script_name="QL_Racetrack.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="SFQL baseline",
                script_name="SFQL_Racetrack.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M / M-Q-M legacy run",
                script_name="RA_Racetrack_correction.py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=_specs(
            [
                ("ours_0_1.csv", "ours_0_1.csv"),
                ("ours_0_3.csv", "ours_0_3.csv"),
                ("ours_0_5.csv", "ours_0_5.csv"),
                ("ours_0_7.csv", "ours_0_7.csv"),
                ("QL_0_1.csv", "QL_0_1.csv"),
                ("QL_0_3.csv", "QL_0_3.csv"),
                ("QL_0_5.csv", "QL_0_5.csv"),
                ("QL_0_7.csv", "QL_0_7.csv"),
                ("SFQL_0_1.csv", "SFQL_0_1.csv"),
                ("SFQL_0_3.csv", "SFQL_0_3.csv"),
                ("SFQL_0_5.csv", "SFQL_0_5.csv"),
                ("SFQL_0_7.csv", "SFQL_0_7.csv"),
            ]
        ),
        plot_script="plot_RT_all.py",
        notes=(
            "This legacy workflow expects seed folders with pre-generated T_*.npy files.",
        ),
    ),
    ExperimentConfig(
        slug="frozen-lake",
        title="Frozen Lake",
        paper_section="Section 4.3",
        description="Frozen Lake reward-adaptation experiment with fixed grid generation.",
        workdir=r"Exp 1 Fixed MDP FIxed R\Frozen lake",
        collect_steps=(
            LegacyCommand(
                name="QL baseline",
                script_name="QL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="SFQL baseline",
                script_name="sfql.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M / M-Q-M legacy run",
                script_name="QM.py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=_specs(
            [
                ("ours_0_1.csv", "ours_0_1.csv"),
                ("ours_0_2.csv", "ours_0_2.csv"),
                ("ours_0_3.csv", "ours_0_3.csv"),
                ("ours_0_4.csv", "ours_0_4.csv"),
                ("QL_0_1.csv", "QL_0_1.csv"),
                ("QL_0_2.csv", "QL_0_2.csv"),
                ("QL_0_3.csv", "QL_0_3.csv"),
                ("QL_0_4.csv", "QL_0_4.csv"),
                ("SFQL_0_1.csv", "SFQL_0_1.csv"),
                ("SFQL_0_2.csv", "SFQL_0_2.csv"),
                ("SFQL_0_3.csv", "SFQL_0_3.csv"),
                ("SFQL_0_4.csv", "SFQL_0_4.csv"),
            ]
        ),
        plot_script="plot.py",
        notes=(
            "The frozen-lake scripts reuse legacy environment snapshots stored in the seed folders.",
        ),
    ),
    ExperimentConfig(
        slug="autogenerated-linear",
        title="Autogenerated MDP (Linear)",
        paper_section="Section 4.4",
        description="Randomized discrete MDP experiment with linear target reward combination.",
        workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_135",
        collect_steps=(
            LegacyCommand(
                name="Generate per-seed MDPs",
                script_name="Generator.py",
                per_seed=True,
                create_seed_dirs=True,
            ),
            LegacyCommand(
                name="QL baseline",
                script_name="QL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="SFQL baseline",
                script_name="SFQL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M / M-Q-M legacy run",
                script_name="RA_autogen_correction_inR.py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=_specs(
            [
                ("ours_0_1.csv", "ours_0_1.csv"),
                ("ours_0_3.csv", "ours_0_3.csv"),
                ("ours_0_5.csv", "ours_0_5.csv"),
                ("QL_0_1.csv", "QL_0_1.csv"),
                ("QL_0_3.csv", "QL_0_3.csv"),
                ("QL_0_5.csv", "QL_0_5.csv"),
                ("sfql_0_1.csv", "SFQL_0_1.csv"),
                ("sfql_0_3.csv", "SFQL_0_3.csv"),
                ("sfql_0_5.csv", "SFQL_0_5.csv"),
            ]
        ),
        plot_script="plot_all.py",
    ),
    ExperimentConfig(
        slug="autogenerated-nonlinear",
        title="Autogenerated MDP (Nonlinear)",
        paper_section="Section 4.4",
        description="Randomized discrete MDP experiment with nonlinear target reward combination.",
        workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_nonlinear - Copy",
        collect_steps=(
            LegacyCommand(
                name="Generate per-seed MDPs",
                script_name="Generator.py",
                per_seed=True,
                create_seed_dirs=True,
            ),
            LegacyCommand(
                name="QL baseline",
                script_name="QL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="SFQL baseline",
                script_name="SFQL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M / M-Q-M legacy run",
                script_name="RA_autogen_correction_inR.py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=_specs(
            [
                ("ours_0_1.csv", "ours_0_1.csv"),
                ("ours_0_3.csv", "ours_0_3.csv"),
                ("ours_0_5.csv", "ours_0_5.csv"),
                ("QL_0_1.csv", "QL_0_1.csv"),
                ("QL_0_3.csv", "QL_0_3.csv"),
                ("QL_0_5.csv", "QL_0_5.csv"),
                ("sfql_0_1.csv", "SFQL_0_1.csv"),
                ("sfql_0_3.csv", "SFQL_0_3.csv"),
                ("sfql_0_5.csv", "SFQL_0_5.csv"),
            ]
        ),
        plot_script="plot_all.py",
    ),
    ExperimentConfig(
        slug="noisy-combination",
        title="Noisy Combination Function",
        paper_section="Section 4.5",
        description="Randomized discrete MDP experiment with bounded noise in the target reward mapping.",
        workdir=r"Exp 6 Noise\autogenerated - works",
        collect_steps=(
            LegacyCommand(
                name="Generate per-seed MDPs",
                script_name="Generator.py",
                per_seed=True,
                create_seed_dirs=True,
            ),
            LegacyCommand(
                name="QL baseline",
                script_name="QL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="M-Q-M legacy run",
                script_name="RA_autogen_correction_inR.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M legacy run",
                script_name="RA_autogen_correction_inR-copy(1).py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=_specs(
            [
                ("QL_0_0.csv", "QL_0_0.csv"),
                ("ours_0_0.006.csv", "ours_0_0.006.csv"),
                ("ours_0_0.012.csv", "ours_0_0.012.csv"),
                ("ours_0_0.018.csv", "ours_0_0.018.csv"),
                ("ours_0_0.024.csv", "ours_0_0.024.csv"),
                ("ours_0_0.03.csv", "ours_0_0.03.csv"),
                ("ours_0_0.006_new_new.csv", "ours_0_0.006_new.csv"),
                ("ours_0_0.012_new_new.csv", "ours_0_0.012_new.csv"),
                ("ours_0_0.018_new_new.csv", "ours_0_0.018_new.csv"),
                ("ours_0_0.024_new_new.csv", "ours_0_0.024_new.csv"),
                ("ours_0_0.03_new_new.csv", "ours_0_0.03_new.csv"),
            ]
        ),
        plot_script="plot_final.py",
    ),
)


EXPERIMENT_BY_SLUG = {experiment.slug: experiment for experiment in EXPERIMENTS}


def list_experiments() -> tuple[ExperimentConfig, ...]:
    return EXPERIMENTS


def get_experiment(slug: str) -> ExperimentConfig:
    try:
        return EXPERIMENT_BY_SLUG[slug]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENT_BY_SLUG))
        raise KeyError(f"Unknown experiment '{slug}'. Available experiments: {available}") from exc
