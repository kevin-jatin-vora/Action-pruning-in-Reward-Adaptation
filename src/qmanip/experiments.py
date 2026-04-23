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
    workdir: str | None = None


@dataclass(frozen=True)
class AggregateSpec:
    output_name: str
    input_template: str
    source_workdir: str | None = None
    output_workdir: str | None = None
    legacy_output_name: str | None = None


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


def _results_dir(slug: str) -> str:
    return fr"results\{slug}"


def _specs(
    names: list[tuple[str, str]],
    *,
    source_workdir: str | None = None,
    output_workdir: str | None = None,
) -> tuple[AggregateSpec, ...]:
    return tuple(
        AggregateSpec(
            output_name=output_name,
            input_template=input_template,
            source_workdir=source_workdir,
            output_workdir=output_workdir,
        )
        for output_name, input_template in names
    )


EXPERIMENTS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        slug="dollar-euro",
        title="Dollar-Euro",
        paper_section="Section 4.3",
        description="Fixed gridworld reward-adaptation experiment with linear reward combination.",
        workdir=r"Exp 1 Fixed MDP FIxed R\Dollar-Euro",
        collect_steps=(
            LegacyCommand(name="QL baseline", script_name="QL.py"),
            LegacyCommand(
                name="SFQL baseline",
                script_name="SFQL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Clipped QL baseline",
                script_name="clipped_QL_proposed_soln.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="M-Q-M",
                script_name="MQM.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M",
                script_name="QM.py",
                stdin_template="{start}\n{end}\n",
                workdir=r"Exp 1 Fixed MDP FIxed R - Copy\Dollar-Euro",
            ),
        ),
        aggregate_specs=(
            *_specs(
                [
                    ("mqm_0_1.csv", "{seed}/ours_0_1.csv"),
                    ("mqm_0_2.csv", "{seed}/ours_0_2.csv"),
                    ("mqm_0_3.csv", "{seed}/ours_0_3.csv"),
                    ("mqm_0_4.csv", "{seed}/ours_0_4.csv"),
                    ("ql_0_1.csv", "{seed}/QL_0_1.csv"),
                    ("ql_0_2.csv", "{seed}/QL_0_2.csv"),
                    ("ql_0_3.csv", "{seed}/QL_0_3.csv"),
                    ("ql_0_4.csv", "{seed}/QL_0_4.csv"),
                    ("sfql_0_1.csv", "{seed}/sfql_0_1.csv"),
                    ("sfql_0_2.csv", "{seed}/sfql_0_2.csv"),
                    ("sfql_0_3.csv", "{seed}/sfql_0_3.csv"),
                    ("sfql_0_4.csv", "{seed}/sfql_0_4.csv"),
                    ("clipped_ql_0_1.csv", "{seed}/clipped_QL_0_1.csv"),
                    ("clipped_ql_0_2.csv", "{seed}/clipped_QL_0_2.csv"),
                    ("clipped_ql_0_3.csv", "{seed}/clipped_QL_0_3.csv"),
                    ("clipped_ql_0_4.csv", "{seed}/clipped_QL_0_4.csv"),
                ],
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Dollar-Euro",
                output_workdir=_results_dir("dollar-euro"),
            ),
            *_specs(
                [
                    ("qm_0_1.csv", "{seed}/ours_0_1.csv"),
                    ("qm_0_2.csv", "{seed}/ours_0_2.csv"),
                    ("qm_0_3.csv", "{seed}/ours_0_3.csv"),
                    ("qm_0_4.csv", "{seed}/ours_0_4.csv"),
                ],
                source_workdir=r"Exp 1 Fixed MDP FIxed R - Copy\Dollar-Euro",
                output_workdir=_results_dir("dollar-euro"),
            ),
        ),
        plot_script="plot_results.py",
        notes=(
            "Aggregated CSVs are written to 'results/dollar-euro/'.",
        ),
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
                script_name="QL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="SFQL baseline",
                script_name="SFQL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Clipped QL baseline",
                script_name="clipped_QL_proposed_soln.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="M-Q-M",
                script_name="MQM.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M",
                script_name="QM.py",
                stdin_template="{start}\n{end}\n",
                workdir=r"Exp 1 Fixed MDP FIxed R - Copy\Racetrack",
            ),
        ),
        aggregate_specs=(
            *_specs(
                [
                    ("mqm_0_1.csv", "{seed}/ours_0_1.csv"),
                    ("mqm_0_3.csv", "{seed}/ours_0_3.csv"),
                    ("mqm_0_5.csv", "{seed}/ours_0_5.csv"),
                    ("mqm_0_7.csv", "{seed}/ours_0_7.csv"),
                    ("ql_0_1.csv", "{seed}/QL_0_1.csv"),
                    ("ql_0_3.csv", "{seed}/QL_0_3.csv"),
                    ("ql_0_5.csv", "{seed}/QL_0_5.csv"),
                    ("ql_0_7.csv", "{seed}/QL_0_7.csv"),
                    ("sfql_0_1.csv", "{seed}/SFQL_0_1.csv"),
                    ("sfql_0_3.csv", "{seed}/SFQL_0_3.csv"),
                    ("sfql_0_5.csv", "{seed}/SFQL_0_5.csv"),
                    ("sfql_0_7.csv", "{seed}/SFQL_0_7.csv"),
                ],
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Racetrack",
                output_workdir=_results_dir("racetrack"),
            ),
            AggregateSpec(
                output_name="clipped_ql_0_1.csv",
                input_template="{seed}/ClippedQL_0_1.csv",
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Racetrack",
                output_workdir=_results_dir("racetrack"),
                legacy_output_name="clipped_QL_0_1.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_3.csv",
                input_template="{seed}/ClippedQL_0_3.csv",
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Racetrack",
                output_workdir=_results_dir("racetrack"),
                legacy_output_name="clipped_QL_0_3.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_5.csv",
                input_template="{seed}/ClippedQL_0_5.csv",
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Racetrack",
                output_workdir=_results_dir("racetrack"),
                legacy_output_name="clipped_QL_0_5.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_7.csv",
                input_template="{seed}/ClippedQL_0_7.csv",
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Racetrack",
                output_workdir=_results_dir("racetrack"),
                legacy_output_name="clipped_QL_0_7.csv",
            ),
            *_specs(
                [
                    ("qm_0_1.csv", "{seed}/ours_0_1.csv"),
                    ("qm_0_3.csv", "{seed}/ours_0_3.csv"),
                    ("qm_0_5.csv", "{seed}/ours_0_5.csv"),
                    ("qm_0_7.csv", "{seed}/ours_0_7.csv"),
                ],
                source_workdir=r"Exp 1 Fixed MDP FIxed R - Copy\Racetrack",
                output_workdir=_results_dir("racetrack"),
            ),
        ),
        plot_script="plot_results.py",
        notes=(
            "This workflow expects seed folders with pre-generated T_*.npy files.",
            "Aggregated CSVs are written to 'results/racetrack/'.",
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
                script_name="SFQL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Clipped QL baseline",
                script_name="clipped_QL_proposed_soln.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="M-Q-M",
                script_name="MQM.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M",
                script_name="QM.py",
                stdin_template="{start}\n{end}\n",
                workdir=r"Exp 1 Fixed MDP FIxed R - Copy\Frozen lake",
            ),
        ),
        aggregate_specs=(
            *_specs(
                [
                    ("mqm_0_1.csv", "{seed}/ours_0_1.csv"),
                    ("mqm_0_2.csv", "{seed}/ours_0_2.csv"),
                    ("mqm_0_3.csv", "{seed}/ours_0_3.csv"),
                    ("mqm_0_4.csv", "{seed}/ours_0_4.csv"),
                    ("ql_0_1.csv", "{seed}/QL_0_1.csv"),
                    ("ql_0_2.csv", "{seed}/QL_0_2.csv"),
                    ("ql_0_3.csv", "{seed}/QL_0_3.csv"),
                    ("ql_0_4.csv", "{seed}/QL_0_4.csv"),
                    ("sfql_0_1.csv", "{seed}/SFQL_0_1.csv"),
                    ("sfql_0_2.csv", "{seed}/SFQL_0_2.csv"),
                    ("sfql_0_3.csv", "{seed}/SFQL_0_3.csv"),
                    ("sfql_0_4.csv", "{seed}/SFQL_0_4.csv"),
                ],
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Frozen lake",
                output_workdir=_results_dir("frozen-lake"),
            ),
            AggregateSpec(
                output_name="clipped_ql_0_1.csv",
                input_template="{seed}/ClippedQL_0_1.csv",
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Frozen lake",
                output_workdir=_results_dir("frozen-lake"),
                legacy_output_name="clipped_QL_0_1.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_2.csv",
                input_template="{seed}/ClippedQL_0_2.csv",
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Frozen lake",
                output_workdir=_results_dir("frozen-lake"),
                legacy_output_name="clipped_QL_0_2.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_3.csv",
                input_template="{seed}/ClippedQL_0_3.csv",
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Frozen lake",
                output_workdir=_results_dir("frozen-lake"),
                legacy_output_name="clipped_QL_0_3.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_4.csv",
                input_template="{seed}/ClippedQL_0_4.csv",
                source_workdir=r"Exp 1 Fixed MDP FIxed R\Frozen lake",
                output_workdir=_results_dir("frozen-lake"),
                legacy_output_name="clipped_QL_0_4.csv",
            ),
            *_specs(
                [
                    ("qm_0_1.csv", "{seed}/ours_0_1.csv"),
                    ("qm_0_2.csv", "{seed}/ours_0_2.csv"),
                    ("qm_0_3.csv", "{seed}/ours_0_3.csv"),
                    ("qm_0_4.csv", "{seed}/ours_0_4.csv"),
                ],
                source_workdir=r"Exp 1 Fixed MDP FIxed R - Copy\Frozen lake",
                output_workdir=_results_dir("frozen-lake"),
            ),
        ),
        plot_script="plot_results.py",
        notes=(
            "The Frozen Lake scripts reuse legacy environment snapshots stored in the seed folders.",
            "Aggregated CSVs are written to 'results/frozen-lake/'.",
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
                script_name="generate_mdp.py",
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
                name="Clipped QL baseline",
                script_name="clipped_QL_proposed_soln.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="M-Q-M",
                script_name="MQM.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M",
                script_name="QM.py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=(
            *_specs(
                [
                    ("mqm_0_1.csv", "{seed}/ours_0_1.csv"),
                    ("mqm_0_3.csv", "{seed}/ours_0_3.csv"),
                    ("mqm_0_5.csv", "{seed}/ours_0_5.csv"),
                    ("ql_0_1.csv", "{seed}/QL_0_1.csv"),
                    ("ql_0_3.csv", "{seed}/QL_0_3.csv"),
                    ("ql_0_5.csv", "{seed}/QL_0_5.csv"),
                    ("sfql_0_1.csv", "{seed}/sfql_0_1.csv"),
                    ("sfql_0_3.csv", "{seed}/sfql_0_3.csv"),
                    ("sfql_0_5.csv", "{seed}/sfql_0_5.csv"),
                ],
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_135",
                output_workdir=_results_dir("autogenerated-linear"),
            ),
            AggregateSpec(
                output_name="clipped_ql_0_1.csv",
                input_template="{seed}/ClippedQL_0_1.csv",
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_135",
                output_workdir=_results_dir("autogenerated-linear"),
                legacy_output_name="clipped-QL_0_1.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_3.csv",
                input_template="{seed}/ClippedQL_0_3.csv",
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_135",
                output_workdir=_results_dir("autogenerated-linear"),
                legacy_output_name="clipped-QL_0_3.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_5.csv",
                input_template="{seed}/ClippedQL_0_5.csv",
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_135",
                output_workdir=_results_dir("autogenerated-linear"),
                legacy_output_name="clipped-QL_0_5.csv",
            ),
            *_specs(
                [
                    ("qm_0_1.csv", "{seed}/ours_0_1_new.csv"),
                    ("qm_0_3.csv", "{seed}/ours_0_3_new.csv"),
                    ("qm_0_5.csv", "{seed}/ours_0_5_new.csv"),
                ],
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_135",
                output_workdir=_results_dir("autogenerated-linear"),
            ),
        ),
        plot_script="plot_results.py",
        notes=(
            "Aggregated CSVs are written to 'results/autogenerated-linear/'.",
        ),
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
                script_name="generate_mdp.py",
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
                name="Clipped QL baseline",
                script_name="clipped_QL_proposed_soln.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="M-Q-M",
                script_name="MQM.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M",
                script_name="QM.py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=(
            *_specs(
                [
                    ("mqm_0_1.csv", "{seed}/ours_0_1.csv"),
                    ("mqm_0_3.csv", "{seed}/ours_0_3.csv"),
                    ("mqm_0_5.csv", "{seed}/ours_0_5.csv"),
                    ("ql_0_1.csv", "{seed}/QL_0_1.csv"),
                    ("ql_0_3.csv", "{seed}/QL_0_3.csv"),
                    ("ql_0_5.csv", "{seed}/QL_0_5.csv"),
                    ("sfql_0_1.csv", "{seed}/sfql_0_1.csv"),
                    ("sfql_0_3.csv", "{seed}/sfql_0_3.csv"),
                    ("sfql_0_5.csv", "{seed}/sfql_0_5.csv"),
                ],
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_nonlinear - Copy",
                output_workdir=_results_dir("autogenerated-nonlinear"),
            ),
            AggregateSpec(
                output_name="clipped_ql_0_1.csv",
                input_template="{seed}/ClippedQL_0_1.csv",
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_nonlinear - Copy",
                output_workdir=_results_dir("autogenerated-nonlinear"),
                legacy_output_name="clipped-QL_0_1.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_3.csv",
                input_template="{seed}/ClippedQL_0_3.csv",
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_nonlinear - Copy",
                output_workdir=_results_dir("autogenerated-nonlinear"),
                legacy_output_name="clipped-QL_0_3.csv",
            ),
            AggregateSpec(
                output_name="clipped_ql_0_5.csv",
                input_template="{seed}/ClippedQL_0_5.csv",
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_nonlinear - Copy",
                output_workdir=_results_dir("autogenerated-nonlinear"),
                legacy_output_name="clipped-QL_0_5.csv",
            ),
            *_specs(
                [
                    ("qm_0_1.csv", "{seed}/ours_0_1_new.csv"),
                    ("qm_0_3.csv", "{seed}/ours_0_3_new.csv"),
                    ("qm_0_5.csv", "{seed}/ours_0_5_new.csv"),
                ],
                source_workdir=r"Exp 2 Randomized MDP fixed R\Autogenerated_nonlinear - Copy",
                output_workdir=_results_dir("autogenerated-nonlinear"),
            ),
        ),
        plot_script="plot_results.py",
        notes=(
            "Aggregated CSVs are written to 'results/autogenerated-nonlinear/'.",
        ),
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
                script_name="generate_mdp.py",
                per_seed=True,
                create_seed_dirs=True,
            ),
            LegacyCommand(
                name="QL baseline",
                script_name="QL.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="M-Q-M",
                script_name="MQM.py",
                stdin_template="{start}\n{end}\n",
            ),
            LegacyCommand(
                name="Q-M",
                script_name="QM.py",
                stdin_template="{start}\n{end}\n",
            ),
        ),
        aggregate_specs=(
            *_specs(
                [
                    ("ql_0_0.csv", "{seed}/QL_0_0.csv"),
                    ("mqm_0_0.006.csv", "{seed}/ours_0_0.006.csv"),
                    ("mqm_0_0.012.csv", "{seed}/ours_0_0.012.csv"),
                    ("mqm_0_0.018.csv", "{seed}/ours_0_0.018.csv"),
                    ("mqm_0_0.024.csv", "{seed}/ours_0_0.024.csv"),
                    ("mqm_0_0.03.csv", "{seed}/ours_0_0.03.csv"),
                ],
                source_workdir=r"Exp 6 Noise\autogenerated - works",
                output_workdir=_results_dir("noisy-combination"),
            ),
            *_specs(
                [
                    ("qm_0_0.006.csv", "{seed}/ours_0_0.006_new_new.csv"),
                    ("qm_0_0.012.csv", "{seed}/ours_0_0.012_new_new.csv"),
                    ("qm_0_0.018.csv", "{seed}/ours_0_0.018_new_new.csv"),
                    ("qm_0_0.024.csv", "{seed}/ours_0_0.024_new_new.csv"),
                    ("qm_0_0.03.csv", "{seed}/ours_0_0.03_new_new.csv"),
                ],
                source_workdir=r"Exp 6 Noise\autogenerated - works",
                output_workdir=_results_dir("noisy-combination"),
            ),
        ),
        plot_script="plot_results.py",
        notes=(
            "Aggregated CSVs are written to 'results/noisy-combination/'.",
        ),
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
