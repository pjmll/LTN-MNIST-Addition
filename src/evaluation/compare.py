import argparse
import csv
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter

from src.utils.artifacts import validate_run_id, write_json
from src.utils.config import GROUPS, ROOT, ExperimentConfig, load_saved_config


METRICS = (
    "train_loss", "val_loss", "train_satisfaction", "val_satisfaction",
    "val_hard_sum_accuracy", "val_prob_sum_accuracy", "val_digit_accuracy",
    "train_seconds", "total_train_seconds", "elapsed_seconds",
)
PLOTS = {
    "sum_accuracy": ("val_hard_sum_accuracy", "Validation HardSum accuracy", True),
    "training_loss": ("train_loss", "Training loss (1 − Sat)", False),
    "satisfiability": ("val_satisfaction", "Validation satisfiability (Sat)", False),
    "digit_accuracy": ("val_digit_accuracy", "Validation single-digit accuracy", True),
}


@dataclass
class Run:
    run_id: str
    config: ExperimentConfig
    metadata: dict
    epochs: list[dict[str, float]]


def read_run(root: Path, run_id: str) -> Run:
    validate_run_id(run_id)
    folder = root / "experiments" / run_id
    config = load_saved_config(folder)
    if config.group is None:
        raise ValueError(f"{run_id} is missing a comparison group")
    if not (root / "checkpoints" / run_id / "final.pt").is_file():
        raise ValueError(f"{run_id} is missing its final checkpoint")
    metadata = json.loads((folder / "run.json").read_text("utf-8"))
    with (folder / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(1, config.epochs + 1)):
        raise ValueError(f"{run_id} must have exactly one row for every configured epoch")
    epochs = []
    for epoch, row in enumerate(rows, 1):
        if int(row["exists_p"]) != config.ltn.exists_p_at(epoch):
            raise ValueError(f"{run_id} has inconsistent p at epoch {epoch}")
        metrics = {key: float(row[key]) for key in METRICS}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"{run_id} has non-finite metrics at epoch {epoch}")
        epochs.append(metrics)
    return Run(run_id, config, metadata, epochs)


def check_controls(runs: list[Run]) -> dict[str, list[Run]]:
    by_group = {group: [] for group in GROUPS}
    seen = set()
    initial_states = {}
    baseline = schedule = None
    for run in runs:
        settings = asdict(run.config)
        group = settings.pop("group")
        seed = settings.pop("seed")
        logic = settings.pop("ltn")
        stages = logic.pop("exists_schedule")
        logic.pop("conjunction")
        runtime = dict(run.metadata)
        initial = runtime.pop("initial_state_sha256")
        controls = settings, logic, runtime
        if baseline is None:
            baseline = controls
        if controls != baseline:
            raise ValueError(f"{run.run_id} differs in controls, environment or data pairing")
        if GROUPS[group][1] is None:
            if schedule is None:
                schedule = stages
            if schedule != stages:
                raise ValueError("A3 and B3 must use the same p schedule")
        if (group, seed) in seen:
            raise ValueError(f"Multiple runs selected for {group}, seed {seed}")
        seen.add((group, seed))
        if initial_states.setdefault(seed, initial) != initial:
            raise ValueError(f"{run.run_id} differs in initialization for seed {seed}")
        by_group[group].append(run)
    seeds = set(initial_states)
    for group, group_runs in by_group.items():
        if not seeds or {run.config.seed for run in group_runs} != seeds:
            raise ValueError(f"{group} must include one completed run for every selected seed")
        group_runs.sort(key=lambda run: run.config.seed)
    return by_group


def aggregate(rows: list[dict[str, float]]) -> dict:
    stats = {"n_seeds": len(rows)}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        stats[f"{metric}_mean"] = mean(values)
        stats[f"{metric}_sd"] = stdev(values) if len(values) > 1 else None
    return stats


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, metric: str, label: str, percent: bool, curves: dict) -> None:
    figure = Figure(figsize=(8, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    styles = (
        ("#0072B2", "o", "-"), ("#007F63", "s", "--"), ("#8E507D", "^", "-."),
        ("#B64F00", "D", "-"), ("#4B555C", "v", "--"), ("#9B6500", "P", "-."),
    )
    for (group, rows), (color, marker, line) in zip(curves.items(), styles, strict=True):
        epochs = [row["epoch"] for row in rows]
        means = [row[f"{metric}_mean"] for row in rows]
        spreads = [row[f"{metric}_sd"] for row in rows]
        ax.plot(
            epochs, means, label=group, color=color, marker=marker, linestyle=line,
            markersize=4, markevery=max(1, len(epochs) // 10),
        )
        if spreads[0] is not None:
            ax.fill_between(
                epochs,
                [m - s for m, s in zip(means, spreads, strict=True)],
                [m + s for m, s in zip(means, spreads, strict=True)],
                color=color, alpha=0.13, linewidth=0,
            )
    n = next(iter(curves.values()))[0]["n_seeds"]
    title = "One seed" if n == 1 else f"Mean across {n} seeds · bands show ±1 sample SD"
    ax.set(xlabel="Epoch", ylabel=label, title=title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    figure.savefig(path, dpi=160)


def compare_runs(run_ids: Iterable[str], output: Path, *, root: Path = ROOT) -> None:
    by_group = check_controls([read_run(root, run_id) for run_id in run_ids])
    curves = {}
    final_runs = []
    for group, runs in by_group.items():
        config = runs[0].config
        curves[group] = [
            {
                "group": group,
                "conjunction": config.ltn.conjunction,
                "epoch": epoch,
                "exists_p": config.ltn.exists_p_at(epoch),
                **aggregate([run.epochs[epoch - 1] for run in runs]),
            }
            for epoch in range(1, config.epochs + 1)
        ]
        final_runs.extend(
            {
                "group": group,
                "seed": run.config.seed,
                "run_id": run.run_id,
                "epoch": config.epochs,
                **run.epochs[-1],
            }
            for run in runs
        )
    output.mkdir(parents=True, exist_ok=False)
    write_table(output / "final_runs.csv", final_runs)
    write_table(output / "summary.csv", [rows[-1] for rows in curves.values()])
    write_table(output / "epoch_summary.csv", [row for rows in curves.values() for row in rows])
    for name, (metric, label, percent) in PLOTS.items():
        plot_comparison(output / f"{name}.png", metric, label, percent, curves)
    write_json(output / "manifest.json", {
        "runs": [
            {"run_id": run.run_id, "metrics": str(root / "experiments" / run.run_id / "metrics.csv"),
             "config": asdict(run.config)}
            for runs in by_group.values() for run in runs
        ],
        "selection": "fixed final epoch, identical seed set in each treatment",
        "aggregation": "arithmetic mean and sample standard deviation across seeds (ddof=1)",
        "plot_bands": "mean ± 1 sample SD; SD is omitted when n_seeds is 1",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed A1–B3 runs.")
    parser.add_argument("--run-id", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        compare_runs(args.run_id, args.output)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
