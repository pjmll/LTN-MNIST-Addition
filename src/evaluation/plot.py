import argparse
import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def plot_learning_curves(metrics_path: Path, output_path: Path) -> None:
    """Render training loss and validation accuracies from the run's CSV."""
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The metrics file contains no completed epochs")
    epochs = [int(row["epoch"]) for row in rows]
    show_activation = bool(rows[0].get("train_conjunction_activation_rate"))
    figure = Figure(figsize=(8, 7.5 if show_activation else 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3 if show_activation else 2, 1, sharex=True)
    loss, accuracy = axes[:2]
    loss.plot(epochs, [float(row["train_loss"]) for row in rows], "o-", color="#403075")
    ltn_run = bool(rows[0].get("train_satisfaction"))
    loss.set_ylabel("Training loss (1 − Sat)" if ltn_run else "Training NLL")
    accuracy.plot(
        epochs, [float(row["val_hard_sum_accuracy"]) for row in rows],
        "o-", color="#00689d", label="HardSum",
    )
    accuracy.plot(
        epochs, [float(row["val_prob_sum_accuracy"]) for row in rows],
        "s--", color="#ad481c", label="ProbSum",
    )
    accuracy.set(ylabel="Validation accuracy", ylim=(0, 1))
    accuracy.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    accuracy.legend()
    if show_activation:
        diagnostics = axes[2]
        diagnostics.plot(
            epochs, [float(row["train_conjunction_activation_rate"]) for row in rows],
            "o-", color="#14735c", label="Active legal candidates",
        )
        diagnostics.plot(
            epochs, [float(row["train_zero_grad_fraction"]) for row in rows],
            "s--", color="#842f6c", label="Zero-gradient batches",
        )
        diagnostics.set_ylabel("Training fraction")
        diagnostics.set_ylim(bottom=0)
        diagnostics.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        diagnostics.legend()
    axes[-1].set_xlabel("Epoch")
    if len(epochs) <= 12:
        axes[-1].set_xticks(epochs)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        if ltn_run:
            for previous, current in zip(rows, rows[1:]):
                if previous["exists_p"] != current["exists_p"]:
                    axis.axvline(int(current["epoch"]), color="#656565", linestyle=":", alpha=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a run's training and validation curves.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_learning_curves(args.metrics, args.output)


if __name__ == "__main__":
    main()
