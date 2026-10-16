"""Redraw training and evaluation charts from saved CSVs, without retraining."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def read_results(path, columns, optional=()):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = []
        for line, row in enumerate(reader, start=2):
            try:
                values = {column: float(row[column]) for column in columns}
                values.update({column: float(row[column]) for column in optional
                               if row.get(column) not in (None, "")})
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid number on line {line}") from error
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{path}: non-finite number on line {line}")
            rows.append(values)
    if not rows:
        raise ValueError(f"{path}: no episodes to plot")
    return rows


def plot_results(training=None, evaluation=None, output=Path("Output/results.png"),
                 title="Snake learning results"):
    if training is None and evaluation is None:
        raise ValueError("provide training and/or evaluation results")
    training_rows = read_results(training, ("episode", "score", "rolling_average"),
                                 optional=("epsilon_used", "epsilon")) if training else []
    evaluation_rows = read_results(evaluation, ("score",)) if evaluation else []

    panels = (2 if training_rows else 0) + bool(evaluation_rows)
    figure = Figure(figsize=(10, 3 * panels), layout="constrained")
    FigureCanvasAgg(figure)  # Save files without opening a GUI or requiring a display.
    axes = list(figure.subplots(panels, 1, squeeze=False)[:, 0])
    figure.suptitle(title, fontsize=14)

    if training_rows:
        score_axis, epsilon_axis = axes[:2]
        episodes = [row["episode"] + 1 for row in training_rows]
        score_axis.plot(episodes, [row["score"] for row in training_rows],
                        color="#94a3b8", alpha=0.6, linewidth=0.8, label="Episode score")
        score_axis.plot(episodes, [row["rolling_average"] for row in training_rows],
                        color="#2563eb", linewidth=2, label="Rolling average (up to 100 episodes)")
        score_axis.set(title="Training episodes", xlabel="Training episode", ylabel="Food eaten")
        score_axis.set_ylim(bottom=0)
        score_axis.legend(loc="upper left")
        epsilon = [row.get("epsilon_used", row.get("epsilon")) for row in training_rows]
        if all(value is not None for value in epsilon):
            epsilon_axis.plot(episodes, epsilon, color="#d97706")
            epsilon_axis.set(title="Exploration probability", xlabel="Training episode", ylabel="Epsilon", ylim=(-0.02, 1.02))
        else:
            epsilon_axis.text(0.5, 0.5, "No epsilon recorded (random agents do not use epsilon)",
                              ha="center", va="center", transform=epsilon_axis.transAxes)
            epsilon_axis.set_axis_off()

    if evaluation_rows:
        scores = [row["score"] for row in evaluation_rows]
        evaluation_axis = axes[-1]
        evaluation_axis.hist(scores, bins="auto", color="#0f766e", edgecolor="white")
        evaluation_axis.axvline(mean(scores), color="#b45309", linewidth=2, label=f"Mean: {mean(scores):.2f}")
        evaluation_axis.set(title=f"Evaluation only — {len(scores)} episodes | median {median(scores):g} | best {max(scores):g}",
                            xlabel="Food eaten per game", ylabel="Episode count")
        evaluation_axis.legend()

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", help="Experiment folder written by agent.py")
    parser.add_argument("--training", type=Path, help="Training CSV (also supports older exports)")
    parser.add_argument("--evaluation", type=Path, help="Evaluation CSV (also supports older exports)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="Snake learning results")
    args = parser.parse_args(argv)
    if args.run_dir and (args.training or args.evaluation):
        parser.error("use a run folder or explicit CSV paths, not both")
    if args.run_dir:
        if not args.run_dir.is_dir():
            parser.error(f"not a run folder: {args.run_dir}")
        args.training = args.run_dir / "training.csv"
        args.evaluation = args.run_dir / "evaluation.csv"
        args.training = args.training if args.training.is_file() else None
        args.evaluation = args.evaluation if args.evaluation.is_file() else None
    output = args.output or (args.run_dir or Path("Output")) / "results.png"
    try:
        path = plot_results(args.training, args.evaluation, output, args.title)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved plot: {path}")
    return path


if __name__ == "__main__":
    main()
