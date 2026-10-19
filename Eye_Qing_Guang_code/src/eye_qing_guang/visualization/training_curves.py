"""Plot loss and validation selection metric from a completed history."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eye_qing_guang.utils.experiment_io import read_csv


def save_figure(figure, output: Path):
    if output.suffix.lower() != ".png":
        raise ValueError("Figure output must have a .png extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            figure.savefig(handle, format="png", dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_training_curves(history: Path, output: Path, metric="roc_auc"):
    rows = read_csv(history)
    if not rows or metric not in rows[0]:
        raise ValueError("History is empty or requested metric is absent.")
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="Train")
    axes[0].plot(epochs, [float(row["valid_loss"]) for row in rows], label="Valid")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [float(row[metric]) if row[metric] else float("nan") for row in rows])
    axes[1].set(xlabel="Epoch", ylabel=metric)
    figure.tight_layout()
    save_figure(figure, output)
