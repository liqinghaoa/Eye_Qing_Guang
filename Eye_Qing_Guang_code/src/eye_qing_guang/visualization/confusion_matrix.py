"""Plot the confusion matrix at the recorded evaluation threshold."""

import json
from pathlib import Path

from eye_qing_guang.visualization.training_curves import save_figure

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(metrics_path: Path, output: Path):
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center",
                  color="white" if value > matrix.max() / 2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label",
             title=f"Threshold = {metrics['threshold']:g}")
    save_figure(figure, output)
