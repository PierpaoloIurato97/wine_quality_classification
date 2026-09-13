import matplotlib.pyplot as plt
import numpy as np

import utils
from model import WineQualityClassifier


def get_data(split: str) -> tuple[np.ndarray, np.ndarray]:
    df = utils.read_csv(f"winequality-red-{split}-standardized")
    return utils.split_df(df)


def calculate_confusion_matrix(
    pred: np.ndarray, labels: np.ndarray
) -> tuple[int, int, int, int]:
    true_positives = int(((pred == 1) & (labels == 1)).sum())
    true_negatives = int(((pred == 0) & (labels == 0)).sum())
    false_positives = int(((pred == 1) & (labels == 0)).sum())
    false_negatives = int(((pred == 0) & (labels == 1)).sum())

    return true_positives, true_negatives, false_positives, false_negatives


def plot_confusion_matrix(
    true_positives: int,
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
    title: str = "Confusion Matrix",
) -> None:
    cm = [[true_positives, false_negatives], [false_positives, true_negatives]]

    plt.title(title)
    plt.imshow(cm, cmap="Blues")

    tick_marks = range(2)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(tick_marks, ["Positive", "Negative"])
    plt.yticks(tick_marks, ["Positive", "Negative"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), horizontalalignment="center", color="black")


def print_accuracy(split: str, accuracy: float) -> None:
    print(f"{split} Accuracy: {accuracy * 100:.2f}%")


def evaluate_split(model: WineQualityClassifier, split: str) -> None:
    X, y = get_data(split)

    pred = model.predict(X)
    num_correct = int((pred == y).sum())

    accuracy = utils.calculate_accuracy(num_correct, X.shape[0])
    true_positives, true_negatives, false_positives, false_negatives = (
        calculate_confusion_matrix(pred, y)
    )

    title = f"Confusion Matrix - {split.capitalize()}"
    utils.make_plot(
        title=title,
        file_name=f"confusion_matrix_{split}",
        plot=lambda: plot_confusion_matrix(
            true_positives, true_negatives, false_positives, false_negatives, title
        ),
    )
    print_accuracy(split.capitalize(), accuracy)


def evaluate():
    model = utils.load_model("wine_quality_model")

    for split in ["train", "validation", "test"]:
        evaluate_split(model, split)
