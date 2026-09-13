import matplotlib.pyplot as plt
import numpy as np

import utils
from model import WineQualityClassifier


def get_data(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the standardized feature matrix and label vector for a given split.

    Abstracts file-loading so that evaluate_split can request any split
    (train / validation / test) by name without knowing the CSV naming scheme.
    """
    df = utils.read_csv(f"winequality-red-{split}-standardized")
    return utils.split_df(df)


def calculate_confusion_matrix(
    pred: np.ndarray, labels: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Computes the four cells of a binary confusion matrix.

    Provides a more detailed picture of model errors than accuracy alone:
    false positives and false negatives reveal different kinds of mistakes
    that may matter differently in a wine quality context.
    """
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
    """
    Renders and saves a visual confusion matrix for a single split.

    The saved plot makes it easy to inspect the distribution of correct and
    incorrect predictions at a glance without reading raw numbers.
    """
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
    """
    Prints a formatted accuracy line for a given split.

    Centralises the output format so that train, validation, and test
    accuracy lines look consistent in the pipeline's console output.
    """
    print(f"{split} Accuracy: {accuracy * 100:.2f}%")


def evaluate_split(model: WineQualityClassifier, split: str) -> None:
    """
    Evaluates the trained model on a single data split.

    Combines accuracy reporting and confusion-matrix plotting into one call
    so that evaluate() can iterate over all splits uniformly.
    """
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
    """
    Entry point for the evaluation stage of the pipeline.

    Loads the persisted model and runs evaluate_split over train, validation,
    and test sets. Reporting all three splits together highlights whether the
    model is overfitting (high train accuracy, low test accuracy) or
    generalising well.
    """
    model = utils.load_model("wine_quality_model")

    for split in ["train", "validation", "test"]:
        evaluate_split(model, split)
