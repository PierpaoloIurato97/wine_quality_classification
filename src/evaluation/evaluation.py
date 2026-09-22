import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import utils
from model import WineQualityClassifier


def get_data(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the standardized feature matrix and label vector for a given split.

    Abstracts file-loading so that evaluate_split can request any split
    (train / test) by name without knowing the CSV naming scheme.
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


def plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, auc: float, title: str = "ROC Curve"
) -> None:
    """
    Renders and saves a visual ROC Curve.

    Shows the trade-off between True Positive Rate and False Positive Rate
    at various threshold settings.
    """
    plt.title(title)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")


def print_metrics(split: str, accuracy: float, auc: float) -> None:
    """
    Prints formatted metrics (accuracy and AUC) for a given split.
    """
    print(f"{split} - Accuracy: {accuracy * 100:.2f}% | AUC: {auc * 100:.2f}%")


def evaluate_split(model: WineQualityClassifier, split: str) -> None:
    """
    Evaluates the trained model on a single data split.

    Combines accuracy and AUC reporting alongside confusion-matrix and ROC-curve
    plotting into one call so that evaluate() can iterate over all splits uniformly.
    """
    X, y = get_data(split)

    pred = model.predict(X)
    num_correct = int((pred == y).sum())

    accuracy = utils.calculate_accuracy(num_correct, X.shape[0])

    scores = model.decision_function(X)
    auc = roc_auc_score(y, scores)
    fpr, tpr, _ = roc_curve(y, scores)

    true_positives, true_negatives, false_positives, false_negatives = (
        calculate_confusion_matrix(pred, y)
    )

    title_cm = f"Confusion Matrix - {split.capitalize()}"
    utils.make_plot(
        title=title_cm,
        file_name=f"confusion_matrix_{split}",
        plot=lambda: plot_confusion_matrix(
            true_positives, true_negatives, false_positives, false_negatives, title_cm
        ),
    )

    title_roc = f"ROC Curve - {split.capitalize()}"
    utils.make_plot(
        title=title_roc,
        file_name=f"roc_curve_{split}",
        plot=lambda: plot_roc_curve(fpr, tpr, auc, title_roc),
    )

    print_metrics(split.capitalize(), accuracy, auc)


def evaluate():
    """
    Entry point for the evaluation stage of the pipeline.

    Loads the persisted model and runs evaluate_split over train and test sets.
    Reporting both splits together highlights whether the model is overfitting
    (high train metrics, low test metrics) or generalising well.
    """
    model = utils.load_model("wine_quality_model")

    for split in ["train", "test"]:
        evaluate_split(model, split)
