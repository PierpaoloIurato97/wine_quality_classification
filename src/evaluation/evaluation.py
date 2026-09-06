import matplotlib.pyplot as plt
import torch

import config
import utils
from model import WineQualityClassifier


def get_data(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    df = utils.read_csv(f"winequality-red-{split}-standardized")
    return utils.split_df_for_inference(df)


def test_step(model: WineQualityClassifier, input: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        pred = model.forward(input)
    return pred.argmax(dim=1)


def calculate_confusion_matrix(
    pred: torch.Tensor, labels: torch.Tensor
) -> tuple[int, int, int, int]:
    true_positives = int(((pred == 1) & (labels == 1)).sum().item())
    true_negatives = int(((pred == 0) & (labels == 0)).sum().item())
    false_positives = int(((pred == 1) & (labels == 0)).sum().item())
    false_negatives = int(((pred == 0) & (labels == 1)).sum().item())

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
    input, labels = get_data(split)
    input = input.to(config.DEVICE)
    labels = labels.to(config.DEVICE)

    pred = test_step(model, input)
    num_correct = int((pred == labels).sum().item())

    accuracy = utils.calculate_accuracy(num_correct, input.shape[0])
    true_positives, true_negatives, false_positives, false_negatives = (
        calculate_confusion_matrix(pred, labels)
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
    model = WineQualityClassifier()
    utils.load_model(model, "wine_quality_model")
    model = model.to(config.DEVICE)

    for split in ["train", "validation", "test"]:
        evaluate_split(model, split)
