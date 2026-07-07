import matplotlib.pyplot as plt
import torch

import config
import utils
from model import WineQualityClassifier


def get_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    df = utils.read_csv("winequality-red-test")
    return utils.split_df_for_inference(df)


def test_step(model: WineQualityClassifier, input: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        pred = model.forward(input)
    return pred.argmax(dim=1)


def calculate_confusion_matrix(
        pred: torch.Tensor,
        labels: torch.Tensor
) -> tuple[int, int, int, int]:
    true_positives = ((pred == 1) & (labels == 1)).sum().item()
    true_negatives = ((pred == 0) & (labels == 0)).sum().item()
    false_positives = ((pred == 1) & (labels == 0)).sum().item()
    false_negatives = ((pred == 0) & (labels == 1)).sum().item()

    return true_positives, true_negatives, false_positives, false_negatives


def plot_confusion_matrix(
        true_positives: int,
        true_negatives: int,
        false_positives: int,
        false_negatives: int
) -> None:
    cm = [[true_positives, false_negatives],
          [false_positives, true_negatives]]

    plt.title('Confusion Matrix')
    plt.imshow(cm, cmap='Blues')

    tick_marks = range(2)
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['Positive', 'Negative'])

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i,
                str(cm[i][j]),
                horizontalalignment="center",
                color="black"
            )


def print_accuracy(accuracy: float) -> None:
    print(f"Accuracy: {accuracy * 100:.2f}%")


def evaluate():
    model = WineQualityClassifier()
    utils.load_model(model, 'wine_quality_model')
    model = model.to(config.DEVICE)

    test_input, test_labels = get_test_data()
    test_input = test_input.to(config.DEVICE)
    test_labels = test_labels.to(config.DEVICE)

    dataset_len = test_input.shape[0]

    pred = test_step(model, test_input)
    num_correct = (pred == test_labels).sum().item()

    accuracy = utils.calculate_accuracy(num_correct, dataset_len)
    true_positives, true_negatives, false_positives, false_negatives = calculate_confusion_matrix(
        pred, test_labels
    )

    utils.make_plot(
        title='Confusion Matrix',
        file_name='confusion_matrix',
        plot=lambda: plot_confusion_matrix(
            true_positives, true_negatives, false_positives, false_negatives
        )
    )
    print_accuracy(accuracy)
