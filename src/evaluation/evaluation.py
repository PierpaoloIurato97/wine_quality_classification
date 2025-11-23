import matplotlib.pyplot as plt
import torch
from model import WineQualityClassifier

import utils


def get_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    df = utils.read_csv("winequality-red-test")
    return utils.split_df_for_inference(df, 'label')


def test_step(model: WineQualityClassifier, input: torch.Tensor) -> torch.Tensor:
    utils.eval_mode(model)
    pred = model.forward(input).argmax(dim=1)
    return pred


def calculate_confusion_matrix(
        pred: torch.Tensor,
        labels: torch.Tensor
) -> tuple[int, int, int, int]:
    true_positives = ((pred >= 0.5) & (labels == 1)).sum().item()
    true_negatives = ((pred < 0.5) & (labels == 0)).sum().item()
    false_positives = ((pred >= 0.5) & (labels == 0)).sum().item()
    false_negatives = ((pred < 0.5) & (labels == 1)).sum().item()

    return true_positives, true_negatives, false_positives, false_negatives


def plot_confusion_matrix(
        true_positives: torch.types.Number,
        true_negatives: torch.types.Number,
        false_positives: torch.types.Number,
        false_negatives: torch.types.Number
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

    test_input, test_labels = get_test_data()
    dataset_len = test_input.shape[0]

    pred = test_step(model, test_input)
    num_correct = (pred == test_labels).sum().item()

    accurancy = utils.calculate_accuracy(num_correct, dataset_len)
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
    print_accuracy(accurancy)
