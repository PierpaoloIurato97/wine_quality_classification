from typing import Callable

import torch
import utils
from model import WineQualityClassifier

EPOCHS = 10000
BATCH_SIZE = 64
EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_DELTA = 1.0


def get_train_loader() -> tuple[torch.utils.data.DataLoader, int]:
    df_train = utils.read_csv("winequality-red-train")
    train_input, train_labels = utils.split_df_for_training(df_train, 'label')
    dataset = torch.utils.data.TensorDataset(train_input, train_labels)

    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True), train_input.shape[0]


def get_val_data() -> tuple[torch.Tensor, torch.Tensor]:
    df_val = utils.read_csv("winequality-red-validation")
    return utils.split_df_for_training(df_val, 'label')


def train_step(
    model: WineQualityClassifier,
    input: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Adam,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    utils.train_mode(model)

    optimizer.zero_grad()
    pred = model.forward(input).squeeze()
    loss = loss_function(pred, labels)
    loss.backward()
    optimizer.step()

    return pred


def train_batch(
    model: WineQualityClassifier,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> int:
    num_correct = 0

    for _, (input, labels) in enumerate(loader):
        pred = train_step(
            model, input, labels, optimizer, loss_function
        )

        num_correct += (pred == labels).sum().item()

    return num_correct


def val_step(
    model: WineQualityClassifier,
    input: torch.Tensor,
    labels: torch.Tensor,
) -> int:
    utils.eval_mode(model)

    pred = model.forward(input).squeeze()
    num_correct = (pred == labels).sum().item()

    return int(num_correct)


def calculate_accuracy(num_correct: int, dataset_len: int) -> float:
    return round(num_correct / dataset_len, 2)


def print_performance(epoch: int, train_accurancy: float, val_accurancy: float) -> None:
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Accurancy: {train_accurancy}, Val Accurancy: {val_accurancy}"
    )


def early_stopping(val_accurancy: float, pred_val_accurancy: float) -> bool:
    return (val_accurancy - pred_val_accurancy) < EARLY_STOPPING_DELTA


def train() -> None:
    model = WineQualityClassifier()
    loss_function = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader, dataset_len = get_train_loader()
    val_input, val_labels = get_val_data()

    try:
        pred_val_accurancy = float('inf')
        for epoch in range(EPOCHS):
            num_correct = train_batch(
                model,
                train_loader,
                optimizer,
                loss_function
            )

            val_correct = val_step(
                model,
                val_input,
                val_labels
            )

            train_accurancy = calculate_accuracy(num_correct, dataset_len)
            val_accurancy = calculate_accuracy(val_correct, val_input.shape[0])

            print_performance(epoch, train_accurancy, val_accurancy)

            if early_stopping(val_accurancy, pred_val_accurancy) and EARLY_STOPPING_ENABLED:
                print("Early stopping...")
                break

            if val_accurancy > pred_val_accurancy:
                pred_val_accurancy = val_accurancy

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        print("Saving model...")
        utils.save_model(model, 'wine_quality_model')


train()
