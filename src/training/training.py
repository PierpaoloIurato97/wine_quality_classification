from typing import Callable

import pandas as pd
import torch
import utils
from model import WineQualityClassifier

EPOCHS = 10000
BATCH_SIZE = 32
EARLY_STOPPING_DELTA = 10.0


def get_train_data() -> tuple[torch.Tensor, torch.Tensor]:
    df_train = utils.read_csv("winequality-red-train")
    return utils.split_df_for_training(df_train, 'label')


def get_val_data() -> tuple[torch.Tensor, torch.Tensor]:
    df_val = utils.read_csv("winequality-red-validation")
    return utils.split_df_for_training(df_val, 'label')


def train_step(
    model: WineQualityClassifier,
    input: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Adam,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> float:
    utils.train_mode(model)

    optimizer.zero_grad()
    pred = model.forward(input).squeeze()
    loss = loss_function(pred, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_batch(
    model: WineQualityClassifier,
    input: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Adam,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> float:
    train_loss = 0.0

    for batch in range(BATCH_SIZE):
        input_batch = input[batch::BATCH_SIZE]
        labels_batch = labels[batch::BATCH_SIZE]

        train_loss += train_step(
            model, input_batch, labels_batch, optimizer, loss_function
        )

    return train_loss


def val_step(
    model: WineQualityClassifier,
    input: torch.Tensor,
    labels: torch.Tensor,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> float:
    utils.eval_mode(model)

    pred = model.forward(input).squeeze()
    loss = loss_function(pred, labels)

    return loss.item()


def print_performance(epoch: int, train_loss: float, val_loss: float) -> None:
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss}"
    )


def early_stopping(val_loss: float, pred_val_loss: float) -> bool:
    return (val_loss - pred_val_loss) > EARLY_STOPPING_DELTA


def shuffle_data(input: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.shape[0])

    return input[indices], labels[indices]


def train() -> None:
    model = WineQualityClassifier()
    loss_function = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_input, train_labels = get_train_data()
    val_input, val_labels = get_val_data()

    pred_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_batch(
            model,
            train_input,
            train_labels,
            optimizer,
            loss_function
        )

        val_loss = val_step(
            model,
            val_input,
            val_labels,
            loss_function
        )

        print_performance(epoch, train_loss, val_loss)

        if early_stopping(val_loss, pred_val_loss):
            print("Early stopping...")
            break

        pred_val_loss = val_loss
        train_input, train_labels = shuffle_data(train_input, train_labels)

    utils.save_model(model, 'wine_quality_model')


train()
