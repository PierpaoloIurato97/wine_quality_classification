from typing import Callable, cast

import torch

import config
import utils
from model import WineQualityClassifier


def get_train_loader() -> tuple[torch.utils.data.DataLoader, int]:
    df = utils.read_csv("winequality-red-train")

    input, labels = utils.split_df_for_inference(df, 'label')
    labels = labels.long()

    dataset_len = input.shape[0]

    dataset: torch.utils.data.TensorDataset = torch.utils.data.TensorDataset(
        input, labels
    )
    loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True
    )

    return loader, dataset_len


def get_val_data() -> tuple[torch.Tensor, torch.Tensor]:
    df = utils.read_csv("winequality-red-validation")
    return utils.split_df_for_inference(df, 'label')


def train_step(
    model: WineQualityClassifier,
    input: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Adam,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    utils.train_mode(model)

    optimizer.zero_grad()
    pred = model.forward(input)
    loss = loss_function(pred, labels)
    loss.backward()
    optimizer.step()

    pred = pred.argmax(dim=1)

    return pred


def train_batch(
    model: WineQualityClassifier,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> int:
    num_correct = 0

    for _, (input, labels) in enumerate(loader):
        input = cast(torch.Tensor, input)
        labels = cast(torch.Tensor, labels)

        input = input.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

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

    pred = model.forward(input).argmax(dim=1)
    num_correct = (pred == labels).sum().item()

    return num_correct


def print_performance(epoch: int, train_accurancy: float, val_accurancy: float) -> None:
    print(
        f"Epoch {epoch + 1}/{config.EPOCHS}, Train Accurancy: {train_accurancy}, Val Accurancy: {val_accurancy}"
    )


def train() -> None:
    model = WineQualityClassifier()
    model = model.to(config.DEVICE)

    loss_function = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    train_loader, dataset_len = get_train_loader()

    val_input, val_labels = get_val_data()
    val_input = val_input.to(config.DEVICE)
    val_labels = val_labels.to(config.DEVICE)

    try:
        for epoch in range(config.EPOCHS):
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

            train_accurancy = utils.calculate_accuracy(
                num_correct, dataset_len
            )
            val_accurancy = utils.calculate_accuracy(
                val_correct, val_input.shape[0]
            )

            print_performance(epoch, train_accurancy, val_accurancy)

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        print("Saving model...")
        utils.save_model(model, 'wine_quality_model')
