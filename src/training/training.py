from typing import cast

import torch

import config
import utils
from model import WineQualityClassifier


def get_train_loader() -> tuple[torch.utils.data.DataLoader, int]:
    df = utils.read_csv("winequality-red-train-standardized")

    input, labels = utils.split_df_for_inference(df)
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
    df = utils.read_csv("winequality-red-validation-standardized")
    return utils.split_df_for_inference(df)


def train_step(
    model: WineQualityClassifier,
    input: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Adam,
    loss_function: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    model.train()

    optimizer.zero_grad()
    loss = loss_function(model(input), labels)
    loss.backward()
    optimizer.step()

    return loss


def train_batch(
    model: WineQualityClassifier,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    loss_function: torch.nn.CrossEntropyLoss,
) -> float:
    total_loss = 0.0

    for _, (input, labels) in enumerate(loader):
        input = cast(torch.Tensor, input)
        labels = cast(torch.Tensor, labels)

        input = input.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        loss = train_step(model, input, labels, optimizer, loss_function)
        total_loss += loss.item()

    return total_loss / len(loader)


def val_loss(
    model: WineQualityClassifier,
    input: torch.Tensor,
    labels: torch.Tensor,
    loss_function: torch.nn.CrossEntropyLoss,
) -> float:
    model.eval()

    with torch.no_grad():
        return loss_function(model(input), labels).item()


def print_performance(
    epoch: int, train_loss: float, val_loss_value: float, lr: float
) -> None:
    print(
        f"Epoch {epoch + 1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss_value:.4f}, LR: {lr:.6f}"
    )


def compute_class_weights(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    class_counts = torch.zeros(2)
    for _, labels in loader:
        for c in range(2):
            class_counts[c] += (labels == c).sum()
    weights = class_counts.sum() / (2 * class_counts)
    return weights


def train() -> None:
    model = WineQualityClassifier()
    model = model.to(config.DEVICE)

    train_loader, dataset_len = get_train_loader()

    class_weights = compute_class_weights(train_loader).to(config.DEVICE)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.ADAM_BETAS,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_PLATEAU_FACTOR,
        patience=config.LR_PLATEAU_PATIENCE,
        min_lr=config.LR_MIN,
    )

    val_input, val_labels = get_val_data()
    val_input = val_input.to(config.DEVICE)
    val_labels = val_labels.to(config.DEVICE)

    best_val_loss = float("inf")
    best_epoch = 0

    try:
        for epoch in range(config.EPOCHS):
            train_loss = train_batch(model, train_loader, optimizer, loss_function)

            val_loss_value = val_loss(model, val_input, val_labels, loss_function)

            scheduler.step(val_loss_value)
            print_performance(
                epoch, train_loss, val_loss_value, optimizer.param_groups[0]["lr"]
            )

            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                best_epoch = epoch + 1
                utils.save_model(model, "wine_quality_model")

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        print(
            f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch}/{config.EPOCHS})"
        )
