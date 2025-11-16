import pandas as pd
import torch
import utils
from model import WineQualityClassifier

EPOCHS = 10000


def split_data(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    input = torch.from_numpy(
        df.drop(columns=['label']).to_numpy()
    ).float()
    labels = torch.from_numpy(df['label'].to_numpy()).float()

    return input, labels


def get_train_data() -> tuple[torch.Tensor, torch.Tensor]:
    df_train = utils.read_csv("winequality-red-train")

    return split_data(df_train)


def get_val_data() -> tuple[torch.Tensor, torch.Tensor]:
    df_val = utils.read_csv("winequality-red-validation")

    return split_data(df_val)


def get_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    df_test = utils.read_csv("winequality-red-test")

    return split_data(df_test)


def train_mode(model):
    torch.set_grad_enabled(True)
    model.train()


def eval_mode(model):
    torch.set_grad_enabled(False)
    model.eval()


def train_step(model, input, labels, optimizer, loss_function):
    train_mode(model)

    optimizer.zero_grad()
    pred = model.forward(input).squeeze()
    loss = loss_function(pred, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def val_step(model, input, labels, loss_function):
    eval_mode(model)

    pred = model.forward(input).squeeze()
    loss = loss_function(pred, labels)

    return loss.item()


def test_step(model, input, labels, loss_function):
    eval_mode(model)

    pred = model.forward(input).squeeze()
    loss = loss_function(pred, labels)

    return loss.item()


def train():
    wine_quality_classifier = WineQualityClassifier()
    loss_function = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(
        wine_quality_classifier.parameters(), lr=0.001
    )
    train_input, train_labels = get_train_data()
    val_input, val_labels = get_val_data()

    pred_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_step(
            wine_quality_classifier, train_input, train_labels, optimizer, loss_function
        )
        val_loss = val_step(
            wine_quality_classifier, val_input, val_labels, loss_function
        )
        test_loss = test_step(
            wine_quality_classifier, val_input, val_labels, loss_function
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}"
        )

        if val_loss > pred_val_loss:
            print("Early stopping...")
            break

        pred_val_loss = val_loss


train()
