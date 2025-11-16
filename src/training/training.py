import pandas as pd
import torch
import utils
from model import WineQualityClassifier

EPOCHS = 10000
BATCH_SIZE = 32
EARLY_STOPPING_DELTA = 2.0


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


def train_batch(model, input, labels, optimizer, loss_function):
    train_loss = 0.0

    for batch in range(BATCH_SIZE):
        input_batch = input[batch::BATCH_SIZE]
        labels_batch = labels[batch::BATCH_SIZE]

        train_loss += train_step(
            model, input_batch, labels_batch, optimizer, loss_function
        )

    return train_loss


def val_step(model, input, labels, loss_function):
    eval_mode(model)

    pred = model.forward(input).squeeze()
    loss = loss_function(pred, labels)

    return loss.item()


def print_performance(epoch, train_loss, val_loss):
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss}"
    )


def early_stopping(val_loss, pred_val_loss) -> bool:
    return (val_loss - pred_val_loss) > EARLY_STOPPING_DELTA


def shuffle_data(input, labels):
    indices = torch.randperm(input.shape[0])
    return input[indices], labels[indices]


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
        train_loss = train_batch(
            wine_quality_classifier,
            train_input,
            train_labels,
            optimizer,
            loss_function
        )

        val_loss = val_step(
            wine_quality_classifier,
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


train()
