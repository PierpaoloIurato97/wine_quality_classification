import config
import utils
from model import WineQualityClassifier


def train() -> None:
    """
    Entry point for the training stage of the pipeline.

    Loads the standardized training set, trains an SVM classifier with the
    hyperparameters defined in config, and persists the resulting model to
    disk so that the evaluation step can load and benchmark it.
    """
    df_train = utils.read_csv("winequality-red-train-standardized")
    X_train, y_train = utils.split_df(df_train)

    model = WineQualityClassifier()
    model.fit(X_train, y_train, C=config.C, gamma=config.GAMMA)

    pred = model.predict(X_train)
    train_accuracy = float((pred == y_train).sum() / len(y_train))
    print(f"Hyperparameters: C={config.C}, gamma={config.GAMMA}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")

    utils.save_model(model, "wine_quality_model")
    print("Model saved successfully.")
