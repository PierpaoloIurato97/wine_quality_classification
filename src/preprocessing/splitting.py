import pandas as pd

import config
import utils


def shuffle(df: pd.DataFrame):
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_rows = len(df)
    train_end = int(total_rows * config.TRAIN_RATIO)
    validation_end = train_end + int(total_rows * config.VALIDATION_RATIO)

    train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    test_df = df.iloc[validation_end:total_rows]

    return train_df, validation_df, test_df


def split():
    df = utils.read_csv("winequality-red-with-label-standardized")

    df = shuffle(df)
    train_df, validation_df, test_df = split_df(df)

    utils.save_csv(train_df, "winequality-red-train")
    utils.save_csv(validation_df, "winequality-red-validation")
    utils.save_csv(test_df, "winequality-red-test")
