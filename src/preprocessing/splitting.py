import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
import utils


def mark_outliers(df: pd.DataFrame) -> pd.Series:
    is_outlier = pd.Series(False, index=df.index)

    for feature in config.FEATURES:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        is_outlier = is_outlier | (df[feature] < lower) | (df[feature] > upper)

    return is_outlier


def build_stratify_key(df: pd.DataFrame) -> np.ndarray:
    is_outlier = mark_outliers(df).astype(int)
    return (df[config.LABEL].astype(str) + '_' + is_outlier.astype(str)).to_numpy()


def split():
    df = utils.read_csv("winequality-red-with-label")

    stratify_key = build_stratify_key(df)

    val_test_ratio = config.VALIDATION_RATIO + config.TEST_RATIO
    train_df, temp_df, _, temp_key = train_test_split(
        df, stratify_key,
        test_size=val_test_ratio,
        random_state=config.RANDOM_STATE,
        stratify=stratify_key
    )

    temp_stratify_key = build_stratify_key(temp_df)
    relative_test_ratio = config.TEST_RATIO / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=config.RANDOM_STATE,
        stratify=temp_stratify_key
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        key = build_stratify_key(split_df)
        unique, counts = np.unique(key, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  {name}: {dist}")

    utils.save_csv(train_df, "winequality-red-train")
    utils.save_csv(val_df, "winequality-red-validation")
    utils.save_csv(test_df, "winequality-red-test")
