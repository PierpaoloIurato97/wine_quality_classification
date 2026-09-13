from typing import cast

import pandas as pd

import config
import utils


def remove_outliers() -> None:
    """
    First preprocessing step in the pipeline.

    Removes rows from the raw dataset whose feature values fall outside the
    physically plausible ranges defined in config.PLAUSIBLE_RANGES. This guards
    against measurement errors or corrupted entries that would otherwise skew
    the model and descriptive statistics.
    """
    df = utils.read_csv("winequality-red")

    rows_before = len(df)

    for feature, (low, high) in config.PLAUSIBLE_RANGES.items():
        utils.ensure_col_exists(df, feature)
        df = cast(pd.DataFrame, df[(df[feature] >= low) & (df[feature] <= high)])

    rows_after = len(df)
    removed = rows_before - rows_after

    print(f"Rows before filtering: {rows_before}")
    print(f"Rows after filtering:  {rows_after}")
    print(f"Rows removed:          {removed} ({removed / rows_before * 100:.1f}%)")

    utils.save_csv(df, "winequality-red-filtered")
