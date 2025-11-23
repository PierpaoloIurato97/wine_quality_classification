from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

import config
import utils


def plot_class_distribution(df: pd.DataFrame):
    plt.hist(df[config.LABEL], bins=2, edgecolor='black')


def plot_feature_correlation(df: pd.DataFrame):
    num_rows, num_cols = len(config.FEATURES), len(config.FEATURES)

    _, axes = plt.subplots(
        num_rows, num_cols-1, figsize=(num_rows*2, (num_cols-1)*2), constrained_layout=True
    )

    def get_axes(r: int, c: int) -> Any:
        if c > r:
            return axes[r, c-1]
        else:
            return axes[r, c]

    colors = list(map(
        lambda v: 'blue' if v == 1 else 'red',
        df[config.LABEL]
    ))

    for r in range(num_rows):
        for c in range(num_cols):
            if r != c:
                ax = get_axes(r, c)

                ax.scatter(
                    df[config.FEATURES[r]],
                    df[config.FEATURES[c]],
                    c=colors
                )

                ax.set_xlabel(config.FEATURES[r])
                ax.set_ylabel(config.FEATURES[c])


def describe_processed_data():
    df = utils.read_csv('winequality-red-with-label-standardized')

    utils.ensure_col_exists(df, config.LABEL)

    print("Descriptive statistics for processed data:", '\n')
    utils.print_descriptive_statistics(df)

    utils.make_plot(
        title='Processed Wine Label Distribution',
        xlabel=config.LABEL,
        ylabel='Frequency',
        file_name='wine_label_distribution',
        plot=lambda: plot_class_distribution(df)
    )

    utils.make_plot(
        title='Processed Wine Label Correlation',
        file_name='wine_label_correlation',
        plot=lambda: plot_feature_correlation(df)
    )
