from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils


def plot_class_distribution(df: pd.DataFrame):
    plt.hist(df[config.LABEL], bins=2, edgecolor="black")


def plot_split_class_distribution(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    datasets = {"Train": train_df, "Validation": val_df, "Test": test_df}
    labels = list(datasets.keys())
    count_0 = [(df[config.LABEL] == 0).sum() for df in datasets.values()]
    count_1 = [(df[config.LABEL] == 1).sum() for df in datasets.values()]

    x = range(len(labels))
    width = 0.35

    bars_0 = plt.bar(
        [i - width / 2 for i in x],
        count_0,
        width,
        label="0",
        color="red",
        alpha=0.6,
        edgecolor="black",
    )
    bars_1 = plt.bar(
        [i + width / 2 for i in x],
        count_1,
        width,
        label="1",
        color="blue",
        alpha=0.6,
        edgecolor="black",
    )

    for bar in bars_0:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars_1:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(list(x), labels)
    plt.legend()


def plot_correlation(df: pd.DataFrame):
    cols = config.FEATURES + [config.LABEL]
    subset: pd.DataFrame = cast(pd.DataFrame, df[cols])
    corr = subset.corr()

    _, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6
            )

    plt.colorbar(im, ax=ax)


def plot_feature_vs_label(df: pd.DataFrame):
    num_features = len(config.FEATURES)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    _, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 4, num_rows * 3),
        constrained_layout=True,
    )

    jitter = np.random.default_rng(config.RANDOM_STATE).uniform(
        -0.15, 0.15, size=len(df)
    )

    colors = list(map(lambda v: "blue" if v == 1 else "red", df[config.LABEL]))

    for i, feature in enumerate(config.FEATURES):
        r, c = divmod(i, num_cols)
        ax = axes[r, c]

        ax.scatter(df[feature], df[config.LABEL] + jitter, alpha=0.3, s=10, c=colors)

        ax.set_xlabel(feature)
        ax.set_ylabel(config.LABEL)
        ax.set_yticks([0, 1])

    for i in range(num_features, num_rows * num_cols):
        r, c = divmod(i, num_cols)
        axes[r, c].set_visible(False)


def describe_processed_data():
    train_df = utils.read_csv("winequality-red-train")
    val_df = utils.read_csv("winequality-red-validation")
    test_df = utils.read_csv("winequality-red-test")

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    utils.ensure_col_exists(df, config.LABEL)

    print("Descriptive statistics for processed data:", "\n")
    utils.print_descriptive_statistics(df)

    utils.make_plot(
        title="Label Distribution",
        xlabel="Label",
        ylabel="Frequency",
        file_name="label_distribution",
        plot=lambda: plot_class_distribution(df),
    )

    utils.make_plot(
        title="Label Distribution per Split",
        xlabel="Dataset",
        ylabel="Count",
        file_name="label_split_distribution",
        plot=lambda: plot_split_class_distribution(train_df, val_df, test_df),
    )

    utils.make_plot(
        title="Processed Data Correlation",
        file_name="processed_data_correlation",
        plot=lambda: plot_correlation(df),
    )

    utils.make_plot(
        title="Feature vs Label",
        file_name="feature_vs_label",
        plot=lambda: plot_feature_vs_label(df),
    )
