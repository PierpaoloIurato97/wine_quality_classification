from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils


def plot_class_distribution(df: pd.DataFrame):
    """
    Plots a histogram of the binary label across the full processed dataset.

    Gives an immediate visual check of class imbalance, which directly
    affects model bias and the interpretation of accuracy as a metric.
    """
    plt.hist(df[config.LABEL], bins=2, edgecolor="black")


def plot_split_class_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Produces a grouped bar chart comparing the class counts in each split.

    Verifies that the stratified split preserved the class ratio across train
    and test sets, which is a prerequisite for unbiased evaluation.
    """
    datasets = {"Train": train_df, "Test": test_df}
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
    """
    Renders a colour-coded correlation heatmap for all features and the binary
    label in the processed dataset.

    Complements the raw-data correlation plot to check whether the binarisation
    of the quality score changes which features are most predictive.
    """
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
    """
    Plots each feature against the binary label with vertical jitter.

    Allows a visual sanity check that the binarisation decision boundary
    separates the two classes in a way that is consistent with each feature's
    distribution.
    """
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


def plot_split_feature_distributions(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Overlays normalised histograms of each feature for train and test sets.

    If the KMeans-stratified split preserved the spatial structure of the data,
    the two distributions should be nearly identical for every feature. Visible
    discrepancies would indicate that certain patterns ended up in only one of
    the two splits.
    """
    num_features = len(config.FEATURES)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    _, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 4, num_rows * 3),
        constrained_layout=True,
    )

    for i, feature in enumerate(config.FEATURES):
        r, c = divmod(i, num_cols)
        ax = axes[r, c]

        ax.hist(
            train_df[feature],
            bins=30,
            density=True,
            alpha=0.5,
            color="blue",
            edgecolor="black",
            linewidth=0.5,
            label="Train",
        )
        ax.hist(
            test_df[feature],
            bins=30,
            density=True,
            alpha=0.5,
            color="red",
            edgecolor="black",
            linewidth=0.5,
            label="Test",
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for i in range(num_features, num_rows * num_cols):
        r, c = divmod(i, num_cols)
        axes[r, c].set_visible(False)


def describe_processed_data():
    """
    Entry point for the processed-data analysis stage of the pipeline.

    Loads both splits, concatenates them for aggregate statistics, and
    generates plots that summarise the final preprocessed dataset. Running
    this after the full preprocessing sequence confirms that the data is
    clean and balanced before training begins.
    """
    train_df = utils.read_csv("winequality-red-train")
    test_df = utils.read_csv("winequality-red-test")

    df = pd.concat([train_df, test_df], ignore_index=True)

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
        plot=lambda: plot_split_class_distribution(train_df, test_df),
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

    utils.make_plot(
        title="Feature Distributions: Train vs Test",
        file_name="feature_distributions_train_vs_test",
        plot=lambda: plot_split_feature_distributions(train_df, test_df),
    )
