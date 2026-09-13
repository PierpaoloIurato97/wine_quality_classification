import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils


def plot_feature_vs_quality(df: pd.DataFrame):
    num_features = len(config.FEATURES)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    _, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 4, num_rows * 3),
        constrained_layout=True,
    )

    jitter = np.random.default_rng(config.RANDOM_STATE).uniform(-0.2, 0.2, size=len(df))

    for i, feature in enumerate(config.FEATURES):
        r, c = divmod(i, num_cols)
        ax = axes[r, c]

        ax.scatter(
            df[feature], df["quality"] + jitter, alpha=0.3, s=10, color="steelblue"
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("quality")

    for i in range(num_features, num_rows * num_cols):
        r, c = divmod(i, num_cols)
        axes[r, c].set_visible(False)


def plot_correlation(df: pd.DataFrame):
    cols = config.FEATURES + ["quality"]
    corr = pd.DataFrame(df[cols]).corr()

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
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

    fig.colorbar(im, ax=ax)


def plot_feature_distribution(df: pd.DataFrame):
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

        ax.hist(df[feature], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")

    for i in range(num_features, num_rows * num_cols):
        r, c = divmod(i, num_cols)
        axes[r, c].set_visible(False)


def describe_raw_data():
    df = utils.read_csv("winequality-red")

    utils.ensure_col_exists(df, "quality")

    print("Descriptive statistics for raw data:", "\n")
    utils.print_descriptive_statistics(df)

    utils.make_plot(
        title="Quality Distribution",
        xlabel="Quality",
        ylabel="Frequency",
        file_name="quality_distribution",
        plot=lambda: plt.hist(df["quality"], bins=5, edgecolor="black"),
    )

    utils.make_plot(
        title="Feature vs Quality",
        file_name="feature_vs_quality",
        plot=lambda: plot_feature_vs_quality(df),
    )

    utils.make_plot(
        title="Raw Data Correlation",
        file_name="raw_data_correlation",
        plot=lambda: plot_correlation(df),
    )

    utils.make_plot(
        title="Feature Distributions",
        file_name="feature_distributions",
        plot=lambda: plot_feature_distribution(df),
    )
