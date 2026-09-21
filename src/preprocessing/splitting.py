import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
import utils


def cluster_samples(df: pd.DataFrame) -> np.ndarray:
    """
    Assigns a cluster ID to every sample using KMeans on standardised features.

    Standardisation is temporary and internal: it is needed because KMeans
    relies on Euclidean distance, and the raw features live on very different
    scales (e.g. total sulfur dioxide ∈ [1, 200] vs density ∈ [0.985, 1.010]).
    The resulting cluster labels are used only for stratification; the output
    CSVs still contain the original, non-standardised values.
    """
    X = df[config.FEATURES].to_numpy(dtype=np.float64)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(
        n_clusters=config.N_CLUSTERS,
        random_state=config.RANDOM_STATE,
        n_init=10,
    )
    return km.fit_predict(X_scaled)


def build_stratify_key(df: pd.DataFrame, cluster_ids: np.ndarray) -> np.ndarray:
    """
    Builds a composite stratification key combining the class label and the
    spatial cluster each sample belongs to.

    Using both dimensions ensures that train/test splits preserve not only
    the global class balance but also the spatial structure of the data: every
    dense region (cluster) of the feature space is represented in every split.
    """
    return (
        df[config.LABEL].astype(str)
        + "_"
        + pd.Series(cluster_ids, index=df.index).astype(str)
    ).to_numpy()


def split():
    """
    Third preprocessing step in the pipeline.

    Splits the labelled dataset into train and test sets according to the
    ratios in config. The split is stratified on the composite key
    (label + cluster_id) so that every spatial cluster of the feature space is
    represented in both splits, preserving both the class balance and the
    local data patterns.
    """
    df = utils.read_csv("winequality-red-with-label")

    cluster_ids = cluster_samples(df)
    stratify_key = build_stratify_key(df, cluster_ids)

    train_df, test_df, train_key, test_key = train_test_split(
        df,
        stratify_key,
        test_size=config.TEST_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=stratify_key,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --- Diagnostics ---
    print(f"Clusters: {config.N_CLUSTERS}")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    for name, split_key in [("Train", train_key), ("Test", test_key)]:
        unique, counts = np.unique(split_key, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  {name}: {dist}")

    utils.save_csv(train_df, "winequality-red-train")
    utils.save_csv(test_df, "winequality-red-test")
