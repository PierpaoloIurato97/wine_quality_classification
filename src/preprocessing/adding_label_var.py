import config
import utils


def add_label_var() -> None:
    """
    Second preprocessing step in the pipeline.

    Converts the raw numeric quality score (1–10) into a binary classification
    label: 1 if quality > QUALITY_THRESHOLD (i.e. "good wine"), 0 otherwise.
    The original quality column is then dropped so that only the binary label
    used by the classifier is retained.
    """
    df = utils.read_csv("winequality-red-filtered")

    utils.ensure_col_exists(df, "quality")

    df[config.LABEL] = (df["quality"] > config.QUALITY_THRESHOLD).astype(int)
    del df["quality"]

    utils.save_csv(df, "winequality-red-with-label")
