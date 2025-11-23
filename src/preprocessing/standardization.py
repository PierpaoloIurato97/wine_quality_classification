import config
import utils


def standardize():
    df = utils.read_csv('winequality-red-with-label')

    for col in config.FEATURES:
        utils.ensure_col_exists(df, col)

    df[config.FEATURES] = (
        df[config.FEATURES] -
        df[config.FEATURES].mean()
    ) / df[config.FEATURES].std()

    utils.save_csv(df, 'winequality-red-with-label-standardized')
