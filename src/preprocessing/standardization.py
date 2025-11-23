import config
import utils


def standardize():
    df = utils.read_csv('winequality-red-with-label')

    for col in config.COLUMNS_TO_STANDARDIZE:
        utils.ensure_col_exists(df, col)

    df[config.COLUMNS_TO_STANDARDIZE] = (
        df[config.COLUMNS_TO_STANDARDIZE] -
        df[config.COLUMNS_TO_STANDARDIZE].mean()
    ) / df[config.COLUMNS_TO_STANDARDIZE].std()

    utils.save_csv(df, 'winequality-red-with-label-standardized')
