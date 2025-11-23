import config
import utils


def add_label_var() -> None:
    df = utils.read_csv('winequality-red')

    utils.ensure_col_exists(df, 'quality')

    df[config.LABEL] = (df['quality'] > 5).astype(int)
    del df['quality']

    utils.save_csv(df, 'winequality-red-with-label')
