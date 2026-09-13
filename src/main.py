import argparse

from data_analysis import describe_processed_data, describe_raw_data
from evaluation import evaluate
from preprocessing import add_label_var, remove_outliers, split, standardize
from training import train


def parse_args():
    """
    Parses the --action CLI argument that selects which pipeline stage to run.

    Keeping argument parsing in its own function makes main() readable and
    makes it easy to add new flags in the future without restructuring the
    entry-point logic.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=[
            "describe_raw_data",
            "remove_outliers",
            "add_label_var",
            "split",
            "standardize",
            "describe_processed_data",
            "train",
            "evaluate",
        ],
        required=True,
        help="Action to perform",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.action == "describe_raw_data":
        describe_raw_data()
    elif args.action == "describe_processed_data":
        describe_processed_data()
    elif args.action == "remove_outliers":
        remove_outliers()
    elif args.action == "add_label_var":
        add_label_var()
    elif args.action == "standardize":
        standardize()
    elif args.action == "split":
        split()
    elif args.action == "train":
        train()
    elif args.action == "evaluate":
        evaluate()
