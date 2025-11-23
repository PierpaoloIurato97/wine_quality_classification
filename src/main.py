import argparse

from data_analysis import (
    describe_and_plot_processed_data,
    describe_and_plot_raw_data
)
from evaluation import evaluate
from preprocessing import add_label_var, split, standardize
from training import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=[
            "describe_and_plot_raw_data",
            "add_label_var",
            "standardize",
            "describe_and_plot_processed_data",
            "split",
            "train",
            "evaluation"
        ],
        required=True,
        help="Azione da eseguire"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.action == "describe_and_plot_raw_data":
        describe_and_plot_raw_data()
    elif args.action == "describe_and_plot_processed_data":
        describe_and_plot_processed_data()
    elif args.action == "add_label_var":
        add_label_var()
    elif args.action == "split":
        split()
    elif args.action == "standardize":
        standardize()
    elif args.action == "train":
        train()
    elif args.action == "evaluation":
        evaluate()
