#!/usr/bin/env python
"""CLI wrapper to run training for Hidden Gems ML models."""
from hidden_gems.ml.train import main as train_main
from pathlib import Path
import argparse
import os
from scripts.generate_labels import main as generate_labels


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Path to the CSV dataset to use for training")
    parser.add_argument("--labelgen", action="store_true", help="Generate labels from merged_dataset.csv using simple heuristic before training")
    parser.add_argument("--model-path", default=None, help="Path to a pretrained model file to use for scoring; if present training will be skipped")
    args = parser.parse_args(argv)

    if args.labelgen:
        # Generate labels from repo root merged_dataset.csv
        repo_root = Path(__file__).resolve().parents[1]
        merged = repo_root / "merged_dataset.csv"
        out = repo_root / "data" / "labeled_from_merged.csv"
        print("Generating labels from:", merged)
        generate_labels(str(merged), str(out))
        dataset = out
    elif args.dataset:
        dataset = Path(args.dataset)
    else:
        # show default behavior: use data/example.csv
        repo_root = Path(__file__).resolve().parents[1]
        dataset = repo_root / "data" / "example.csv"

    os.environ["DATASET_PATH"] = str(dataset)
    train_main(str(dataset), model_path_arg=args.model_path)


if __name__ == "__main__":
    main()
