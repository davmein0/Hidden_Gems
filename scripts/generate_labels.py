#!/usr/bin/env python
"""Create a labeled dataset from an existing `merged_dataset.csv` using a quick heuristic.

This is intended for prototypes only and avoids rescraping or waiting for future prices.

Heuristic (default):
- Label 1 (undervalued) if:
  - PE_Ratio is not null and < 15
  - PB_Ratio is not null and < 3
  - FreeCashFlow > 0
- Otherwise label 0.

Output: writes `data/labeled_from_merged.csv` (relative to repo root by default)
"""
import argparse
import os
from pathlib import Path
import pandas as pd


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate labels using a simple heuristic. Returns df with 'Label' column."""
    # Make a copy so we don't mutate original
    out = df.copy()

    def is_undervalued(row):
        try:
            pe = float(row.get("PE_Ratio")) if row.get("PE_Ratio") not in (None, "", "inf") else None
        except Exception:
            pe = None
        try:
            pb = float(row.get("PB_Ratio")) if row.get("PB_Ratio") not in (None, "", "inf") else None
        except Exception:
            pb = None
        try:
            fcf = float(row.get("FreeCashFlow")) if row.get("FreeCashFlow") not in (None, "") else None
        except Exception:
            fcf = None

        if pe is not None and pb is not None and fcf is not None:
            return int(pe < 15 and pb < 3 and fcf > 0)
        # fallback conservative label
        return 0

    out["Label"] = out.apply(is_undervalued, axis=1)
    return out


def main(in_path: str | None = None, out_path: str | None = None):
    repo_root = Path(__file__).resolve().parents[1]
    if in_path is None:
        in_path = repo_root / "merged_dataset.csv"
    else:
        in_path = Path(in_path)

    if out_path is None:
        out_path = repo_root / "data" / "labeled_from_merged.csv"
    else:
        out_path = Path(out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input merged file not found: {in_path}")

    df = pd.read_csv(in_path)
    labeled = generate_labels(df)

    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_path, index=False)
    print(f"Wrote labeled dataset to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to merged_dataset.csv (default: repo root merged_dataset.csv)")
    parser.add_argument("--output", default=None, help="Path to write labeled CSV (default: data/labeled_from_merged.csv)")
    args = parser.parse_args()
    main(args.input, args.output)
