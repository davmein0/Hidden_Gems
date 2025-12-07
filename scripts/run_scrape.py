#!/usr/bin/env python
"""CLI wrapper to run scrape pipeline."""
import argparse
from hidden_gems.scrape.pipeline import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    date = args.date or None
    run_pipeline(date or "2025-11-28", limit_tickers=args.limit)


if __name__ == "__main__":
    main()
