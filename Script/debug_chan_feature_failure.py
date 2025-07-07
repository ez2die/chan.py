#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick diagnostic: measure how often Chan-structure feature extraction fails
and collect top exception reasons.

The script slides a fixed-size window through the training dataset, invokes
`EnhancedChanFeatureCalculator._add_chan_features()` on each window, and checks
whether structural features (e.g. `bi_count`) are successfully produced.
Success criterion: `bi_count` in the last row > 0.

Outputs:
    • total windows evaluated
    • success / failure counts & ratio
    • top 10 failure reasons with counts
Usage:
    $ python Script/debug_chan_feature_failure.py [--window 200] [--step 200]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict

import pandas as pd

# Ensure project root in path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ModelStrategy.EnhancedFeatureCalculator import (
    EnhancedChanFeatureCalculator,
)
from tools.ml_dataset_example import MLDatasetLoader

DEFAULT_DATASET_FILE = "btc_swap_1h_train.json"


def load_dataset() -> pd.DataFrame:
    """Load OHLCV DataFrame using existing dataset loader"""
    loader = MLDatasetLoader("data/ml_datasets")
    kline_data, _ = loader.load_dataset(DATASET_FILE)
    df = loader.kline_data_to_dataframe(kline_data)
    df.reset_index(inplace=True)
    df.rename(columns={"datetime": "timestamp"}, inplace=True)
    return df


def analyse(window: int, step: int) -> None:
    df = load_dataset()
    calc = EnhancedChanFeatureCalculator()

    total = 0
    success = 0
    failure_reasons: Dict[str, int] = Counter()

    for start in range(0, len(df) - window + 1, step):
        slice_df = df.iloc[start : start + window].copy()
        # Only keep price columns expected by calculator to speed up
        needed_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        slice_df = slice_df[needed_cols]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _ = calc._add_chan_features(slice_df)
        log = buf.getvalue()

        total += 1
        if "缠论特征计算失败" in log:
            # parse reason text
            reason = log.split("缠论特征计算失败:")[-1].split("\n")[0].strip()
            failure_reasons[reason] += 1
        else:
            success += 1

    print("=== Chan feature diagnostic ===")
    print(f"window size      : {window}")
    print(f"step             : {step}")
    print(f"total windows    : {total}")
    print(f"successes        : {success}  ({success/total:.2%})")
    print(f"failures         : {total-success}  ({(total-success)/total:.2%})")
    print("\nTop failure reasons (count):")
    for reason, cnt in failure_reasons.most_common(10):
        print(f"  {cnt:4d} × {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=200, help="K-line rows per evaluation window")
    parser.add_argument("--step", type=int, default=200, help="Stride between windows")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_FILE, help="Dataset JSON filename in data/ml_datasets/")
    args = parser.parse_args()

    global DATASET_FILE
    DATASET_FILE = args.dataset

    analyse(window=args.window, step=args.step) 