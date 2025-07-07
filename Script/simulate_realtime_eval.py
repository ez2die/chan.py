#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate real-time evaluation of ML ensemble on historical dataset.

Usage:
  python Script/simulate_realtime_eval.py \
      --dataset btc_swap_1h_swaponly \
      --horizon 1 \
      --limit 5000

The script feeds bars one-by-one to ChanRealTimePredictor, mimicking
online inference. When the future bar(s) arrive, it settles the previous
prediction and updates rolling metrics (AUC, accuracy).
"""

import argparse
import json
import sys
from pathlib import Path
from collections import deque
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ModelStrategy.DatasetManager import ChanDatasetManager
from ModelStrategy.RealTimePredictor import ChanRealTimePredictor

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def load_bars(dataset_name: str) -> pd.DataFrame:
    dm = ChanDatasetManager("data/ml_datasets")
    df, _ = dm.load_training_dataset(dataset_name)

    # Ensure a 'timestamp' column exists
    if "timestamp" not in df.columns:
        if df.index.name is not None:
            df = df.reset_index().rename(columns={df.index.name: "timestamp"})
        else:
            df = df.reset_index().rename(columns={"index": "timestamp"})

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def parse_args():
    p = argparse.ArgumentParser("Dry-run online evaluation")
    p.add_argument("--dataset", default="btc_swap_1h_swaponly", help="dataset prefix or file name")
    p.add_argument("--horizon", type=int, default=1, help="look-ahead bars to compute label")
    p.add_argument("--limit", type=int, default=None, help="max bars to simulate (debug)")
    p.add_argument("--model-config", type=str, default=None, help="JSON file mapping model names to paths")
    p.add_argument("--history-window", type=int, default=300, help="Bars kept in rolling buffer for history-dependent features")
    p.add_argument("--feature-window", type=int, default=120, help="Bars slice passed to feature calculator each step")
    p.add_argument("--print-every", type=int, default=500, help="Print rolling AUC every N settled predictions")
    return p.parse_args()


def main():
    args = parse_args()

    df = load_bars(args.dataset)
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"Loaded {len(df)} bars from {args.dataset}")

    # Predictor setup
    cfg = {
        "signal_thresholds": {"buy": 0.7, "sell": 0.3},
        "history_window": args.history_window,
        "feature_window": args.feature_window,
    }
    predictor = ChanRealTimePredictor(cfg)

    # Load models if mapping provided, else rely on RealTimePredictor internal default paths
    if args.model_config:
        model_paths = json.loads(Path(args.model_config).read_text())
        predictor.load_models(model_paths)
        predictor.setup_ensemble({"method": "weighted_average"})

    cache: Dict[pd.Timestamp, float] = {}
    probs, labels = [], []

    horizon = max(1, args.horizon)

    bar_iter = tqdm(range(len(df)), desc="simulate")
    print_every = max(1, args.print_every)
    for i in bar_iter:
        cur_bar = df.iloc[i]
        cur_ts = cur_bar["timestamp"]

        # Settle bar predicted horizon bars ago
        settle_idx = i - horizon
        if settle_idx >= 0:
            prev_bar = df.iloc[settle_idx]
            prev_ts = prev_bar["timestamp"]
            p = cache.pop(prev_ts, None)
            if p is not None:
                future_close = cur_bar["close"]
                label = int(future_close > prev_bar["close"])
                probs.append(p)
                labels.append(label)

                # periodic logging
                if len(probs) % print_every == 0 and len(set(labels)) > 1:
                    try:
                        auc_tmp = roc_auc_score(labels, probs)
                        bar_iter.write(f"[online] settled={len(probs)}  AUC={auc_tmp:.4f}")
                    except Exception:
                        pass

        # Prepare current bar dict
        bar_dict = {
            "timestamp": str(cur_ts),
            "open": float(cur_bar["open"]),
            "high": float(cur_bar["high"]),
            "low": float(cur_bar["low"]),
            "close": float(cur_bar["close"]),
            "volume": float(cur_bar.get("volume", 0)),
        }
        try:
            pred_res = predictor.predict_single(bar_dict)
            cache[cur_ts] = pred_res["prediction"]
        except Exception as exc:
            bar_iter.write(f"Predict error at {cur_ts}: {exc}")

    if probs:
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, np.asarray(probs) > 0.5)
        print(f"Simulated online AUC={auc:.4f}, Accuracy={acc:.4f} (n={len(probs)})")
    else:
        print("No settled predictions – check dataset/horizon")


if __name__ == "__main__":
    main() 