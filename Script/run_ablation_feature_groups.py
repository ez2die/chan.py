#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run multiple feature-group ablation experiments sequentially.
Each experiment trains LGB/XGB/CAT + stacking on the swap-only dataset with
specific feature groups toggled, then logs validation AUCs.

Usage
-----
$ python Script/run_ablation_feature_groups.py
"""

import os
import sys
import datetime
import pprint
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from ModelStrategy.ChanMLTrainer import ChanMLTrainer

# common LightGBM params (same as previous best)
lgb_params = {
    "num_leaves": 107,
    "max_depth": 2,
    "learning_rate": 0.07,
    "feature_fraction": 0.793221170827609,
    "bagging_fraction": 0.6235484260359628,
    "bagging_freq": 1,
    "min_child_samples": 323,
    "min_child_weight": 0.0016808683292686042,
    "reg_alpha": 4.878392120871905,
    "reg_lambda": 0.723578648726878,
    "num_iterations": 1600,
    "num_threads": 1,
    "force_row_wise": True,
}

experiment_defs = {
    "baseline_all": {
        # keep v2 defaults (all True)
        "feature_config": None,
    },
    "no_momentum_vol_mt_pattern": {
        "feature_config": {
            "basic": True,
            "technical": True,
            "volume": True,
            "chan_theory": True,
            "multi_timeframe": False,
            "volatility": False,
            "momentum": False,
            "pattern": False,
        },
    },
    "basic_plus_chan": {
        "feature_config": {
            "basic": True,
            "technical": False,
            "volume": False,
            "chan_theory": True,
            "multi_timeframe": False,
            "volatility": False,
            "momentum": True,
            "pattern": False,
        },
    },
    "basic_only": {
        "feature_config": {
            "basic": True,
            "technical": False,
            "volume": False,
            "chan_theory": False,
            "multi_timeframe": False,
            "volatility": False,
            "momentum": True,
            "pattern": False,
        },
    },
}

results_summary = {}

ts_base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def run_one(tag: str, feature_cfg):
    model_dir = f"models/ablation_{tag}_{ts_base}"
    log_dir   = f"logs/ablation_{tag}_{ts_base}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cfg = {
        "data_dir": "data/ml_datasets",
        "dataset_name": "btc_swap_1h_swaponly",
        "model_dir": model_dir,
        "log_dir": log_dir,
        "target_column": "binary_direction",
        "models": ["lgb", "xgb", "cat"],
        "ensemble_method": "stacking",
        "random_state": 42,
        "feature_calc_version": "v2",
        "feature_config": feature_cfg if feature_cfg else {},
        "model_params": {"lgb": lgb_params},
    }

    print("\n========== Experiment:", tag, "==========")
    pprint.pprint({"feature_config": feature_cfg})

    trainer = ChanMLTrainer(cfg)
    if feature_cfg:  # ensure runtime instance updated
        trainer.feature_calculator.feature_config.update(feature_cfg)

    res = trainer.run_full_training_pipeline()

    aucs = {m.upper(): info["val_metrics"]["auc"] for m, info in res.items() if isinstance(info, dict)}
    auc_ens = res.get("ensemble", {}).get("val_metrics", {}).get("auc", None)
    if auc_ens:
        aucs["ENSEMBLE"] = auc_ens

    results_summary[tag] = aucs
    print("AUC:", aucs)


for name, cfgdict in experiment_defs.items():
    run_one(name, cfgdict["feature_config"])

print("\n===== Summary =====")
for exp, aucs in results_summary.items():
    print(exp, aucs) 