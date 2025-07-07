#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run full training pipeline with EnhancedChanFeatureCalculator_v2.

Usage
-----
$ python Script/train_with_v2_stack.py
The script mirrors the reference snippet provided by the user but switches
ChanMLTrainer to use the v2 feature calculator and passes selected-feature
whitelist automatically.
"""
import os
import sys
import datetime
import pprint
from pathlib import Path

# Ensure project root is on PYTHONPATH so that `ModelStrategy` can be imported
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from ModelStrategy.ChanMLTrainer import ChanMLTrainer
from ModelStrategy.EnhancedFeatureCalculator_v2 import (
    EnhancedChanFeatureCalculator_v2,
)

# ----------------- single-thread to avoid OpenMP issues -----------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ----------------- LightGBM best params (given by user) -----------------
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

# ------------------------- output dirs ----------------------------------
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"models/btc_swap_stack_{ts}"
log_dir = f"logs/btc_swap_stack_{ts}"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ------------------------- trainer config -------------------------------
cfg = {
    "data_dir": "data/ml_datasets",
    "model_dir": model_dir,
    "log_dir": log_dir,
    "dataset_name": "btc_swap_1h",
    "target_column": "binary_direction",
    "models": ["lgb", "xgb", "cat"],
    "random_state": 42,
    # whitelist file produced by Phase-2.2.1 – one feature per line
    "selected_features_path": "enhanced_features_list.txt",
    "ensemble_method": "stacking",
    "model_params": {"lgb": lgb_params},
}

pprint.pprint({"config": cfg})
print("\n开始训练 (stacking + v2 features)…\n")

# Instantiate trainer
trainer = ChanMLTrainer(cfg)
# Swap in v2 feature calculator
trainer.feature_calculator = EnhancedChanFeatureCalculator_v2()
# If a whitelist exists, propagate into v2 config to enable trimming
if trainer.selected_features:
    trainer.feature_calculator.feature_config["selected_features_only"] = True
    trainer.feature_calculator.feature_config["selected_features"] = trainer.selected_features

# ------------------------- run pipeline ---------------------------------
results = trainer.run_full_training_pipeline()

# ------------------------- print summary --------------------------------
print("\n=== Validation AUC ===")
for name, info in results.items():
    if isinstance(info, dict) and "val_metrics" in info:
        print(f"{name.upper():8}: {info['val_metrics']['auc']:.4f}")

print(f"\n产物目录: {model_dir}") 