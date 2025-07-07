#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training pipeline using:
    • Clean swap-only dataset (btc_swap_1h_swaponly_chan.json)
    • EnhancedChanFeatureCalculator_v2
    • Stacking ensemble (LGB + XGB + CAT)
Run:
    python Script/train_v2_swaponly_stack.py
"""
import os
import sys
import datetime
import pprint
from pathlib import Path

# Project root to PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------- single-thread to avoid OpenMP issues -------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ------- LightGBM tuned parameters (given) -------
lgb_params = {
    'num_leaves': 107,
    'max_depth': 2,
    'learning_rate': 0.07,
    'feature_fraction': 0.793221170827609,
    'bagging_fraction': 0.6235484260359628,
    'bagging_freq': 1,
    'min_child_samples': 323,
    'min_child_weight': 0.0016808683292686042,
    'reg_alpha': 4.878392120871905,
    'reg_lambda': 0.723578648726878,
    'num_iterations': 1600,
    'num_threads': 1,
    'force_row_wise': True,
}

# ------- feature group configuration (disable potential redundancy) -------
feature_cfg = {
    "basic": True,
    "technical": True,
    "volume": True,
    "chan_theory": True,
    "multi_timeframe": False,  # disabled
    "volatility": False,       # disabled (was empty on sample)
    "momentum": True,
    "pattern": False,          # disabled
}

# ------- output directories -------
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = f"models/btc_swap_stack_swaponly_{ts}"
log_dir = f"logs/btc_swap_stack_swaponly_{ts}"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ------- Trainer configuration -------
cfg = {
    'data_dir': 'data/ml_datasets',
    'model_dir': model_dir,
    'log_dir':   log_dir,
    # use dataset prefix so *_train.json / *_val.json are auto-loaded
    'dataset_name': 'btc_swap_1h_swaponly',
    'target_column': 'binary_direction',
    'models': ['lgb', 'xgb', 'cat'],
    'random_state': 42,
    'selected_features_path': 'enhanced_features_list.txt',  # Top-200 list (optional)
    'ensemble_method': 'stacking',
    'feature_calc_version': 'v2',  # instruct ChanMLTrainer to use v2 calculator
    'feature_config': feature_cfg,
    'model_params': {
        'lgb': lgb_params
    }
}

pprint.pprint({'config': cfg})
print("\n开始训练 (swap-only + v2 features, stacking)…\n")

from ModelStrategy.ChanMLTrainer import ChanMLTrainer

trainer = ChanMLTrainer(cfg)  # ChanMLTrainer will instantiate v2 based on config

# ensure feature config updated (in case Trainer created new instance)
if hasattr(trainer.feature_calculator, 'feature_config'):
    trainer.feature_calculator.feature_config.update(feature_cfg)

results = trainer.run_full_training_pipeline()

print("\n=== Validation AUC ===")
for name, info in results.items():
    if isinstance(info, dict) and 'val_metrics' in info:
        print(f"{name.upper():8}: {info['val_metrics']['auc']:.4f}")

print(f"\n产物目录: {model_dir}") 