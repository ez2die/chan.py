#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate model performance with different Top-N feature whitelists (200/100/50).
Assumes `enhanced_features_list.txt` contains ranked features (one per line).
Produces validation AUC for each N.
"""
import os, sys, datetime, pprint, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from ModelStrategy.ChanMLTrainer import ChanMLTrainer

# load master list
master_path = ROOT / "enhanced_features_list.txt"
if not master_path.exists():
    print("Feature list file not found:", master_path)
    sys.exit(1)
features = [ln.strip() for ln in master_path.read_text().splitlines() if ln.strip()]

sizes = [200, 100, 50]
results = {}

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

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

def run_with_n(top_n:int):
    sub_list = features[:top_n]
    tmp_list_file = ROOT / f"tmp_top{top_n}_{ts}.txt"
    tmp_list_file.write_text("\n".join(sub_list))

    model_dir = f"models/feat_top{top_n}_{ts}"
    log_dir   = f"logs/feat_top{top_n}_{ts}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cfg = {
        "data_dir":"data/ml_datasets",
        "dataset_name":"btc_swap_1h_swaponly",
        "model_dir":model_dir,
        "log_dir":log_dir,
        "target_column":"binary_direction",
        "models":["lgb","xgb","cat"],
        "ensemble_method":"stacking",
        "random_state":42,
        "feature_calc_version":"v2",
        "selected_features_path": str(tmp_list_file),
        "model_params": {"lgb": lgb_params},
    }

    print(f"\n=== Top-{top_n} experiment ===")
    trainer = ChanMLTrainer(cfg)
    res = trainer.run_full_training_pipeline()
    auc = {m.upper():info["val_metrics"]["auc"] for m,info in res.items() if isinstance(info,dict)}
    ens = res.get("ensemble",{}).get("val_metrics",{}).get("auc")
    if ens: auc["ENSEMBLE"]=ens
    results[top_n]=auc
    print("AUC", auc)
    # cleanup temp file
    tmp_list_file.unlink(missing_ok=True)

for n in sizes:
    run_with_n(n)

print("\n===== Summary =====")
for n, auc in results.items():
    print(f"Top-{n}", auc) 