#!/usr/bin/env python3
"""Train 5-fold out-of-fold stacking ensemble on Top-100 feature set.

* Trains LightGBM, XGBoost, CatBoost with best hyper-parameters on each fold.
* Collects out-of-fold (OOF) predictions.
* Fits LogisticRegression meta-learner on OOF preds.
* Evaluates ensemble on held-out validation set.
Usage:
    python Script/train_oof_stacking.py --best_params models/hpo_cat_top100_20250705.json \
        --features mixed_features_top100_20250704_164132.txt
"""
from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ModelStrategy.ChanMLTrainer import ChanMLTrainer
from ModelStrategy.models.LightGBMModelGenerator import CLightGBMModelGenerator
from ModelStrategy.models.XGBModelGenerator import CXGBModelGenerator
from ModelStrategy.models.CatBoostModelGenerator import CCatBoostModelGenerator
from ModelStrategy.EnsemblePredictor import ChanEnsemblePredictor


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OOF stacking trainer")
    p.add_argument("--best_params", type=str, required=True, help="JSON file with best params")
    p.add_argument("--features", type=str, required=True, help="Feature whitelist path")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--dataset", type=str, default="btc_swap_1h_swaponly")
    return p.parse_args()


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    best = json.load(open(args.best_params, "r", encoding="utf-8"))

    trainer_cfg = {
        "data_dir": "data/ml_datasets",
        "model_dir": "models/oof_stack",
        "log_dir": "logs/oof_stack",
        "dataset_name": args.dataset,
        "target_column": "binary_direction",
        "models": [],  # manual training inside script
        "selected_features_path": args.features,
        "random_state": 42,
    }

    trainer = ChanMLTrainer(trainer_cfg)
    train_df, val_df = trainer.prepare_training_data()

    y = train_df[trainer_cfg["target_column"]].values.astype(np.float32)
    feature_cols = trainer.feature_calculator.get_feature_names(train_df)
    X = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[trainer_cfg["target_column"]].values.astype(np.float32)

    oof_preds = {"lgb": np.zeros_like(y, dtype=float), "xgb": np.zeros_like(y, dtype=float), "cat": np.zeros_like(y, dtype=float)}

    tss = TimeSeriesSplit(n_splits=args.folds)
    for fold, (train_idx, valid_idx) in enumerate(tss.split(X, y), 1):
        print(f"Fold {fold}/{args.folds} …")
        X_tr, X_va = X[train_idx], X[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]

        # LightGBM
        lgb_gen = CLightGBMModelGenerator(model_tag=f"lgb_fold{fold}", lgb_params=best.get("lightgbm", {}))
        lgb_gen.feature_names = feature_cols
        lgb_gen.train(lgb_gen.create_data_set(X_tr, y_tr), lgb_gen.create_data_set(X_va, y_va))
        oof_preds["lgb"][valid_idx] = lgb_gen.predict(lgb_gen.create_data_set(X_va))

        # XGBoost
        xgb_gen = CXGBModelGenerator(model_tag=f"xgb_fold{fold}", xgb_params=best.get("xgboost", {}))
        xgb_gen.feature_names = feature_cols
        xgb_gen.train(xgb_gen.create_data_set(X_tr, y_tr), xgb_gen.create_data_set(X_va, y_va))
        oof_preds["xgb"][valid_idx] = xgb_gen.predict(xgb_gen.create_data_set(X_va))

        # CatBoost
        cat_params = best.get("catboost", {}).copy()
        cat_params.setdefault("early_stopping_rounds", 200)
        cat_params.setdefault("verbose", 50)
        cat_gen = CCatBoostModelGenerator(model_tag=f"cat_fold{fold}", cb_params=cat_params)
        cat_gen.feature_names = feature_cols
        cat_gen.train(cat_gen.create_data_set(X_tr, y_tr), cat_gen.create_data_set(X_va, y_va))
        oof_preds["cat"][valid_idx] = cat_gen.predict(cat_gen.create_data_set(X_va))

    # Train meta LR
    X_meta = np.column_stack([oof_preds[k] for k in ("lgb", "xgb", "cat")])
    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(X_meta, y)
    print("Meta coefficients:", meta.coef_)

    # Prepare base models trained on full training set for validation prediction
    full_lgb = CLightGBMModelGenerator(model_tag="lgb_full", lgb_params=best.get("lightgbm", {}))
    full_lgb.feature_names = feature_cols
    full_lgb.train(full_lgb.create_data_set(X, y), full_lgb.create_data_set(X_val, y_val))
    pred_lgb = np.array(full_lgb.predict(full_lgb.create_data_set(X_val)))

    full_xgb = CXGBModelGenerator(model_tag="xgb_full", xgb_params=best.get("xgboost", {}))
    full_xgb.feature_names = feature_cols
    full_xgb.train(full_xgb.create_data_set(X, y), full_xgb.create_data_set(X_val, y_val))
    pred_xgb = np.array(full_xgb.predict(full_xgb.create_data_set(X_val)))

    full_cat = CCatBoostModelGenerator(model_tag="cat_full", cb_params=cat_params)
    full_cat.feature_names = feature_cols
    full_cat.train(full_cat.create_data_set(X, y), full_cat.create_data_set(X_val, y_val))
    pred_cat = np.array(full_cat.predict(full_cat.create_data_set(X_val)))

    ens_pred = meta.predict_proba(np.column_stack([pred_lgb, pred_xgb, pred_cat]))[:, 1]
    auc_val = roc_auc_score(y_val, ens_pred)
    print(f"Validation AUC (OOF stacking): {auc_val:.4f}")

if __name__ == "__main__":
    main() 