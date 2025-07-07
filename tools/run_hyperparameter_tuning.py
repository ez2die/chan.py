#!/usr/bin/env python
"""Run hyper-parameter optimisation for LightGBM and CatBoost.

Example:
    python tools/run_hyperparameter_tuning.py \
        --features enhanced_features_top300.txt \
        --output models/hpo_top300.json --n_trials 30 \
        | tee logs/tmp_hpo/optuna_$(date +%H%M%S).log

The script will:
1. Load dataset via ChanMLTrainer (to reuse preprocessing / feature subset)
2. Extract numpy arrays (X_train, y_train)
3. Run Optuna search for LightGBM, XGBoost & CatBoost
4. Save the best params as JSON for later use
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Ensure project root import
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ModelStrategy.ChanMLTrainer import ChanMLTrainer
from ModelStrategy.HyperparameterOptimizer import ChanHyperparameterOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chan.py – Hyperparameter tuning runner")
    p.add_argument("--features", type=str, default="enhanced_features_top300.txt", help="Path to selected feature list")
    p.add_argument("--n_trials", type=int, default=30, help="Optuna trials per model")
    p.add_argument("--cv_folds", type=int, default=3, help="CV folds")
    p.add_argument("--n_jobs", type=int, default=4, help="Parallel jobs (-1 for all CPUs)")
    p.add_argument("--cv_type", type=str, choices=["stratified", "time"], default="time", help="Cross-validation type")
    p.add_argument("--purge_gap", type=int, default=24, help="Gap size for Purged KFold when cv_type=time")
    p.add_argument("--output", type=str, default="models/hpo_results.json", help="Output json with best params")
    p.add_argument("--dataset", type=str, default="main_training_BTC_USDT_1h", help="Dataset name (ChanMLTrainer) ")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Prepare data via ChanMLTrainer (only preprocessing, no training)
    # ------------------------------------------------------------------
    trainer_cfg = {
        "data_dir": "data/ml_datasets",
        "model_dir": "models/tmp_hpo",
        "log_dir": "logs/tmp_hpo",
        "dataset_name": args.dataset,
        "target_column": "binary_direction",
        "models": [],  # we won't train inside trainer
        "selected_features_path": args.features,
        "random_state": 42,
    }

    trainer = ChanMLTrainer(trainer_cfg)
    train_df, _ = trainer.prepare_training_data()

    # trainer.prepare_training_data returns cleaned df with labels
    y = train_df[trainer_cfg["target_column"]].values.astype(np.float32)

    # ---- Fix potential data leakage ----
    # Use FeatureCalculator helper to obtain *only* valid feature columns, excluding
    # any future-return-derived or label columns (future_returns, direction, etc.).
    feature_cols = trainer.feature_calculator.get_feature_names(train_df)
    X = train_df[feature_cols].values.astype(np.float32)

    print(f"🔍 Dataset ready – X shape {X.shape}, features {len(feature_cols)}, positive ratio {y.mean():.3f}")

    # ------------------------------------------------------------------
    # Run optimisation
    # ------------------------------------------------------------------
    optimizer = ChanHyperparameterOptimizer(n_trials=args.n_trials, cv_folds=args.cv_folds, n_jobs=args.n_jobs, cv_type=args.cv_type, purge_gap=args.purge_gap, random_state=42)

    print("⚡ Optimising LightGBM parameters …")
    best_lgb = optimizer.optimize_lightgbm(X, y)
    print("✅ LightGBM best params:", best_lgb)

    print("⚡ Optimising CatBoost parameters …")
    best_cat = optimizer.optimize_catboost(X, y)
    print("✅ CatBoost best params:", best_cat)

    print("⚡ Optimising XGBoost parameters …")
    best_xgb = optimizer.optimize_xgboost(X, y)
    print("✅ XGBoost best params:", best_xgb)

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "dataset": args.dataset,
                "feature_list": args.features,
                "n_trials": args.n_trials,
                "cv_folds": args.cv_folds,
                "best_params": optimizer.best_params,
            },
            fp,
            indent=2,
        )
    try:
        rel = out_path.relative_to(ROOT_DIR)
    except ValueError:
        rel = out_path
    print(f"💾 Saved best parameters to {rel}")


if __name__ == "__main__":  # pragma: no cover
    main() 