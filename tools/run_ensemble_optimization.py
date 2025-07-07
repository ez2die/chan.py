from __future__ import annotations

"""Optimize ensemble weights or stacking meta-model and evaluate metrics.
Usage:
    python tools/run_ensemble_optimization.py --features enhanced_features_top300.txt \
        --models_dir models/feature_top300_run --log_dir logs/ensemble_opt
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# ensure project root in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
)

from ModelStrategy.ChanMLTrainer import ChanMLTrainer
from ModelStrategy.EnsemblePredictor import ChanEnsemblePredictor


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="enhanced_features_top300.txt")
    ap.add_argument("--models_dir", default="models/ensemble_opt_run")
    ap.add_argument("--log_dir", default="logs/ensemble_opt_run")
    args = ap.parse_args()

    cfg = {
        "data_dir": "data/ml_datasets",
        "model_dir": args.models_dir,
        "log_dir": args.log_dir,
        "dataset_name": "main_training_BTC_USDT_1h",
        "target_column": "binary_direction",
        "models": ["xgb", "lgb", "cat"],
        "random_state": 42,
        "selected_features_path": args.features,
    }

    trainer = ChanMLTrainer(cfg)

    # --- prepare data & train (quick, they run fast with <10 rounds) ---
    train_df, val_df = trainer.prepare_training_data()
    results = trainer.train_models(train_df, val_df)

    y_val = val_df[cfg["target_column"].strip()].values

    pred_dict = {m: np.array(info["val_pred"]) for m, info in results.items() if m in ["xgb", "lgb", "cat"]}

    # baseline equal-weight ensemble
    eq_pred = np.mean(list(pred_dict.values()), axis=0)
    baseline_metrics = evaluate_metrics(y_val, eq_pred)

    # -- weight optimization --
    predictor = ChanEnsemblePredictor()
    for m, info in results.items():
        if m in pred_dict:
            class _Stub:  # minimal predict API to satisfy add_model
                def __init__(self, arr):
                    self._arr = arr
                def predict(self, X):
                    return self._arr
            predictor.add_model(m, _Stub(pred_dict[m]))
    opt_weights = predictor.optimize_weights(pred_dict, y_val)
    opt_pred, _ = predictor.predict(None)  # predictor uses cached predictions
    opt_metrics = evaluate_metrics(y_val, opt_pred)

    # -- stacking meta-model --
    X_meta = np.column_stack(list(pred_dict.values()))
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_meta, y_val)
    stack_pred = meta_clf.predict_proba(X_meta)[:, 1]
    stack_metrics = evaluate_metrics(y_val, stack_pred)

    report = {
        "generated_at": datetime.now().isoformat(),
        "feature_file": args.features,
        "baseline_equal_weight": baseline_metrics,
        "optimized_weights": {
            "weights": opt_weights,
            "metrics": opt_metrics,
        },
        "stacking_meta_model": {
            "coefficients": meta_clf.coef_.tolist(),
            "metrics": stack_metrics,
        },
    }
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.models_dir) / "ensemble_opt_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    print("==== Ensemble Optimization Report ====")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main() 