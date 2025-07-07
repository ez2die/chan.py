#!/usr/bin/env python3
"""Generate mixed Top-200 feature whitelist (old Top-200 ∪ 新 CatBoost Top-Gain) and retrain.

1. Load old whitelist (`enhanced_features_list.txt`, top 200 lines)
2. Load latest CatBoost model meta/importance; compute前 200 Gain features
3. Merge: keep CatBoost importance顺序；依次填入特征直到200
4. Save list as `mixed_features_top200_<ts>.txt`
5. Run ChanMLTrainer (feature_calc_version='v1') with selected_features_path pointing to this file.

Usage:
    python Script/train_mixed_top200.py [--dataset DATA]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import catboost as cb
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ModelStrategy.ChanMLTrainer import ChanMLTrainer


def latest_cat_model_meta() -> tuple[str, dict]:
    metas = sorted(Path("models").glob("catboost_btc_1h_*.json"))
    if not metas:
        raise FileNotFoundError("No catboost meta found")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text())
    return str(meta_path), meta


def get_cat_importance(meta_path: str, meta: dict) -> pd.Series:
    model_path = meta_path.replace("_meta.json", ".model")
    model = cb.CatBoostClassifier()
    model.load_model(model_path)
    names = meta["feature_names"]
    importance = model.get_feature_importance(type="FeatureImportance")
    return pd.Series(importance, index=names).sort_values(ascending=False)


def build_mixed_list(old_path: str, cat_imp: pd.Series, top_k: int = 200) -> list[str]:
    old_list = [ln.strip() for ln in Path(old_path).read_text().splitlines() if ln.strip()][:top_k]
    mixed: list[str] = []
    seen = set()
    # 先按 CatBoost 重要度加入
    for feat in cat_imp.index:
        if feat not in seen:
            mixed.append(feat)
            seen.add(feat)
        if len(mixed) == top_k:
            return mixed
    # 不足再补旧表
    for feat in old_list:
        if feat not in seen:
            mixed.append(feat)
            seen.add(feat)
        if len(mixed) == top_k:
            break
    return mixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="btc_swap_1h_swaponly")
    args = parser.parse_args()

    meta_path, meta = latest_cat_model_meta()
    imp_series = get_cat_importance(meta_path, meta)

    mixed = build_mixed_list("enhanced_features_list.txt", imp_series, 200)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    mixed_file = Path("mixed_features_top200_" + ts_tag + ".txt")
    mixed_file.write_text("\n".join(mixed), encoding="utf-8")
    print(f"[INFO] Mixed whitelist saved to {mixed_file} (len={len(mixed)})")

    # Prepare trainer config
    base_cfg = ChanMLTrainer()._get_default_config()
    base_cfg.update({
        "dataset_name": args.dataset,
        "feature_calc_version": "v1",  # use full calculator
        "feature_selection": True,
        "selected_features_path": str(mixed_file),
    })

    trainer = ChanMLTrainer(config=base_cfg)
    trainer.run_full_training_pipeline()


if __name__ == "__main__":
    main() 