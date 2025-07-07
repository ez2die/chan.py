#!/usr/bin/env python3
"""Generate mixed Top-K feature whitelist and optionally run full training.

Highlights:
1. Load historical whitelist as baseline (enhanced_features_top300.txt or enhanced_features_list.txt).
2. Load the latest CatBoost model and compute feature importance.
3. Merge: iterate CatBoost importance order, then fill with baseline features until K features collected.
4. Save as `mixed_features_top{K}_<timestamp>.txt`.
5. If --no-train not specified, invoke ChanMLTrainer with the new whitelist.

Usage examples:
    python Script/train_mixed_topk.py --topk 300            # generate + train
    python Script/train_mixed_topk.py --topk 100 --no-train # only generate list
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import catboost as cb
import pandas as pd

# --- Resolve project root --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ModelStrategy.ChanMLTrainer import ChanMLTrainer


# ---------------------------------------------------------------------------
def latest_cat_model_meta() -> tuple[Path, dict]:
    """Return latest CatBoost meta file path and loaded JSON."""
    metas = sorted(Path("models").glob("catboost_btc_1h_*.json"))
    if not metas:
        raise FileNotFoundError("No catboost meta found under models/")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text())
    return meta_path, meta


def get_cat_importance(meta_path: Path, meta: dict) -> pd.Series:
    """Load CatBoost model and return importance Series (descending)."""
    model_path = Path(str(meta_path).replace("_meta.json", ".model"))
    model = cb.CatBoostClassifier()
    model.load_model(model_path)
    names: List[str] = meta["feature_names"]
    importance = model.get_feature_importance(type="FeatureImportance")
    return pd.Series(importance, index=names).sort_values(ascending=False)


def build_mixed_list(old_path: Path, cat_imp: pd.Series, top_k: int) -> List[str]:
    """Merge CatBoost-importance order with baseline list until top_k reached."""
    baseline = [ln.strip() for ln in old_path.read_text().splitlines() if ln.strip()]
    mixed: List[str] = []
    seen = set()
    # 1) Add CatBoost important features first
    for feat in cat_imp.index:
        if feat not in seen:
            mixed.append(feat)
            seen.add(feat)
        if len(mixed) == top_k:
            return mixed
    # 2) Fill with baseline list
    for feat in baseline:
        if feat not in seen:
            mixed.append(feat)
            seen.add(feat)
        if len(mixed) == top_k:
            break
    return mixed


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=200, help="Number of top features to keep")
    ap.add_argument("--dataset", default="btc_swap_1h_swaponly", help="Dataset name for training")
    ap.add_argument("--no-train", action="store_true", help="Only generate feature list, skip training")
    args = ap.parse_args()

    top_k = args.topk
    assert top_k > 0, "topk must be positive"

    # Decide baseline whitelist path
    baseline_path = Path("enhanced_features_top300.txt") if top_k > 200 else Path("enhanced_features_list.txt")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline whitelist {baseline_path} not found")

    # Build mixed list -------------------------------------------------------
    meta_path, meta = latest_cat_model_meta()
    cat_imp = get_cat_importance(meta_path, meta)
    mixed = build_mixed_list(baseline_path, cat_imp, top_k)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    mixed_file = Path(f"mixed_features_top{top_k}_{ts_tag}.txt")
    mixed_file.write_text("\n".join(mixed), encoding="utf-8")
    print(f"[INFO] Mixed whitelist saved to {mixed_file}  (len={len(mixed)})")

    # Optionally run training -----------------------------------------------
    if args.no_train:
        print("[INFO] --no-train specified, exiting after list generation.")
        return

    print("[INFO] Launching ChanMLTrainer with new whitelist…")
    base_cfg = ChanMLTrainer()._get_default_config()
    base_cfg.update({
        "dataset_name": args.dataset,
        "feature_calc_version": "v1",  # use full calc to avoid v2 instability on big feature sets
        "feature_selection": True,
        "selected_features_path": str(mixed_file),
    })

    trainer = ChanMLTrainer(config=base_cfg)
    trainer.run_full_training_pipeline()


if __name__ == "__main__":
    main() 