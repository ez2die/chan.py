import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure project root is importable when script is executed directly from tools/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ModelStrategy.ChanMLTrainer import ChanMLTrainer
from ModelStrategy.FeatureSelector import ChanFeatureSelector


def main():
    """Run Phase-2.2.1 feature importance analysis and selection.

    The script trains (or re-trains) the three baseline models, aggregates their
    feature importances, selects the top-200 features and stores artefacts under
    the project root:

    • enhanced_features_list.txt        – one feature per line
    • enhanced_features_stats.json      – full importance analysis result
    """

    # ------------------------------------------------------------------
    # 1. Prepare data & models via ChanMLTrainer
    # ------------------------------------------------------------------
    config = {
        "data_dir": "data/ml_datasets",
        "model_dir": "models/feature_selection_tmp",  # separate dir in case we retrain
        "log_dir": "logs/feature_selection",
        "dataset_name": "main_training_BTC_USDT_1h",
        "target_column": "binary_direction",
        "models": ["xgb", "lgb", "cat"],
        "random_state": 42,
    }

    trainer = ChanMLTrainer(config)
    train_df, val_df = trainer.prepare_training_data()
    results = trainer.train_models(train_df, val_df)

    # ------------------------------------------------------------------
    # 2. Analyse feature importance across models
    # ------------------------------------------------------------------
    selector = ChanFeatureSelector()
    model_objs = {name: info["model"] for name, info in results.items() if name != "ensemble"}
    feature_names = results["xgb"]["feature_names"]

    analysis = selector.analyze_feature_importance(model_objs, feature_names)

    # ------------------------------------------------------------------
    # 3. Select top-200 features (default method=model_importance)
    # ------------------------------------------------------------------
    selected_feats = selector.select_features(
        train_df[feature_names],  # X
        train_df[config["target_column"]],
        method="model_importance",
        target_features=200,
    )

    # ------------------------------------------------------------------
    # 4. Persist artefacts
    # ------------------------------------------------------------------
    with open("enhanced_features_list.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(selected_feats))

    meta_out = {
        "generated_at": datetime.now().isoformat(),
        "total_features": len(feature_names),
        "selected_features": len(selected_feats),
        "top_50": analysis["top_features"],
    }
    meta_out.update(analysis)

    with open("enhanced_features_stats.json", "w", encoding="utf-8") as fp:
        json.dump(meta_out, fp, indent=2, ensure_ascii=False)

    print(f"✅ Feature selection completed: {len(selected_feats)} features saved to enhanced_features_list.txt")


if __name__ == "__main__":
    main() 