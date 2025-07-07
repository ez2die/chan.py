#!/usr/bin/env python3
"""Run full training using ALL features produced by EnhancedFeatureCalculator_v2.

Usage:
    python Script/train_all_features.py [--dataset DATASET_NAME]

It reuses default trainer settings except disables the Top-N feature whitelist so
that every computed feature (≈400+) participates in model training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root in import path so that `ModelStrategy` can be resolved
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ModelStrategy.ChanMLTrainer import ChanMLTrainer


def parse_args():
    p = argparse.ArgumentParser("Train models with full feature space")
    p.add_argument(
        "--dataset", default="btc_swap_1h_swaponly", help="Dataset name registered in DatasetManager"
    )
    p.add_argument(
        "--recompute-features",
        action="store_true",
        help="Delete existing cached feature parquet files and recompute with current calculator",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build full config: start from trainer defaults then override
    base_cfg = ChanMLTrainer()._get_default_config()
    overrides = {
        "dataset_name": args.dataset,
        "feature_calc_version": "v1",
        "feature_selection": False,  # disable whitelist
        "selected_features_path": None,
    }
    base_cfg.update(overrides)

    # Optionally remove existing cached feature parquet files to force recompute
    if args.recompute_features:
        feat_dir = Path("data/processed/features")
        if feat_dir.exists():
            for fp in feat_dir.glob(f"{args.dataset}_*all*.parquet"):
                print(f"[INFO] Removing cached feature file: {fp}")
                try:
                    fp.unlink()
                except Exception as e:
                    print(f"  ⚠️  Failed to delete {fp}: {e}")

    trainer = ChanMLTrainer(config=base_cfg)
    try:
        trainer.run_full_training_pipeline()
    except Exception as exc:
        import traceback, sys
        print("\n[ERROR] Pipeline aborted due to exception:\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 