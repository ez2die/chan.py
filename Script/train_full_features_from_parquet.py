#!/usr/bin/env python3
"""train_full_features_from_parquet.py

使用完整增强特征 & 已对齐 fundingRate 列的训练脚本。

特点:
1. 直接读取 `data/processed/BTC_USDT_combined_processed.parquet` (或 --data 参数指定)；
2. 基于 EnhancedChanFeatureCalculator (v1) 计算所有特征, 包括 fundingRate 派生特征;
3. 不做任何特征筛选, 参与模型训练的列 ≈400+;
4. 默认按时间顺序 80/20 切分为训练/验证集;
5. 使用项目内封装的 LightGBMModelGenerator 训练; 输出模型与 meta 文件至 models/ 目录;
6. 支持 --recompute-features 强制删除缓存, --train-ratio 覆盖切分比例;

用法:
    python Script/train_full_features_from_parquet.py \
        [--data data/processed/BTC_USDT_combined_processed.parquet] \
        [--train-ratio 0.8] [--recompute-features]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 将项目根目录加入 path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ModelStrategy.EnhancedFeatureCalculator import EnhancedChanFeatureCalculator
from ModelStrategy.models.LightGBMModelGenerator import CLightGBMModelGenerator

CACHE_DIR = Path("data/processed/features_parquet")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser("Train full-feature model from parquet data")
    p.add_argument(
        "--data",
        default="data/processed/BTC_USDT_combined_processed.parquet",
        help="Parquet file path containing 1h OHLCV + fundingRate",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Portion of earliest samples used for training (0<ratio<1)",
    )
    p.add_argument(
        "--recompute-features",
        action="store_true",
        help="Ignore existing cached feature parquet files and recompute",
    )
    return p.parse_args()


def load_raw_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # 确保索引为 datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index or timestamp column")
    df = df.sort_index()
    return df


def compute_features(calc: EnhancedChanFeatureCalculator, raw_df: pd.DataFrame, cache_fp: Path, force: bool) -> pd.DataFrame:
    if cache_fp.exists() and not force:
        print(f"[INFO] Loading cached features: {cache_fp}")
        return pd.read_parquet(cache_fp)
    print(f"[INFO] Computing features for {len(raw_df)} rows…")
    feat_df = calc.calculate_all_features(raw_df)

    # 强制将 last_bsp_type 列统一为字符串，避免 PyArrow 类型推断混淆
    if 'last_bsp_type' in feat_df.columns:
        feat_df['last_bsp_type'] = feat_df['last_bsp_type'].astype(str)

    feat_df.to_parquet(cache_fp)
    print(f"[INFO] Saved feature cache to {cache_fp}")
    return feat_df


def main():
    args = parse_args()

    raw_df = load_raw_df(args.data)

    # 时间切分
    split_idx = int(len(raw_df) * args.train_ratio)
    train_raw = raw_df.iloc[:split_idx]
    val_raw = raw_df.iloc[split_idx:]
    print(f"Dataset split: train={len(train_raw)}, val={len(val_raw)}")

    calc = EnhancedChanFeatureCalculator()

    # 计算或加载特征
    ts_suffix = Path(args.data).stem  # e.g., BTC_USDT_combined_processed
    train_cache = CACHE_DIR / f"{ts_suffix}_train_all.parquet"
    val_cache = CACHE_DIR / f"{ts_suffix}_val_all.parquet"
    train_feat = compute_features(calc, train_raw, train_cache, args.recompute_features)
    val_feat = compute_features(calc, val_raw, val_cache, args.recompute_features)

    # 创建标签 (未来24h方向)
    train_lbl = calc.create_labels(train_feat)
    val_lbl = calc.create_labels(val_feat)

    # 对齐特征列
    feat_cols = calc.get_feature_names(train_lbl)

    # 过滤非数值列（object / category 等）
    numeric_feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train_lbl[c])]

    # 确保验证集也包含相同列
    missing_in_val = [c for c in numeric_feat_cols if c not in val_lbl.columns]
    for col in missing_in_val:
        val_lbl[col] = 0.0

    X_train = train_lbl[numeric_feat_cols].values
    y_train = train_lbl["binary_direction"].values
    X_val = val_lbl[numeric_feat_cols].values
    y_val = val_lbl["binary_direction"].values

    print(f"Training LightGBM with {len(numeric_feat_cols)} numeric features, samples: {len(X_train)}")
    model_tag = f"fullfeat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    lgb_gen = CLightGBMModelGenerator(model_tag=model_tag)
    lgb_gen.feature_names = numeric_feat_cols

    train_set = lgb_gen.create_data_set(X_train, y_train)
    val_set = lgb_gen.create_data_set(X_val, y_val)
    lgb_gen.train(train_set, val_set)

    # 评估
    val_pred = lgb_gen.predict(val_set)
    auc = roc_auc_score(y_val, val_pred)
    print(f"[RESULT] Validation AUC: {auc:.4f}")

    # 保存模型
    lgb_gen.save_model()
    print(f"Model saved with tag {model_tag}")


if __name__ == "__main__":
    main() 