from __future__ import annotations
"""
Phase-2 任务 2.2.2
---------------------------------
优化版本特征计算器（v2）

• 通过分组开关、白名单裁剪等机制，在保持完整功能的前提下降低
  训练/推理负担。
• 设计为 "轻量包装" —— 继承 v1 版 EnhancedChanFeatureCalculator，
  复用全部底层实现，避免重复维护。
"""
from typing import Dict, List

import pandas as pd

from .EnhancedFeatureCalculator import (
    EnhancedChanFeatureCalculator as _BaseCalculator,
)


class EnhancedChanFeatureCalculator_v2(_BaseCalculator):
    """v2 – 可按组裁剪 / 只输出指定特征的优化版"""

    # ----------------------------- 初始化 ----------------------------- #
    def __init__(self, feature_config: Dict | None = None) -> None:
        super().__init__()
        # 默认配置 + 外部覆盖
        self.feature_config: Dict = feature_config or self._get_default_config()

        # 映射组名 -> 基类已有的特征计算函数
        self.feature_groups = {
            "basic": self._add_basic_features,
            "technical": self._add_technical_features,
            "volume": self._add_volume_features,
            "chan_theory": self._add_chan_features,
            "multi_timeframe": self._add_multi_timeframe_features,
            "volatility": self._add_volatility_features,
            "momentum": self._add_momentum_features,
            "funding_rate": self._add_funding_rate_features,
            # pattern 统一复用高级缠论/交互特征，可按需要替换
            "pattern": self._add_advanced_chan_features,
        }

    # -------------------------- 配置模板 ----------------------------- #
    @staticmethod
    def _get_default_config() -> Dict:
        """默认组开关 & 白名单设置"""
        return {
            "basic": True,
            "technical": True,
            "volume": True,
            "chan_theory": True,
            "multi_timeframe": True,
            "volatility": True,
            "momentum": True,
            "funding_rate": True,
            "pattern": True,  # 高级形态/交互
            # 仅保留白名单特征（Top-N）
            "selected_features_only": False,
            "selected_features": [],
        }

    # ------------------------- 主计算入口 ---------------------------- #
    def calculate_optimized_features(
        self, df: pd.DataFrame, selected_features: List[str] | None = None
    ) -> pd.DataFrame:
        """
        根据配置计算所需特征。

        参数
        ----
        df : 原始 K 线 DataFrame（必须包含 open/high/low/close/volume/timestamp）
        selected_features : 可选，若传入则自动开启 `selected_features_only`
        """
        if selected_features:
            self.feature_config["selected_features_only"] = True
            self.feature_config["selected_features"] = selected_features

        features_df = df.copy()

        # 逐组计算
        for group_name, calc_func in self.feature_groups.items():
            if self.feature_config.get(group_name, False):
                try:
                    features_df = calc_func(features_df)
                except Exception as exc:  # 容错 —— 单组失败不影响整体
                    print(f"⚠️  {group_name} 特征计算失败: {exc}")

        # 如果只保留白名单特征
        if self.feature_config.get("selected_features_only", False):
            base_cols = ["open", "high", "low", "close", "volume", "timestamp"]
            keep = base_cols + [
                f
                for f in self.feature_config.get("selected_features", [])
                if f in features_df.columns
            ]
            features_df = features_df[keep]

        return features_df

    # Backwards compatibility so Trainer can call old interface
    def calculate_all_features(
        self, df: pd.DataFrame, selected_features: List[str] | None = None
    ) -> pd.DataFrame:  # type: ignore[override]
        """Alias for calculate_optimized_features to preserve v1 API."""
        return self.calculate_optimized_features(df, selected_features)

    # ------------------------- 信息统计 ----------------------------- #
    def get_feature_groups_info(self) -> Dict:
        """
        快速统计：各组会产生多少列？（基于 100 行示例数据）

        用于在实验前评估特征膨胀程度。
        """
        sample = pd.DataFrame(
            {
                "open": [100] * 100,
                "high": [105] * 100,
                "low": [95] * 100,
                "close": [102] * 100,
                "volume": [1000] * 100,
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="H"),
            }
        )

        info: Dict[str, Dict] = {}
        for name, func in self.feature_groups.items():
            try:
                orig_cols = set(sample.columns)
                out = func(sample.copy())
                new_cols = set(out.columns) - orig_cols
                info[name] = {"feature_count": len(new_cols), "features": list(new_cols)}
            except Exception as exc:
                info[name] = {"feature_count": 0, "features": [], "error": str(exc)}
        return info 