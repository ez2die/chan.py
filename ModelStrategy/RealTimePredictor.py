"""Real-time predictor for Chan.py Phase-2.

Provides lightweight implementation sufficient for offline simulation &
strategy integration (predict_single).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import os
from glob import glob
from pathlib import Path
from collections import deque

import numpy as np

# lazy imports to avoid heavy deps when not used
from ModelStrategy.EnsemblePredictor import ChanEnsemblePredictor
from ModelStrategy.EnhancedFeatureCalculator_v2 import (
    EnhancedChanFeatureCalculator_v2,
)
from ModelStrategy.models.LightGBMModelGenerator import (
    CLightGBMModelGenerator,
)
from ModelStrategy.models.XGBModelGenerator import CXGBModelGenerator
from ModelStrategy.models.CatBoostModelGenerator import (
    CCatBoostModelGenerator,
)


class ChanRealTimePredictor:
    """Simple real-time predictor wrapper.

    It loads pre-trained generators, computes selected features for a single
    bar dict ({timestamp, open, high, low, close, volume}), and returns
    prediction probability together with metadata.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config: Dict = config or {}
        self.models: Dict[str, object] = {}
        self.ensemble_predictor: Optional[ChanEnsemblePredictor] = None
        self.feature_calculator = EnhancedChanFeatureCalculator_v2()
        # cache last bar timestamp to avoid duplicate logs
        self.prediction_cache: Dict[str, Dict] = {}

        # maintain recent bars for features requiring history (e.g., Chan theory)
        self._bar_buffer: deque = deque(maxlen=self.config.get("history_window", 300))

        # 自动加载默认模型与特征白名单
        if self.config.get("auto_load", True):
            try:
                self._auto_load_default_models()
                # After models loaded, align feature list to model meta if available
                self._setup_feature_list_from_models()
            except Exception as e:
                print(f"[RealTimePredictor] Auto-load models failed: {e}")
            try:
                # fallback if still no selected features
                if not self.feature_calculator.feature_config.get("selected_features"):
                    self._setup_default_feature_selection()
            except Exception as e:
                print(f"[RealTimePredictor] Auto-setup features failed: {e}")

    # ------------------------------------------------------------------
    # Model loading / ensemble
    # ------------------------------------------------------------------
    def load_models(self, model_paths: Dict[str, str]):
        for name, path in model_paths.items():
            name_low = name.lower()
            if name_low == "lgb":
                gen = CLightGBMModelGenerator("rt")
                gen.get_model_path = lambda p=path: p  # override
                gen.load_model()
            elif name_low == "xgb":
                gen = CXGBModelGenerator("rt")
                gen.get_model_path = lambda p=path: p
                gen.load_model()
            elif name_low == "cat":
                gen = CCatBoostModelGenerator("rt")
                gen.get_model_path = lambda p=path: p
                gen.load_model()
            else:
                raise ValueError(f"unsupported model {name}")
            self.models[name_low] = gen

    def setup_ensemble(self, ensemble_config: Dict):
        ens = ChanEnsemblePredictor()
        for name, gen in self.models.items():
            ens.add_model(name, gen, weight=1.0)
        ens.ensemble_method = ensemble_config.get("method", "weighted_average")
        if ens.ensemble_method == "weighted_average":
            ens.weights = ensemble_config.get("weights", {n: 1 for n in self.models})
        self.ensemble_predictor = ens

    # ------------------------------------------------------------------
    # 自动设置
    # ------------------------------------------------------------------
    def _auto_load_default_models(self):
        """查找 models/ 目录下最新日期戳的 lgb/xgb/cat 模型并加载。"""
        model_dir = Path(self.config.get("model_dir", "models"))
        pattern_map = {
            "lgb": "lgb*_1h_*.model",
            "xgb": "xgb*_1h_*.model",
            "cat": "catboost*_1h_*.model",
        }

        found = {}
        for mname, pat in pattern_map.items():
            cand = sorted(model_dir.glob(pat))
            if cand:
                found[mname] = str(cand[-1])  # 取最新

        if not found:
            raise FileNotFoundError("No default model files found in 'models/'")

        self.load_models(found)
        self.setup_ensemble({"method": "weighted_average"})

    def _setup_default_feature_selection(self):
        """读取 enhanced_features_list.txt 前 200 行作为白名单。"""
        whitelist_path = Path(self.config.get("feature_list", "enhanced_features_list.txt"))
        if not whitelist_path.exists():
            raise FileNotFoundError(str(whitelist_path))

        features = [ln.strip() for ln in whitelist_path.read_text().splitlines() if ln.strip()][:200]
        cfg = self.feature_calculator.feature_config
        cfg["selected_features_only"] = True
        cfg["selected_features"] = features

        # 确保所有特征组打开（与训练一致）
        for key in cfg.keys():
            if key not in {"selected_features_only", "selected_features"}:
                cfg[key] = True

    def _setup_feature_list_from_models(self):
        """If loaded models contain feature_names metadata, use it as canonical order."""
        candidate_lists = []
        min_len = None
        for gen in self.models.values():
            fn = getattr(gen, "feature_names", None)
            if fn:
                candidate_lists.append(fn)
            else:
                # fallback to num_features if available
                if hasattr(gen, "model") and gen.model is not None:
                    # Try to detect expected feature count from booster
                    m = gen.model
                    expected_n = None
                    if hasattr(m, "num_feature") and callable(getattr(m, "num_feature")):
                        expected_n = m.num_feature()
                    elif hasattr(m, "num_feature"):
                        expected_n = m.num_feature
                    elif hasattr(m, "num_features") and callable(getattr(m, "num_features")):
                        expected_n = m.num_features()
                    elif hasattr(m, "num_features"):
                        expected_n = m.num_features

                    dynamic_cols = [c for c in feat_df.columns if c not in base_exclude]

                    if expected_n is None:
                        # fallback to length of dynamic_cols
                        expected_n = len(dynamic_cols)

                    if hasattr(self, "_ordered_feature_list") and len(self._ordered_feature_list) >= expected_n:
                        model_cols = self._ordered_feature_list[:expected_n]
                    else:
                        model_cols = dynamic_cols[:expected_n]

            if not candidate_lists and min_len is None:
                return  # nothing found

            if candidate_lists:
                ordered = min(candidate_lists, key=len)
            else:
                # derive from whitelist
                wl_path = Path(self.config.get("feature_list", "enhanced_features_list.txt"))
                features = [ln.strip() for ln in wl_path.read_text().splitlines() if ln.strip()][:min_len]
                ordered = features

            cfg = self.feature_calculator.feature_config
            cfg["selected_features_only"] = True
            cfg["selected_features"] = ordered
            self._ordered_feature_list = ordered

    def _features_from_bar(self, bar: Dict) -> Tuple[np.ndarray, str]:
        import pandas as pd

        df = pd.DataFrame([bar])
        feat_df = self.feature_calculator.calculate_optimized_features(df)
        base_exclude = {"open", "high", "low", "close", "volume", "timestamp"}
        if hasattr(self, "_ordered_feature_list"):
            feature_cols = [c for c in self._ordered_feature_list if c in feat_df.columns]
        else:
            feature_cols = [c for c in feat_df.columns if c not in base_exclude]
        arr = feat_df[feature_cols].values.astype(float)
        return arr, feature_cols  # cols for future use

    def predict_single(self, bar: Dict) -> Dict:
        ts = bar.get("timestamp", "0")

        # 1) 生成完整特征 DataFrame
        import pandas as pd
        # update buffer
        self._bar_buffer.append(bar)
        window_len = self.config.get("feature_window", 120)
        buf_list = list(self._bar_buffer)[-window_len:]
        df = pd.DataFrame(buf_list)
        feat_df = self.feature_calculator.calculate_optimized_features(df)
        # take only last row as current bar features
        feat_last = feat_df.iloc[-1:]

        base_exclude = {"open", "high", "low", "close", "volume", "timestamp"}

        preds = {}
        for name, gen in self.models.items():
            # 2) 找到该模型预期的列顺序
            model_cols = None
            if getattr(gen, "feature_names", None):
                model_cols = [c for c in gen.feature_names if c in feat_df.columns]
            elif hasattr(gen, "model") and gen.model is not None:
                # Try to detect expected feature count from booster
                m = gen.model
                expected_n = None
                if hasattr(m, "num_feature") and callable(getattr(m, "num_feature")):
                    expected_n = m.num_feature()
                elif hasattr(m, "num_feature"):
                    expected_n = m.num_feature
                elif hasattr(m, "num_features") and callable(getattr(m, "num_features")):
                    expected_n = m.num_features()
                elif hasattr(m, "num_features"):
                    expected_n = m.num_features

                dynamic_cols = [c for c in feat_df.columns if c not in base_exclude]

                if expected_n is None:
                    # fallback to length of dynamic_cols
                    expected_n = len(dynamic_cols)

                if hasattr(self, "_ordered_feature_list") and len(self._ordered_feature_list) >= expected_n:
                    model_cols = self._ordered_feature_list[:expected_n]
                else:
                    model_cols = dynamic_cols[:expected_n]
            else:
                model_cols = [c for c in feat_df.columns if c not in base_exclude]

            # 3) 保障列存在
            model_cols = [c for c in model_cols if c in feat_df.columns]
            if not model_cols:
                raise ValueError(f"No matching feature columns for model {name}")

            X = feat_last[model_cols].values.astype(float)
            ds = gen.create_data_set(X)
            prob = gen.predict(ds)[0]
            preds[name] = prob

        if self.ensemble_predictor and len(preds) > 1:
            # weighted average using current weights
            weights = self.ensemble_predictor.weights or {k:1.0 for k in preds}
            total_w = sum(weights.values())
            prob = sum(weights.get(k,1.0)*v for k,v in preds.items())/total_w
            meta_info = {"method":"weighted_average","weights":weights}
        else:
            # fallback to first model
            prob = list(preds.values())[0]
            meta_info = {}

        # simple threshold
        buy_th = self.config.get("signal_thresholds", {}).get("buy", 0.7)
        sell_th = self.config.get("signal_thresholds", {}).get("sell", 0.3)
        if prob > buy_th:
            sig = {"type": "BUY", "strength": (prob - buy_th) / max(1 - buy_th, 1e-6), "probability": prob}
        elif prob < sell_th:
            sig = {"type": "SELL", "strength": (sell_th - prob) / max(sell_th, 1e-6), "probability": prob}
        else:
            sig = {"type": "HOLD", "strength": 0.0, "probability": prob}

        result = {
            "timestamp": ts,
            "prediction": prob,
            "individual_predictions": preds,
            "signal": sig,
        }
        result.update(meta_info)
        self.prediction_cache[str(ts)] = result
        return result 