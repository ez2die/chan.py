from __future__ import annotations

"""Chan.py Phase-2 – Feature importance analysis and selection module.

This standalone helper is used in Phase 2.2.1 to quantify per-feature contributions
from multiple GBDT-family models and generate a compact feature subset.

Key public methods
------------------
ChanFeatureSelector.analyze_feature_importance(models, feature_names)
    Aggregate importance from each model into a dict and compute mean importance.

ChanFeatureSelector.select_features(X, y, method="model_importance", target_features=200)
    Return a pruned list of feature names.

The class is intentionally dependency-light so it can be invoked inside both
analysis notebooks and automated pipelines.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize a 1-D vector to sum=1, fallback to uniform if all zeros."""
    total = arr.sum()
    if total == 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


class ChanFeatureSelector:
    """Feature importance analyser & selector for chan.py."""

    def __init__(self):
        # {model_name: {feature: score}}
        self.feature_importance_: Dict[str, Dict[str, float]] = {}
        self.selected_features_: List[str] = []

    # ---------------------------------------------------------------------
    # Importance aggregation
    # ---------------------------------------------------------------------
    def analyze_feature_importance(
        self,
        models: Dict[str, object],
        feature_names: List[str],
    ) -> Dict:
        """Compute per-model and average gain-based importance.

        Parameters
        ----------
        models: mapping of name -> model / ModelGenerator instance.
        feature_names: full feature list in correct order.
        """
        individual: Dict[str, Dict[str, float]] = {}

        for mname, model in models.items():
            try:
                # ModelGenerator has get_feature_importance
                if hasattr(model, "get_feature_importance"):
                    imp = model.get_feature_importance()
                # XGBoost Booster or ModelGenerator without wrapper
                elif hasattr(model, "model") and hasattr(model.model, "get_score"):
                    booster = model.model
                    imp = booster.get_score(importance_type="gain")
                # raw CatBoost model
                elif hasattr(model, "model") and hasattr(model.model, "get_feature_importance"):
                    cat = model.model
                    scores = cat.get_feature_importance(type="PredictionValuesChange")
                    imp = dict(zip(feature_names, scores))
                else:
                    # fallback: zero importance
                    imp = {f: 0.0 for f in feature_names}
            except Exception:
                # robust fallback
                imp = {f: 0.0 for f in feature_names}

            # Ensure every feature exists
            filled = {f: float(imp.get(f, 0.0)) for f in feature_names}
            individual[mname] = filled

        avg_imp = self._calculate_average_importance(individual, feature_names)
        top_feats = self._get_top_features(avg_imp, top_k=50)

        self.feature_importance_ = individual
        return {
            "individual_importance": individual,
            "average_importance": avg_imp,
            "top_features": top_feats,
        }

    # ------------------------------------------------------------------
    # Selection interfaces
    # ------------------------------------------------------------------
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "model_importance",
        target_features: int = 200,
    ) -> List[str]:
        """Return a pruned feature list according to *method*.

        Supported methods: model_importance, statistical, mutual_info, correlation.
        Currently we mainly implement model_importance and correlation which are
        sufficient for Phase 2.2.1. Others gracefully fall back to model_importance.
        """
        method = method.lower()
        if method == "model_importance" and self.feature_importance_:
            mean_imp = self._calculate_average_importance(self.feature_importance_, list(X.columns))
            ranked = sorted(mean_imp.items(), key=lambda kv: kv[1], reverse=True)
            self.selected_features_ = [f for f, _ in ranked[:target_features]]
            return self.selected_features_

        if method == "correlation":
            return self._correlation_based_selection(X, y, target_features)

        # Placeholder for statistical / mutual_info – fallback
        return self.select_features(X, y, "model_importance", target_features)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_average_importance(imp_dict: Dict[str, Dict[str, float]], feature_names: List[str]) -> Dict[str, float]:
        if not imp_dict:
            return {f: 0.0 for f in feature_names}

        all_scores = np.vstack([
            np.array([imp_dict[m][f] for f in feature_names]) for m in imp_dict
        ])
        # normalize each model then mean
        all_scores = np.apply_along_axis(_safe_normalize, 1, all_scores)
        mean_scores = all_scores.mean(axis=0)
        return {f: float(s) for f, s in zip(feature_names, mean_scores)}

    @staticmethod
    def _get_top_features(avg_importance: Dict[str, float], top_k: int = 50) -> List[str]:
        return [f for f, _ in sorted(avg_importance.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]

    # ------------------------------------------------------------------
    # Correlation-based pruning (simple version)
    # ------------------------------------------------------------------
    def _correlation_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: int,
        corr_threshold: float = 0.95,
    ) -> List[str]:
        # 1. compute abs corr matrix
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # 2. drop features with high corr but lower model importance (if available)
        drop_cols = set()
        importance_ref = None
        if self.feature_importance_:
            importance_ref = self._calculate_average_importance(self.feature_importance_, list(X.columns))
        for col in upper.columns:
            if col in drop_cols:
                continue
            highly_corr = [row for row in upper.index if upper.loc[row, col] > corr_threshold]
            for row in highly_corr:
                # decide which to drop
                if importance_ref:
                    keep = col if importance_ref[col] >= importance_ref[row] else row
                else:
                    # fallback – keep arbitrary but deterministic
                    keep = max(col, row)
                drop = row if keep == col else col
                drop_cols.add(drop)
        remaining = [c for c in X.columns if c not in drop_cols]
        # 3. if still too many features, cut by importance or simply head
        if len(remaining) > target_features:
            if self.feature_importance_:
                imp = importance_ref if importance_ref else {f: 0.0 for f in remaining}
                remaining = [f for f, _ in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:target_features]]
            else:
                remaining = remaining[:target_features]
        self.selected_features_ = remaining
        return remaining 