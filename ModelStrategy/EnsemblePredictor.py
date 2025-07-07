from __future__ import annotations

"""Ensemble predictor module for chan.py Phase-2.

This class aggregates multiple model generators (XGB, LightGBM, CatBoost …)
and produces a combined probability prediction via weighted average or stacking.
The implementation intentionally keeps external dependencies轻量；`scipy` is only
used for weight optimisation.
"""

from typing import Dict, List, Tuple, Optional, Any

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    from scipy.optimize import minimize  # type: ignore
except ImportError:  # pragma: no cover – will be caught in unit tests
    minimize = None  # type: ignore


class ChanEnsemblePredictor:
    """Chan.py 集成预测器

    Attributes
    ----------
    models : Dict[str, Any]
        已加入集成的模型对象（需实现 ``predict`` 接口）。
    weights : Dict[str, float]
        对应模型的权重（仅在 *weighted_average* 方法下使用）。
    meta_model : LogisticRegression | None
        stacking 模式下使用的二级模型。
    ensemble_method : str
        集成方式：``'weighted_average'`` 或 ``'stacking'``。
    """

    def __init__(self, method: str = "weighted_average") -> None:
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.meta_model: Optional[LogisticRegression] = None
        self.ensemble_method = method

    # ------------------------------------------------------------------
    #  模型管理
    # ------------------------------------------------------------------
    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """向集成中添加一个基模型。"""
        # --- 参数校验 -----------------------------------------------------
        if model is None:
            raise ValueError(
                f"模型 '{name}' 为 None，无法加入集成，请确认模型已正确初始化或加载完成。"
            )

        if not hasattr(model, "predict"):
            raise TypeError(
                f"模型 '{name}' 不包含 predict 方法，无法参与集成。"
            )

        self.models[name] = model
        self.weights[name] = weight

    # ------------------------------------------------------------------
    #  权重优化 / stacking 训练
    # ------------------------------------------------------------------
    def optimize_weights(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
        """基于验证集预测结果，使用 `scipy.optimize` 优化加权平均权重。"""
        if minimize is None:
            raise ImportError("scipy 未安装，无法执行权重优化，请 `pip install scipy`.")

        def objective(w: np.ndarray) -> float:
            w = w / w.sum()
            ens = sum(w_i * p for w_i, p in zip(w, predictions.values()))
            if ens.ndim == 1:
                return -roc_auc_score(y_true, ens)
            return -roc_auc_score(y_true, ens, multi_class='ovr', average='macro')

        n_models = len(predictions)
        init_w = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0)] * n_models
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        res = minimize(objective, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
        best_w = res.x / res.x.sum()

        for m_name, w_val in zip(predictions.keys(), best_w):
            self.weights[m_name] = float(w_val)
        return self.weights

    def fit_meta_model(self, X_meta: np.ndarray, y_true: np.ndarray) -> None:
        """训练 stacking 二级模型。"""
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_model.fit(X_meta, y_true)

    # ------------------------------------------------------------------
    #  预测接口
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, method: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.models:
            raise ValueError("Ensemble 中尚未添加任何模型。")

        method = method or self.ensemble_method

        # 收集各模型预测
        preds: Dict[str, np.ndarray] = {}
        for name, mdl in self.models.items():
            if mdl is None:
                raise ValueError(
                    f"Ensemble 中的模型 '{name}' 为 None，预测失败，请检查模型添加和加载流程。"
                )

            if not hasattr(mdl, "predict"):
                raise TypeError(
                    f"Ensemble 中的模型 '{name}' 缺少 predict 方法，无法进行预测。"
                )

            p = mdl.predict(X)
            preds[name] = np.asarray(p, dtype=float)

        if method == "weighted_average":
            return self._weighted_average(preds)
        elif method == "stacking":
            return self._stacking(preds)
        else:
            raise ValueError(f"不支持的集成方法: {method}")

    # ------------------------------------------------------------------
    #  内部实现
    # ------------------------------------------------------------------
    def _weighted_average(self, preds: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # 归一化权重
        total_w = sum(self.weights.values()) or 1.0
        norm_w = {k: v / total_w for k, v in self.weights.items()}

        first_pred = next(iter(preds.values()))
        ensemble_pred = np.zeros_like(first_pred, dtype=float)
        for name, p in preds.items():
            ensemble_pred += norm_w.get(name, 0.0) * p

        # 若为二分类概率向量 [p1,...] 长度1，保持一致输出
        meta_info = {
            "method": "weighted_average",
            "weights": norm_w,
            "individual_predictions": preds,
            "confidence": self._calc_confidence(preds),
        }
        return ensemble_pred, meta_info

    def _stacking(self, preds: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.meta_model is None:
            raise ValueError("meta_model 尚未训练，请先调用 fit_meta_model().")
        X_meta = np.column_stack(list(preds.values()))
        ens = self.meta_model.predict_proba(X_meta)[:, 1]
        meta_info = {
            "method": "stacking",
            "individual_predictions": preds,
            "meta_model": type(self.meta_model).__name__,
        }
        return ens, meta_info

    @staticmethod
    def _calc_confidence(preds: Dict[str, np.ndarray]) -> np.ndarray:
        # 对于多分类概率矩阵，使用信息熵衡量不确定性
        from scipy.stats import entropy
        first = next(iter(preds.values()))
        if first.ndim == 1:  # binary scalar prob
            pred_arr = np.column_stack(list(preds.values()))
            std = pred_arr.std(axis=1)
            return 1.0 / (1.0 + std)
        else:
            # 汇总加权平均概率后再算熵
            ens = np.mean(np.stack(list(preds.values()), axis=0), axis=0)
            conf = 1.0 - entropy(ens.T) / np.log(ens.shape[1])
            return conf

    # ------------------------------------------------------------------
    #  评估与持久化
    # ------------------------------------------------------------------
    def get_model_performance(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        perf = {}
        for name, mdl in self.models.items():
            p = np.asarray(mdl.predict(X_val), dtype=float)
            perf[name] = {"auc": roc_auc_score(y_val, p)}

        ens_pred, _ = self.predict(X_val)
        perf["ensemble"] = {"auc": roc_auc_score(y_val, ens_pred)}
        return perf

    def save_ensemble(self, filepath: str) -> None:
        data = {
            "ensemble_method": self.ensemble_method,
            "weights": self.weights,
            "model_names": list(self.models.keys()),
        }
        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)

    def load_ensemble(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
        self.weights = {k: float(v) for k, v in cfg.get("weights", {}).items()}
        self.ensemble_method = cfg.get("ensemble_method", "weighted_average")
        # 模型对象由调用者在外部 `add_model` 后再补充；这里仅恢复权重与方法。 