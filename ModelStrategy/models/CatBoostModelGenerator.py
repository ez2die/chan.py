from __future__ import annotations

"""CatBoost ModelGenerator for chan.py Phase-2."""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import catboost as cb

from ModelStrategy.ModelGenerator import CModelGenerator, CDataSet


class CCatDataSet(CDataSet):
    """CatBoost 数据集封装."""

    def get_count(self) -> int:
        return self.data.num_row()

    def get_pos_count(self) -> int:
        return int(sum(self.get_label()))

    def get_label(self) -> List[float]:
        return self.data.get_label()


class CCatBoostModelGenerator(CModelGenerator):
    """CatBoost 模型生成器."""

    def __init__(self, model_tag: str, cb_params: Dict[str, Any] | None = None):
        super().__init__(model_type="catboost", model_tag=model_tag)
        base = self._default_params()
        if cb_params:
            base.update(cb_params)
        self.cb_params = base
        self.model: cb.CatBoostClassifier | None = None
        self.feature_names: List[str] | None = None

    @staticmethod
    def _default_params() -> Dict[str, Any]:
        return {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "random_seed": 42,
            "verbose": 50,
            "early_stopping_rounds": 30,
        }

    # ------------------------------------------------------------------
    #  CModelGenerator API 实现
    # ------------------------------------------------------------------
    def create_data_set(self, X: np.ndarray, y: np.ndarray | None = None) -> CCatDataSet:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        pool = cb.Pool(X, label=y)
        return CCatDataSet(pool)

    def create_train_test_set(self, sample_iter) -> Tuple[CCatDataSet, CCatDataSet]:
        raise NotImplementedError

    def train(self, train_set: CCatDataSet, test_set: CCatDataSet) -> None:
        # 提取 verbose 参数，若未提供则默认为 100；并从参数字典中移除避免重复
        verbose_lvl = self.cb_params.pop("verbose", 100)
        self.model = cb.CatBoostClassifier(**self.cb_params)
        self.model.fit(train_set.data, eval_set=test_set.data, verbose=verbose_lvl)

    def predict(self, dataset: CCatDataSet) -> List[float]:
        if self.model is None:
            raise ValueError("模型未训练")
        proba = self.model.predict_proba(dataset.data)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1].tolist()
        if proba.shape[1] == 1:
            return proba[:, 0].tolist()
        return proba.tolist()

    def save_model(self) -> None:
        if self.model is None:
            raise ValueError("模型未训练")
        model_path = self.get_model_path()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
        meta = {
            "model_type": self.model_type,
            "model_tag": self.model_tag,
            "cb_params": self.cb_params,
            "num_features": self.model.get_n_features_in() if hasattr(self.model, "get_n_features_in") else len(self.model.feature_names_ or []),
            "feature_names": self.feature_names,
            "best_iteration": self.model.get_best_iteration(),
        }
        with open(model_path.replace(".model", "_meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)

    def load_model(self) -> int:
        model_path = self.get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model = cb.CatBoostClassifier()
        self.model.load_model(model_path)
        meta_path = model_path.replace(".model", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
                self.cb_params = meta.get("cb_params", self.cb_params)
                self.feature_names = meta.get("feature_names", None)
        if self.feature_names:
            return len(self.feature_names)
        if hasattr(self.model, "feature_names_") and self.model.feature_names_:
            return len(self.model.feature_names_)
        return 0 