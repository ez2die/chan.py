# PEP 563/PEP 649: postpone evaluation of type annotations to avoid runtime errors on Python<3.10
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from typing import List, Tuple, Dict, Any
import os
import json

import sys
from pathlib import Path
# Ensure the root of the repo is on the path so that "ModelStrategy" can be imported when
# this file is executed directly (e.g. during unit tests or ad-hoc scripts).
sys.path.append(str(Path(__file__).parent.parent.parent))

from ModelStrategy.ModelGenerator import CModelGenerator, CDataSet


class CLightGBMDataSet(CDataSet):
    """LightGBM 数据集封装类。"""

    def __init__(self, data: lgb.Dataset, tag: str = "tmp", raw_features: np.ndarray | None = None):
        super().__init__(data, tag)
        # 额外保存原始特征矩阵，便于推理阶段直接使用。
        self.raw_features = raw_features

    # LightGBM 原生接口提供了 size/label 获取函数，这里做一层封装方便与父类抽象保持一致。
    def get_count(self) -> int:
        return self.data.num_data()

    def get_pos_count(self) -> int:
        labels = self.get_label()
        return int(sum(labels))

    def get_label(self) -> List[float]:
        return self.data.get_label().tolist()


class CLightGBMModelGenerator(CModelGenerator):
    """LightGBM 模型生成器。"""

    def __init__(self, model_tag: str, lgb_params: Dict[str, Any] | None = None):
        # 这里的 model_type 参数用于构造保存路径，如 models/lgb_xxx.model
        super().__init__(model_type="lgb", model_tag=model_tag)
        # 如果没有显式传入参数，使用经过实践验证的一套防过拟合的默认参数。
        base = self._get_optimized_params()
        if lgb_params:
            base.update(lgb_params)  # 传入参数覆盖默认
        self.lgb_params = base
        self.model: lgb.Booster | None = None
        self.feature_names: List[str] | None = None

    # ------------------------------------------------------------------ #
    #                         参数与数据相关方法                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_optimized_params() -> Dict[str, Any]:
        """返回一组对过拟合更友好的默认 LightGBM 参数。"""
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 127,  # 控制复杂度
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 100,
            "min_child_weight": 0.001,
            "reg_alpha": 0.1,
            "reg_lambda": 0.3,
            "max_depth": -1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

    # CModelGenerator 接口实现 ---------------------------------------- #

    def create_data_set(self, X: np.ndarray, y: np.ndarray | None = None) -> CLightGBMDataSet:
        """根据给定的特征矩阵/标签构造 LightGBM DataSet。"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        dataset = lgb.Dataset(X, label=y)
        return CLightGBMDataSet(dataset, raw_features=X)

    def create_train_test_set(self, sample_iter) -> Tuple[CLightGBMDataSet, CLightGBMDataSet]:
        """该项目在外部完成数据集的划分，因此此方法不在当前实现范围。"""
        raise NotImplementedError("数据切分逻辑由调用端决定，此接口在当前代码库中无需实现。")

    def train(self, train_set: CLightGBMDataSet, test_set: CLightGBMDataSet) -> None:
        """训练 LightGBM 模型。"""
        valid_sets = [train_set.data, test_set.data]
        valid_names = ["train", "eval"]

        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=20),
        ]

        # 若参数字典中包含 num_iterations / num_boost_round，用于覆盖默认 1000
        n_rounds = self.lgb_params.pop("num_iterations", self.lgb_params.pop("num_boost_round", 1000))

        self.model = lgb.train(
            params=self.lgb_params,
            train_set=train_set.data,
            num_boost_round=n_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, dataset: CLightGBMDataSet) -> List[float]:
        """对给定数据集进行预测。"""
        if self.model is None:
            raise ValueError("模型未训练，不能进行预测。")

        # 推理阶段优先取保存的原始特征矩阵，若不可用则退回到 Dataset 内部数据。
        if dataset.raw_features is not None:
            X = dataset.raw_features
        else:
            X = dataset.data.get_data()
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        # LightGBM 二分类返回 shape (n,), 多分类 shape (n, num_class)
        predictions = np.asarray(predictions)
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            return predictions[:, 1].tolist()
        return predictions.tolist()

    def save_model(self) -> None:
        """将训练好的模型与元数据保存到磁盘。"""
        if self.model is None:
            raise ValueError("模型未训练，无法保存。")

        model_path = self.get_model_path()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 1. 保存 Booster 模型
        self.model.save_model(model_path)

        # 2. 保存元信息，方便后续加载和推理时复现环境
        meta = {
            "model_type": self.model_type,
            "model_tag": self.model_tag,
            "lgb_params": self.lgb_params,
            "num_features": self.model.num_feature(),
            "feature_names": self.feature_names,
            "best_iteration": self.model.best_iteration,
            "best_score": dict(self.model.best_score),
        }
        meta_path = model_path.replace(".model", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def load_model(self) -> int:
        """从磁盘加载模型，返回模型预期的特征维度。"""
        model_path = self.get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 1. 加载 Booster
        self.model = lgb.Booster(model_file=model_path)

        # 2. 尝试同步 meta 信息
        meta_path = model_path.replace(".model", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
                self.lgb_params = meta_info.get("lgb_params", self.lgb_params)
                self.feature_names = meta_info.get("feature_names", None)

        return self.model.num_feature()

    # ------------------------------------------------------------------ #
    #                        业务相关的辅助接口                         #
    # ------------------------------------------------------------------ #
    def get_feature_importance(self) -> Dict[str, float]:
        """返回按特征名称聚合的 gain 重要性。"""
        if self.model is None:
            raise ValueError("模型未训练，无法获取特征重要性。")

        importance = self.model.feature_importance(importance_type="gain")
        names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
        return dict(zip(names, importance)) 