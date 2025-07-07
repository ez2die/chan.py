from __future__ import annotations

"""Chan.py – Hyper-parameter optimization utilities (Phase-2.3.1).

This module wraps Optuna to search for better parameters for tree-based models
currently supported in chan.py.  The focus is on LightGBM and CatBoost because
XGBoost already achieves near-optimal performance given the current feature
set, and also because they are typically the bottlenecks.

Typical usage
-------------
>>> optimizer = ChanHyperparameterOptimizer(n_trials=50, cv_folds=3)
>>> best_lgb_params = optimizer.optimize_lightgbm(X_train, y_train)
>>> best_cb_params  = optimizer.optimize_catboost(X_train, y_train)

The returned dictionaries can be passed to the respective *ModelGenerator*
classes (CLightGBMModelGenerator, CCatBoostModelGenerator).
"""

from typing import Dict, Any, Tuple

import numpy as np

# Third-party
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# LightGBM / CatBoost are optional – import lazily inside methods to keep the
# import cost low for users that may not have GPU versions installed.

__all__ = [
    "ChanHyperparameterOptimizer",
]


class ChanHyperparameterOptimizer:
    """Simple Optuna-based hyper-parameter search helper."""

    def __init__(self, *, n_trials: int = 50, cv_folds: int = 3, n_jobs: int | None = None, cv_type: str = "stratified", purge_gap: int = 24, random_state: int = 42):
        """Parameters
        ----------
        n_trials : int
            Optuna trials per model.
        cv_folds : int
            Stratified KFold splits for inner CV.
        n_jobs : int | None
            Parallel jobs.  ``None`` 或 ``-1`` 表示使用所有 CPU；默认为 1（与 Optuna 默认相同）。
        cv_type : str
            Type of cross-validation. Must be 'stratified' or 'time'.
        purge_gap : int
            Gap between training and validation indices for purged KFold.
        random_state : int
            RNG seed for reproducibility.
        """

        import os

        self.n_trials = n_trials
        self.cv_folds = cv_folds
        # 统一处理 -1 / None → cpu_count
        cpu_cnt = os.cpu_count() or 1
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = cpu_cnt
        else:
            self.n_jobs = max(1, n_jobs)
        self.cv_type = cv_type.lower()
        assert self.cv_type in {"stratified", "time"}, "cv_type must be 'stratified' or 'time'"
        self.purge_gap = purge_gap
        self.random_state = random_state
        self.best_params: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helper – PurgedKFold implementation
    # ------------------------------------------------------------------
    class _PurgedKFold:
        """Simple implementation of Purged K-Fold for time series.

        After splitting by TimeSeriesSplit, it removes the last ``purge_gap``
        samples from the training indices so they do not overlap with the
        validation block whose first index is ``val_start``.
        """

        def __init__(self, n_splits: int, purge_gap: int):
            from sklearn.model_selection import TimeSeriesSplit

            self.base = TimeSeriesSplit(n_splits=n_splits)
            self.purge_gap = purge_gap

        def split(self, X, y=None, groups=None):  # noqa: D401
            for train_idx, val_idx in self.base.split(X, y, groups):
                # 保证训练集尾部与验证集首部间隔 purge_gap
                cutoff = val_idx[0] - self.purge_gap
                train_idx = train_idx[train_idx < cutoff]
                yield train_idx, val_idx

    # ------------------------------------------------------------------
    def _get_cv(self, X, y):
        """Return an iterator of (train_idx, val_idx) according to settings."""
        if self.cv_type == "time":
            return self._PurgedKFold(self.cv_folds, self.purge_gap).split(X)
        else:
            # 使用真实标签进行分层，以确保各折类别分布一致
            return StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            ).split(X, y)

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Bayesian optimise LightGBM parameters using Optuna.

        Parameters
        ----------
        X, y : np.ndarray
            Training features and labels.  Shape (n_samples, n_features).

        Returns
        -------
        Dict[str, Any]
            The best set of parameters discovered.
        """

        import lightgbm as lgb  # local import

        def _objective(trial: optuna.trial.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                # search space
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "max_depth": trial.suggest_int("max_depth", -1, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.8),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.8),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 100, 400),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "random_state": self.random_state,
                "verbosity": -1,
                "n_jobs": self.n_jobs,
            }

            # 迭代次数（num_boost_round）属于 train 调用参数而非 params，本处单独采样
            n_rounds = trial.suggest_int("num_iterations", 300, 600)

            cv_scores = []

            for train_idx, valid_idx in self._get_cv(X, y):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                train_set = lgb.Dataset(X_train, label=y_train)
                valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

                booster = lgb.train(
                    params,
                    train_set,
                    num_boost_round=n_rounds,
                    valid_sets=[valid_set],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(0)],
                )

                preds_val = booster.predict(X_valid, num_iteration=booster.best_iteration)
                preds_train = booster.predict(X_train, num_iteration=booster.best_iteration)
                val_auc = roc_auc_score(y_valid, preds_val)
                train_auc = roc_auc_score(y_train, preds_train)

                # 过拟合惩罚：若训练/验证差距超过 0.3，则降低分数 0.2
                if train_auc - val_auc > 0.3:
                    val_auc -= 0.2

                cv_scores.append(val_auc)

            return float(np.mean(cv_scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(_objective, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=False)

        # study.best_params 不包含我们额外传入的 n_rounds；需要手动添加
        best = study.best_params.copy()
        if "num_iterations" not in best:
            best["num_iterations"] = int(study.best_trial.params["num_iterations"])

        self.best_params["lightgbm"] = best
        return best

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------
    def optimize_catboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimise CatBoost hyper-parameters."""

        import catboost as cb  # local import

        def _objective(trial: optuna.trial.Trial) -> float:
            params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                # iterations is part of search space (catboost benefits from longer training)
                "iterations": trial.suggest_int("iterations", 1500, 4000),
                # search space
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_seed": self.random_state,
                "verbose": 0,
                "early_stopping_rounds": 200,
                "thread_count": self.n_jobs,
            }

            cv_scores = []

            for train_idx, valid_idx in self._get_cv(X, y):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                pool_train = cb.Pool(X_train, y_train)
                pool_valid = cb.Pool(X_valid, y_valid)

                model = cb.CatBoostClassifier(**params)
                model.fit(pool_train, eval_set=pool_valid, use_best_model=True, verbose=False)

                preds = model.predict_proba(pool_valid)[:, 1]
                cv_scores.append(roc_auc_score(y_valid, preds))

            return float(np.mean(cv_scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(_objective, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=False)

        self.best_params["catboost"] = study.best_params
        return study.best_params

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Bayesian optimise XGBoost hyper-parameters using Optuna.

        Parameters
        ----------
        X, y : np.ndarray
            Training features and labels. Shape (n_samples, n_features).

        Returns
        -------
        Dict[str, Any]
            The best set of parameters discovered.
        """

        import xgboost as xgb  # local import

        def _objective(trial: optuna.trial.Trial) -> float:
            actual_jobs = max(1, min(self.n_jobs, 8))  # macOS OpenMP often segfaults with too many threads
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                # search space
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "n_jobs": actual_jobs,
                "random_state": self.random_state,
                "tree_method": "hist",
            }

            n_estimators = trial.suggest_int("n_estimators", 300, 600)

            cv_scores = []
            for train_idx, valid_idx in self._get_cv(X, y):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                model = xgb.XGBClassifier(**params, n_estimators=n_estimators, verbosity=0)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=50,
                    verbose=False,
                )

                preds = model.predict_proba(X_valid)[:, 1]
                cv_scores.append(roc_auc_score(y_valid, preds))

            # 返回均值以评估整体表现
            return float(np.mean(cv_scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(_objective, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=False)

        best = study.best_params.copy()
        if "n_estimators" not in best:
            best["n_estimators"] = int(study.best_trial.params["n_estimators"])

        self.best_params["xgboost"] = best
        return best


# ---------------------------------------------------------------------------
# CLI helper (optional) – allows module to be executed directly
# ---------------------------------------------------------------------------

def _cli():  # pragma: no cover
    import argparse, json, sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Chan.py hyper-parameter optimiser")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials per model")
    parser.add_argument("--cv_folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--output", type=str, default="models/hpo_results.json", help="Where to save best params JSON")
    args = parser.parse_args()

    # NOTE: Minimal example – user expected to supply prepared numpy arrays.
    print("This CLI is only a stub. Import the class and feed data via Python API.")
    sys.exit(0)


if __name__ == "__main__":
    _cli() 