import numpy as np

from ModelStrategy.EnsemblePredictor import ChanEnsemblePredictor


class _MockModel:
    """简单的模拟模型：predict 返回固定数组或线性函数。"""

    def __init__(self, bias: float = 0.0):
        self.bias = bias

    def predict(self, X):  # noqa: D401  # 简单模拟
        n = len(X)
        # 简化：用第一列 + bias 经过 sigmoid
        raw = X[:, 0] + self.bias
        return 1 / (1 + np.exp(-raw))


def test_weighted_average():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = (rng.random(100) > 0.5).astype(int)

    m1 = _MockModel(bias=0.0)
    m2 = _MockModel(bias=1.0)

    ens = ChanEnsemblePredictor(method="weighted_average")
    ens.add_model("m1", m1, weight=0.5)
    ens.add_model("m2", m2, weight=0.5)

    pred, meta = ens.predict(X)
    assert pred.shape[0] == X.shape[0]
    assert meta["method"] == "weighted_average"

    # 权重优化不报错且归一化
    pred_dict = {"m1": m1.predict(X), "m2": m2.predict(X)}
    new_w = ens.optimize_weights(pred_dict, y)
    assert abs(sum(new_w.values()) - 1) < 1e-6


def test_stacking():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    y = (rng.random(80) > 0.5).astype(int)

    m1 = _MockModel(bias=-0.5)
    m2 = _MockModel(bias=0.7)

    ens = ChanEnsemblePredictor(method="stacking")
    ens.add_model("m1", m1)
    ens.add_model("m2", m2)

    X_meta = np.column_stack([m1.predict(X), m2.predict(X)])
    ens.fit_meta_model(X_meta, y)

    pred, meta = ens.predict(X)
    assert pred.shape[0] == 80
    assert meta["method"] == "stacking" 