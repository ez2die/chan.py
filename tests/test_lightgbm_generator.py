import importlib.util
import pathlib
import sys
import typing
import types

# Python <3.11 兼容：为 typing 增加 Self 定义，避免旧版本解释器导入失败。
if not hasattr(typing, 'Self'):
    from typing import TypeVar
    typing.Self = TypeVar('Self')  # type: ignore

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent

# -------------------------------------------------------------------------
# 动态加载 ModelStrategy.ModelGenerator 作为独立模块，避免执行 ModelStrategy/__init__.py
# -------------------------------------------------------------------------

model_generator_path = ROOT / "ModelStrategy" / "ModelGenerator.py"
spec_gen = importlib.util.spec_from_file_location("ModelStrategy.ModelGenerator", model_generator_path)
model_generator_mod = importlib.util.module_from_spec(spec_gen)
sys.modules[spec_gen.name] = model_generator_mod  # type: ignore
spec_gen.loader.exec_module(model_generator_mod)  # type: ignore

# 创建一个最小化的"ModelStrategy"包对象，并注册到 sys.modules，
# 仅包含 ModelGenerator 子模块即可满足依赖。
minimal_pkg = types.ModuleType("ModelStrategy")
minimal_pkg.ModelGenerator = model_generator_mod  # type: ignore
sys.modules["ModelStrategy"] = minimal_pkg

# -------------------------------------------------------------------------
# 现在再动态加载 LightGBMModelGenerator
# -------------------------------------------------------------------------

lgb_gen_path = ROOT / "ModelStrategy" / "models" / "LightGBMModelGenerator.py"
spec_lgb = importlib.util.spec_from_file_location("LightGBMModelGenerator", lgb_gen_path)
lgb_mod = importlib.util.module_from_spec(spec_lgb)
sys.modules[spec_lgb.name] = lgb_mod  # type: ignore
spec_lgb.loader.exec_module(lgb_mod)  # type: ignore

CLightGBMModelGenerator = lgb_mod.CLightGBMModelGenerator

def test_lightgbm_basic_train_predict():
    """Basic sanity check for the LightGBM model generator."""
    # Generate synthetic binary classification data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 20))
    y = (rng.random(200) > 0.5).astype(int)

    # Split into train / validation
    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    gen = CLightGBMModelGenerator(model_tag="unit_test")

    train_set = gen.create_data_set(X_train, y_train)
    val_set = gen.create_data_set(X_val, y_val)

    # Train should not raise errors
    gen.train(train_set, val_set)

    # Predict on validation set
    preds = gen.predict(val_set)

    assert len(preds) == len(y_val), "Prediction length mismatch"

    # AUC should be within valid range [0,1]
    auc = roc_auc_score(y_val, preds)
    assert 0.0 <= auc <= 1.0 