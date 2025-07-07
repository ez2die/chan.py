import numpy as np

from ModelStrategy.models.CatBoostModelGenerator import CCatBoostModelGenerator


def test_catboost_basic(tmp_path):
    """Basic sanity check for CatBoostModelGenerator."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 10))
    y = (rng.random(120) > 0.5).astype(int)

    X_train, y_train = X[:100], y[:100]
    X_val, y_val = X[100:], y[100:]

    gen = CCatBoostModelGenerator(model_tag="unit_test_cat")
    train_set = gen.create_data_set(X_train, y_train)
    val_set = gen.create_data_set(X_val, y_val)

    # Train and predict
    gen.train(train_set, val_set)
    preds = gen.predict(val_set)
    assert len(preds) == len(y_val)

    # Save / load round-trip
    gen.save_model()
    feature_cnt = gen.load_model()
    assert feature_cnt > 0 