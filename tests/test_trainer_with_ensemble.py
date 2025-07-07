import numpy as np
import pandas as pd
import pathlib
import tempfile

import pytest

from ModelStrategy.ChanMLTrainer import ChanMLTrainer


def test_trainer_with_ensemble(monkeypatch, tmp_path):
    """Run a minimal training pipeline and ensure ensemble is produced."""

    # Synthetic minimal OHLCV dataset
    n = 120
    df = pd.DataFrame({
        'open': np.random.rand(n),
        'high': np.random.rand(n),
        'low': np.random.rand(n),
        'close': np.random.rand(n),
        'volume': np.random.rand(n),
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='H')
    })
    df['binary_direction'] = (df['close'] > df['open']).astype(int)

    # Temporary directories
    model_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    model_dir.mkdir()
    log_dir.mkdir()

    trainer = ChanMLTrainer(config={
        'data_dir': '',
        'model_dir': str(model_dir),
        'log_dir': str(log_dir),
        'dataset_name': 'dummy',
        'target_column': 'binary_direction',
        'models': ['xgb', 'lgb', 'cat'],
        'random_state': 42
    })

    # Stub DatasetManager methods
    def _load_train(self, name):
        return df.iloc[:100].copy(), {}
    def _load_val(self, name):
        return df.iloc[100:].copy(), {}

    monkeypatch.setattr(trainer.dataset_manager, 'load_training_dataset', _load_train.__get__(trainer.dataset_manager))
    monkeypatch.setattr(trainer.dataset_manager, 'load_validation_dataset', _load_val.__get__(trainer.dataset_manager))

    # Replace heavy feature calculator with pass-through version
    class _DummyFC:
        def calculate_all_features(self, data):
            return data
        def create_labels(self, data):
            return data
        def get_feature_names(self, data):
            return ['open', 'high', 'low', 'close', 'volume']
    trainer.feature_calculator = _DummyFC()

    results = trainer.run_full_training_pipeline()

    assert 'ensemble' in results, "Ensemble result missing"
    assert results['ensemble']['val_metrics']['auc'] >= 0.0 