"""Unit tests for SepsisXGBModel: output range, binary predict, inference speed."""

from __future__ import annotations

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model():
    """Train a small XGBoost model and return (model, X_test)."""
    from ml.models.xgboost_model import SepsisXGBModel

    model = SepsisXGBModel(
        params={
            "n_estimators": 50,   # small for fast tests
            "max_depth": 3,
        }
    )
    X = np.random.rand(200, 10).astype(np.float32)
    y = np.array([0] * 170 + [1] * 30)

    model.fit(X[:160], y[:160], X[160:180], y[160:180])
    return model, X[180:]


# ---------------------------------------------------------------------------
# TestSepsisXGBModel
# ---------------------------------------------------------------------------

class TestSepsisXGBModel:
    """Tests for SepsisXGBModel."""

    def test_predict_proba_shape(self, trained_model):
        """predict_proba must return (n, 2) array."""
        model, X_test = trained_model
        proba = model.predict_proba(X_test)
        assert proba.ndim == 2
        assert proba.shape[1] == 2

    def test_predict_proba_range(self, trained_model):
        """All probabilities must be in [0, 1]."""
        model, X_test = trained_model
        proba = model.predict_proba(X_test)
        assert (proba >= 0).all(), "Probabilities contain values < 0"
        assert (proba <= 1).all(), "Probabilities contain values > 1"

    def test_predict_proba_sums_to_one(self, trained_model):
        """Rows of predict_proba must sum to 1 (within tolerance)."""
        model, X_test = trained_model
        proba = model.predict_proba(X_test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_binary(self, trained_model):
        """predict() must return only {0, 1}."""
        model, X_test = trained_model
        preds = model.predict(X_test)
        assert set(preds.tolist()).issubset({0, 1}), f"Unexpected values: {set(preds.tolist())}"

    def test_predict_returns_ndarray(self, trained_model):
        model, X_test = trained_model
        preds = model.predict(X_test)
        assert isinstance(preds, np.ndarray)

    def test_inference_speed(self, trained_model):
        """Single-row inference must finish in < 200 ms."""
        model, X_test = trained_model
        start = time.time()
        model.predict_proba(X_test[:1])
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 200, f"Inference too slow: {elapsed_ms:.1f} ms"

    def test_model_not_none_after_fit(self, trained_model):
        """Internal XGBClassifier must be set after fit."""
        model, _ = trained_model
        assert model.model is not None

    def test_untrained_model_raises(self):
        """predict_proba on an untrained model must raise ValueError."""
        from ml.models.xgboost_model import SepsisXGBModel

        model = SepsisXGBModel()
        X = np.random.rand(5, 10)
        with pytest.raises(ValueError):
            model.predict_proba(X)

    def test_save_and_load(self, trained_model, tmp_path):
        """save() then load() must return a working model."""
        from ml.models.xgboost_model import SepsisXGBModel

        model, X_test = trained_model
        path = str(tmp_path / "model.joblib")
        model.save(path)

        loaded = SepsisXGBModel.load(path)
        proba_original = model.predict_proba(X_test)
        proba_loaded = loaded.predict_proba(X_test)
        np.testing.assert_allclose(proba_original, proba_loaded, atol=1e-5)
