import pathlib

import numpy as np
import pytest

from finance_lstm.models import lstm


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _make_small_sequences(
    n_train: int = 10,
    n_test: int = 4,
    lookback: int = 5,
    n_features: int = 3,
):
    """
    Create small synthetic LSTM-ready sequences for testing.
    """
    X_train = np.random.randn(n_train, lookback, n_features).astype("float32")
    y_train_reg = np.random.randn(n_train).astype("float32")
    y_train_cls = np.random.randint(0, 2, size=n_train).astype("int64")

    X_test = np.random.randn(n_test, lookback, n_features).astype("float32")
    y_test_reg = np.random.randn(n_test).astype("float32")
    y_test_cls = np.random.randint(0, 2, size=n_test).astype("int64")

    return X_train, y_train_reg, y_train_cls, X_test, y_test_reg, y_test_cls


# --------------------------------------------------------------------------------------
# load_lstm_data
# --------------------------------------------------------------------------------------


def test_load_lstm_data_reads_arrays_from_lstm_dir(monkeypatch, tmp_path: pathlib.Path):
    """
    load_lstm_data should load the six expected .npy files from LSTM_DIR
    and cast X/y_reg to float32.
    """
    # Patch LSTM_DIR to our temp directory
    monkeypatch.setattr(lstm, "LSTM_DIR", tmp_path)

    # Create minimal synthetic arrays
    X_train, y_train_reg, y_train_cls, X_test, y_test_reg, y_test_cls = (
        _make_small_sequences(n_train=6, n_test=3, lookback=4, n_features=2)
    )

    np.save(tmp_path / "X_train_seq.npy", X_train)
    np.save(tmp_path / "y_train_reg_seq.npy", y_train_reg)
    np.save(tmp_path / "y_train_cls_seq.npy", y_train_cls)
    np.save(tmp_path / "X_test_seq.npy", X_test)
    np.save(tmp_path / "y_test_reg_seq.npy", y_test_reg)
    np.save(tmp_path / "y_test_cls_seq.npy", y_test_cls)

    (
        X_train_loaded,
        y_train_reg_loaded,
        y_train_cls_loaded,
        X_test_loaded,
        y_test_reg_loaded,
        y_test_cls_loaded,
    ) = lstm.load_lstm_data()

    assert X_train_loaded.shape == X_train.shape
    assert X_test_loaded.shape == X_test.shape
    assert y_train_reg_loaded.shape == y_train_reg.shape
    assert y_test_reg_loaded.shape == y_test_reg.shape
    assert np.array_equal(y_train_cls_loaded, y_train_cls)
    assert np.array_equal(y_test_cls_loaded, y_test_cls)

    # Types: X_* and y_*_reg should be float32
    assert X_train_loaded.dtype == np.float32
    assert X_test_loaded.dtype == np.float32
    assert y_train_reg_loaded.dtype == np.float32
    assert y_test_reg_loaded.dtype == np.float32


# --------------------------------------------------------------------------------------
# build_lstm_model
# --------------------------------------------------------------------------------------


def test_build_lstm_model_respects_config(monkeypatch):
    """
    build_lstm_model should honor config.LSTM_CONFIG for units/dropout/lr
    and create a model with the right input shape and output size.
    """
    # Backup and patch config
    old_cfg = dict(lstm.config.LSTM_CONFIG)
    try:
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "units1", 4)
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "units2", 2)
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "dropout", 0.1)
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "learning_rate", 1e-3)

        lookback = 7
        n_features = 3
        model = lstm.build_lstm_model(lookback=lookback, n_features=n_features)

        # Input shape: (None, lookback, n_features)
        assert model.input_shape == (None, lookback, n_features)

        # LSTM layers should have units [4, 2]
        from tensorflow.keras.layers import LSTM as LSTMLayer, Dense as DenseLayer

        lstm_layers = [layer for layer in model.layers if isinstance(layer, LSTMLayer)]
        assert [layer.units for layer in lstm_layers] == [4, 2]

        # Final layer should be Dense(1)
        dense_layers = [
            layer for layer in model.layers if isinstance(layer, DenseLayer)
        ]
        assert dense_layers[-1].units == 1
    finally:
        # Restore config
        lstm.config.LSTM_CONFIG.clear()
        lstm.config.LSTM_CONFIG.update(old_cfg)


# --------------------------------------------------------------------------------------
# train_lstm_model (run only 1 epoch on tiny data)
# --------------------------------------------------------------------------------------


def test_train_lstm_model_runs_one_epoch(monkeypatch, tmp_path: pathlib.Path):
    """
    train_lstm_model should call model.fit without error and create
    a checkpoint file in LSTM_DIR.
    """
    # Patch LSTM_DIR to temp dir
    monkeypatch.setattr(lstm, "LSTM_DIR", tmp_path)

    # Patch config for a tiny, fast training run
    old_cfg = dict(lstm.config.LSTM_CONFIG)
    try:
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "units1", 2)
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "units2", 2)
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "dropout", 0.0)
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "batch_size", 4)
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "epochs", 1)
        monkeypatch.setitem(lstm.config.LSTM_CONFIG, "learning_rate", 1e-3)

        # Tiny synthetic data
        lookback = 4
        n_features = 2
        X_train = np.random.randn(8, lookback, n_features).astype("float32")
        y_train_reg = np.random.randn(8).astype("float32")

        model = lstm.build_lstm_model(lookback=lookback, n_features=n_features)

        history = lstm.train_lstm_model(model, X_train, y_train_reg, val_split=0.25)

        # We don't care about actual metrics, just that fit ran and
        # a checkpoint was created.
        ckpt_path = tmp_path / "lstm_model_best.keras"
        assert ckpt_path.exists(), "Expected checkpoint file was not created."
        assert hasattr(history, "history")
    finally:
        lstm.config.LSTM_CONFIG.clear()
        lstm.config.LSTM_CONFIG.update(old_cfg)


# --------------------------------------------------------------------------------------
# evaluate_and_save_predictions
# --------------------------------------------------------------------------------------


def test_evaluate_and_save_predictions_saves_npy(monkeypatch, tmp_path: pathlib.Path):
    """
    evaluate_and_save_predictions should:
      - call model.evaluate and model.predict
      - save y_pred_reg_lstm.npy and y_pred_dir_lstm.npy under LSTM_DIR
      - return (test_loss, test_mae) as floats.
    """
    monkeypatch.setattr(lstm, "LSTM_DIR", tmp_path)

    # Simple deterministic "model"
    class DummyModel:
        def __init__(self):
            self.evaluate_called = False
            self.predict_called = False

        def evaluate(self, X, y, verbose=1):
            self.evaluate_called = True
            # Return fixed loss/mae to check passthrough
            return 0.25, 0.5

        def predict(self, X, verbose=0):
            self.predict_called = True
            # 3 samples, arbitrary values
            return np.array([[0.1], [-0.2], [0.0]], dtype="float32")

    dummy = DummyModel()

    X_test = np.zeros((3, 4, 2), dtype="float32")
    y_test_reg = np.array([0.05, -0.1, 0.0], dtype="float32")
    y_test_cls = np.array([1, 0, 0], dtype="int64")

    test_loss, test_mae = lstm.evaluate_and_save_predictions(
        dummy, X_test, y_test_reg, y_test_cls
    )

    assert dummy.evaluate_called
    assert dummy.predict_called

    # Returned metrics match dummy values
    assert test_loss == pytest.approx(0.25)
    assert test_mae == pytest.approx(0.5)

    # Files saved
    pred_reg_path = tmp_path / "y_pred_reg_lstm.npy"
    pred_dir_path = tmp_path / "y_pred_dir_lstm.npy"
    assert pred_reg_path.exists()
    assert pred_dir_path.exists()

    y_pred_reg = np.load(pred_reg_path)
    y_pred_dir = np.load(pred_dir_path)

    # Predictions exactly what DummyModel produced
    np.testing.assert_array_equal(
        y_pred_reg, np.array([0.1, -0.2, 0.0], dtype="float32")
    )
    # Direction from sign (>0 -> 1, else 0)
    np.testing.assert_array_equal(y_pred_dir, np.array([1, 0, 0], dtype="int64"))


# --------------------------------------------------------------------------------------
# train_and_evaluate_lstm (orchestration)
# --------------------------------------------------------------------------------------


def test_train_and_evaluate_lstm_orchestrates(monkeypatch, tmp_path: pathlib.Path):
    """
    train_and_evaluate_lstm should:
      - call load_lstm_data
      - call build_lstm_model with correct shapes
      - call train_lstm_model
      - save final model under LSTM_DIR
      - call evaluate_and_save_predictions and return its metrics.
    """
    monkeypatch.setattr(lstm, "LSTM_DIR", tmp_path)

    # 1) Stub load_lstm_data to return small deterministic arrays
    X_train, y_train_reg, y_train_cls, X_test, y_test_reg, y_test_cls = (
        _make_small_sequences(n_train=6, n_test=3, lookback=4, n_features=2)
    )
    load_called = {}

    def fake_load():
        load_called["called"] = True
        return X_train, y_train_reg, y_train_cls, X_test, y_test_reg, y_test_cls

    monkeypatch.setattr(lstm, "load_lstm_data", fake_load)

    # 2) Dummy model that can be "saved"
    class DummyModel:
        def __init__(self):
            self.saved_paths = []

        def save(self, path: str):
            self.saved_paths.append(path)

    dummy_model = DummyModel()
    build_called = {}

    def fake_build(lookback: int, n_features: int):
        build_called["args"] = (lookback, n_features)
        return dummy_model

    monkeypatch.setattr(lstm, "build_lstm_model", fake_build)

    # 3) Stub train_lstm_model so we don't run real Keras training
    train_called = {}

    class MockHistory:
        """Mock Keras History object."""

        def __init__(self):
            self.history = {
                "loss": [0.5, 0.4, 0.3],
                "val_loss": [0.6, 0.5, 0.4],
                "mae": [0.3, 0.25, 0.2],
                "val_mae": [0.35, 0.3, 0.25],
            }

    def fake_train(model, X, y, val_split=0.2):
        train_called["called"] = True
        # Ensure we received the dummy model and the synthetic data
        assert model is dummy_model
        assert X.shape == X_train.shape
        assert y.shape == y_train_reg.shape
        return MockHistory()

    monkeypatch.setattr(lstm, "train_lstm_model", fake_train)

    # 4) Stub evaluate_and_save_predictions to return known metrics
    eval_called = {}

    def fake_eval(model, X, y_reg, y_cls):
        eval_called["args"] = (X.shape, y_reg.shape, y_cls.shape)
        assert model is dummy_model
        return 0.1234, 0.5678

    monkeypatch.setattr(lstm, "evaluate_and_save_predictions", fake_eval)

    # Call the orchestrator
    test_mse, test_mae, history = lstm.train_and_evaluate_lstm()

    # Assertions
    assert load_called.get("called", False)
    assert train_called.get("called", False)
    assert "args" in build_called

    # Model was built with the correct shapes inferred from X_train
    lookback_expected = X_train.shape[1]
    n_features_expected = X_train.shape[2]
    assert build_called["args"] == (lookback_expected, n_features_expected)

    # Final model path used in save
    final_path = tmp_path / "lstm_model_final.keras"
    assert str(final_path) in dummy_model.saved_paths

    # evaluate_and_save_predictions return values are passed through
    assert test_mse == pytest.approx(0.1234)
    assert test_mae == pytest.approx(0.5678)

    # History object is returned (mocked history from fake_train)
    assert history is not None

    # Shapes passed to fake_eval are those of X_test and y_test_*
    X_shape, y_reg_shape, y_cls_shape = eval_called["args"]
    assert X_shape == X_test.shape
    assert y_reg_shape == y_test_reg.shape
    assert y_cls_shape == y_test_cls.shape
