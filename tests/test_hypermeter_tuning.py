"""
Tests for hyperparameter tuning module.

This test suite covers:
- Threshold optimization utilities
- Data preparation functions
- Model building functions
- Model tuning functions (with mocked expensive operations)
- Save/load functionality
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.hyperparameter_tuning import (
    find_best_threshold,
    compute_f1_from_predictions,
    create_sequences_for_tuning,
    build_lstm_with_params,
    build_lstm_classifier_with_params,
    build_lstm_multitask_with_params,
    save_best_params,
    load_best_params,
    tune_linear_regression_threshold,
    tune_random_forest,
    tune_xgboost,
    tune_lstm,
    tune_lstm_classifier,
    tune_lstm_multitask,
    BEST_PARAMS_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_classification_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred_continuous = np.random.randn(n_samples) * 0.5
    return y_true, y_pred_continuous


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y_reg = np.random.randn(n_samples) * 0.02  # Returns
    y_cls = (y_reg > 0).astype(int)  # Direction
    return X[:150], y_reg[:150], y_cls[:150], X[150:], y_reg[150:], y_cls[150:]


@pytest.fixture
def sample_lstm_params():
    """Sample LSTM hyperparameters."""
    return {
        'units1': 64,
        'units2': 32,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
    }


@pytest.fixture
def sample_sequence_data():
    """Generate sample sequence data for LSTM."""
    np.random.seed(42)
    lookback = 10
    n_features = 8
    n_samples = 50

    X_seq = np.random.randn(n_samples, lookback, n_features)
    y_reg = np.random.randn(n_samples) * 0.02
    y_cls = (y_reg > 0).astype(int)

    return X_seq, y_reg, y_cls


# ---------------------------------------------------------------------------
# Test Threshold Optimization
# ---------------------------------------------------------------------------

def test_find_best_threshold_f1(sample_classification_data):
    """Test threshold optimization using F1 score."""
    y_true, y_pred = sample_classification_data

    best_thresh, best_score = find_best_threshold(
        y_true, y_pred, metric="f1", n_quantiles=50
    )

    assert isinstance(best_thresh, float)
    assert isinstance(best_score, float)
    assert 0.0 <= best_score <= 1.0
    assert best_score > 0.0  # Should find some reasonable threshold


def test_find_best_threshold_balanced_acc(sample_classification_data):
    """Test threshold optimization using balanced accuracy."""
    y_true, y_pred = sample_classification_data

    best_thresh, best_score = find_best_threshold(
        y_true, y_pred, metric="balanced_acc", n_quantiles=50
    )

    assert isinstance(best_thresh, float)
    assert isinstance(best_score, float)
    assert 0.0 <= best_score <= 1.0


def test_find_best_threshold_invalid_metric(sample_classification_data):
    """Test that invalid metric raises error."""
    y_true, y_pred = sample_classification_data

    with pytest.raises(ValueError, match="Unknown metric"):
        find_best_threshold(y_true, y_pred, metric="invalid")


def test_find_best_threshold_all_same_class():
    """Test threshold optimization when predictions lead to all same class."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # All same

    best_thresh, best_score = find_best_threshold(y_true, y_pred, n_quantiles=10)

    # Should still return some values
    assert isinstance(best_thresh, float)
    assert isinstance(best_score, float)


def test_compute_f1_from_predictions():
    """Test F1 score computation from predictions."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([-0.1, 0.2, 0.3, -0.05, 0.15])

    # Threshold = 0.0: predictions become [0, 1, 1, 0, 1] -> perfect match
    f1 = compute_f1_from_predictions(y_true, y_pred, threshold=0.0)
    assert f1 == 1.0

    # Different threshold
    f1 = compute_f1_from_predictions(y_true, y_pred, threshold=0.2)
    assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# Test Sequence Creation
# ---------------------------------------------------------------------------

def test_create_sequences_for_tuning():
    """Test sequence creation for LSTM."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    lookback = 10

    X = np.random.randn(n_samples, n_features)
    y_reg = np.random.randn(n_samples)
    y_cls = np.random.randint(0, 2, n_samples)

    X_seq, y_reg_seq, y_cls_seq = create_sequences_for_tuning(
        X, y_reg, y_cls, lookback=lookback
    )

    expected_samples = n_samples - lookback + 1
    assert X_seq.shape == (expected_samples, lookback, n_features)
    assert y_reg_seq.shape == (expected_samples,)
    assert y_cls_seq.shape == (expected_samples,)

    # Check that last sequence uses the correct window
    assert np.allclose(X_seq[-1], X[-lookback:])
    assert y_reg_seq[-1] == y_reg[-1]
    assert y_cls_seq[-1] == y_cls[-1]


def test_create_sequences_edge_case():
    """Test sequence creation with minimal data."""
    lookback = 5
    X = np.ones((5, 3))  # Exactly lookback samples
    y_reg = np.ones(5)
    y_cls = np.zeros(5)

    X_seq, y_reg_seq, y_cls_seq = create_sequences_for_tuning(
        X, y_reg, y_cls, lookback=lookback
    )

    assert X_seq.shape == (1, lookback, 3)
    assert y_reg_seq.shape == (1,)
    assert y_cls_seq.shape == (1,)


# ---------------------------------------------------------------------------
# Test Model Building
# ---------------------------------------------------------------------------

def test_build_lstm_with_params(sample_lstm_params):
    """Test LSTM regressor model building."""
    params = sample_lstm_params.copy()
    params['loss'] = 'mse'

    model = build_lstm_with_params(
        lookback=10,
        n_features=8,
        params=params
    )

    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 5  # Input, LSTM, Dropout, LSTM, Dropout, Dense
    assert model.output_shape == (None, 1)

    # Check optimizer learning rate
    assert abs(model.optimizer.learning_rate.numpy() - params['learning_rate']) < 1e-6

    tf.keras.backend.clear_session()


def test_build_lstm_with_different_loss_functions(sample_lstm_params):
    """Test LSTM building with different loss functions."""
    for loss in ['mse', 'mae', 'huber']:
        params = sample_lstm_params.copy()
        params['loss'] = loss

        model = build_lstm_with_params(10, 8, params)
        assert model is not None
        assert model.loss == loss

        tf.keras.backend.clear_session()


def test_build_lstm_classifier_with_params(sample_lstm_params):
    """Test LSTM classifier model building."""
    model = build_lstm_classifier_with_params(
        lookback=10,
        n_features=8,
        params=sample_lstm_params
    )

    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, 1)

    # Check that it uses sigmoid activation
    assert model.layers[-1].activation.__name__ == 'sigmoid'

    # Check loss is binary crossentropy
    assert model.loss == 'binary_crossentropy'

    tf.keras.backend.clear_session()


def test_build_lstm_multitask_with_params(sample_lstm_params):
    """Test LSTM multitask model building."""
    params = sample_lstm_params.copy()
    params['alpha_return'] = 0.5

    model = build_lstm_multitask_with_params(
        lookback=10,
        n_features=8,
        params=params
    )

    assert isinstance(model, tf.keras.Model)
    assert len(model.outputs) == 2  # Two heads
    assert model.output_names == ['return_out', 'dir_out']

    # Check output shapes
    assert model.outputs[0].shape == (None, 1)  # Return head
    assert model.outputs[1].shape == (None, 1)  # Direction head

    tf.keras.backend.clear_session()


def test_build_lstm_multitask_different_alpha():
    """Test multitask model with different alpha values."""
    params = {
        'units1': 32,
        'units2': 16,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'alpha_return': 0.7,
    }

    model = build_lstm_multitask_with_params(10, 5, params)

    # Check loss weights (approximately, as they're wrapped)
    assert model is not None
    assert len(model.outputs) == 2

    tf.keras.backend.clear_session()


# ---------------------------------------------------------------------------
# Test Save/Load Functions
# ---------------------------------------------------------------------------

def test_save_and_load_best_params(tmp_path):
    """Test saving and loading best parameters."""
    # Create test results
    results = {
        'LinearRegression': {
            'best_threshold': 0.05,
            'best_f1': 0.62,
        },
        'RandomForest': {
            'best_params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
            },
            'best_threshold': 0.02,
            'best_f1': 0.65,
        },
        'LSTM': {
            'best_params': {
                'units1': 64,
                'units2': 32,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'loss': 'mse',
            },
            'best_threshold': 0.0,
            'best_f1': 0.68,
        },
    }

    # Mock BEST_PARAMS_PATH to use temp directory
    temp_file = tmp_path / "best_params.json"

    with patch('src.hyperparameter_tuning.BEST_PARAMS_PATH', temp_file):
        # Save
        save_best_params(results)
        assert temp_file.exists()

        # Load
        loaded = load_best_params()

        assert loaded['LinearRegression']['best_threshold'] == 0.05
        assert loaded['RandomForest']['best_params']['n_estimators'] == 200
        assert loaded['LSTM']['best_params']['units1'] == 64


def test_save_best_params_with_numpy_types(tmp_path):
    """Test saving parameters with numpy types (should be converted)."""
    results = {
        'XGBoost': {
            'best_params': {
                'n_estimators': np.int64(300),
                'learning_rate': np.float64(0.1),
                'max_depth': None,
            },
            'best_threshold': np.float64(0.01),
            'best_f1': np.float64(0.67),
        },
    }

    temp_file = tmp_path / "best_params.json"

    with patch('src.hyperparameter_tuning.BEST_PARAMS_PATH', temp_file):
        save_best_params(results)

        # Load and verify types are native Python
        with open(temp_file, 'r') as f:
            loaded = json.load(f)

        assert isinstance(loaded['XGBoost']['best_params']['n_estimators'], int)
        assert isinstance(loaded['XGBoost']['best_params']['learning_rate'], float)
        assert loaded['XGBoost']['best_params']['max_depth'] is None


def test_save_best_params_skips_none(tmp_path, capsys):
    """Test that models with None params are skipped."""
    results = {
        'FailedModel': {
            'best_params': None,
            'best_f1': 0.0,
        },
        'SuccessModel': {
            'best_params': {'param': 1},
            'best_f1': 0.5,
        },
    }

    temp_file = tmp_path / "best_params.json"

    with patch('src.hyperparameter_tuning.BEST_PARAMS_PATH', temp_file):
        save_best_params(results)

        loaded = load_best_params()

        # Failed model should be skipped
        assert 'FailedModel' not in loaded
        assert 'SuccessModel' in loaded

        # Check warning was printed
        captured = capsys.readouterr()
        assert 'Skipping FailedModel' in captured.out


def test_load_best_params_file_not_found():
    """Test loading when file doesn't exist."""
    fake_path = Path("/nonexistent/path/params.json")

    with patch('src.hyperparameter_tuning.BEST_PARAMS_PATH', fake_path):
        with pytest.raises(FileNotFoundError):
            load_best_params()


# ---------------------------------------------------------------------------
# Test Model Tuning Functions
# ---------------------------------------------------------------------------

def test_tune_linear_regression_threshold(sample_regression_data):
    """Test LinearRegression threshold tuning."""
    X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls = sample_regression_data

    # Use a small grid for faster testing
    with patch('src.hyperparameter_tuning.LR_THRESHOLD_GRID', np.linspace(-0.1, 0.1, 11)):
        results = tune_linear_regression_threshold(
            X_train, y_train_reg, X_val, y_val_reg, y_val_cls
        )

    assert 'best_threshold' in results
    assert 'best_f1' in results
    assert 'search_results' in results

    assert isinstance(results['best_threshold'], float)
    assert 0.0 <= results['best_f1'] <= 1.0
    assert len(results['search_results']) == 11


def test_tune_random_forest_small_grid(sample_regression_data):
    """Test RandomForest tuning with small grid."""
    X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls = sample_regression_data

    # Mock small parameter grid
    small_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
    }

    with patch('src.hyperparameter_tuning.RF_PARAM_GRID', small_grid):
        with patch('src.hyperparameter_tuning.LR_THRESHOLD_GRID', np.linspace(-0.05, 0.05, 5)):
            results = tune_random_forest(
                X_train, y_train_reg, X_val, y_val_reg, y_val_cls, y_train_cls
            )

    assert 'best_params' in results
    assert 'best_threshold' in results
    assert 'best_f1' in results
    assert 'search_results' in results

    assert results['best_params']['n_estimators'] in [50, 100]
    assert results['best_params']['max_depth'] in [5, 10]
    assert 0.0 <= results['best_f1'] <= 1.0


def test_tune_xgboost_small_grid(sample_regression_data):
    """Test XGBoost tuning with small grid."""
    X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls = sample_regression_data

    # Mock small parameter grid
    small_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5],
        'gamma': [0, 0.1],
    }

    with patch('src.hyperparameter_tuning.XGB_PARAM_GRID', small_grid):
        with patch('src.hyperparameter_tuning.LR_THRESHOLD_GRID', np.linspace(-0.05, 0.05, 5)):
            results = tune_xgboost(
                X_train, y_train_reg, X_val, y_val_reg, y_val_cls, y_train_cls
            )

    assert 'best_params' in results
    assert 'best_threshold' in results
    assert 'best_f1' in results

    assert results['best_params']['n_estimators'] in [50, 100]
    assert results['best_params']['learning_rate'] in [0.1, 0.2]


def test_tune_lstm_small_sample(sample_sequence_data):
    """Test LSTM tuning with very small sample size."""
    X_seq, y_reg, y_cls = sample_sequence_data

    # Split into train/val
    split = 40
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_reg_train, y_reg_val = y_reg[:split], y_reg[split:]
    y_cls_val = y_cls[split:]

    # Create a scaler
    scaler = StandardScaler()
    scaler.fit(y_reg_train.reshape(-1, 1))

    # Mock very small grid
    small_grid = {
        'units1': [32],
        'units2': [16],
        'dropout': [0.2],
        'learning_rate': [0.001],
        'batch_size': [16],
        'loss': ['mse'],
    }

    with patch('src.hyperparameter_tuning.LSTM_PARAM_GRID', small_grid):
        results = tune_lstm(
            X_train, y_reg_train,
            X_val, y_reg_val, y_cls_val,
            target_scaler=scaler,
            n_samples=1,
        )

    assert 'best_params' in results
    assert 'best_f1' in results
    assert 'best_threshold' in results

    # Should have attempted at least one configuration
    assert results['best_params'] is not None or len(results['search_results']) > 0

    tf.keras.backend.clear_session()


def test_tune_lstm_classifier_small_sample(sample_sequence_data):
    """Test LSTM classifier tuning with small sample."""
    X_seq, y_reg, y_cls = sample_sequence_data

    split = 40
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_cls_train, y_cls_val = y_cls[:split], y_cls[split:]

    small_grid = {
        'units1': [32],
        'units2': [16],
        'dropout': [0.2],
        'learning_rate': [0.001],
        'batch_size': [16],
    }

    with patch('src.hyperparameter_tuning.LSTM_CLASSIFIER_PARAM_GRID', small_grid):
        results = tune_lstm_classifier(
            X_train, y_cls_train,
            X_val, y_cls_val,
            n_samples=1,
        )

    assert 'best_params' in results
    assert 'best_f1' in results
    assert 'best_threshold' in results

    tf.keras.backend.clear_session()


def test_tune_lstm_multitask_small_sample(sample_sequence_data):
    """Test LSTM multitask tuning with small sample."""
    X_seq, y_reg, y_cls = sample_sequence_data

    split = 40
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_reg_train, y_reg_val = y_reg[:split], y_reg[split:]
    y_cls_train, y_cls_val = y_cls[:split], y_cls[split:]

    scaler = StandardScaler()
    scaler.fit(y_reg_train.reshape(-1, 1))

    small_grid = {
        'units1': [32],
        'units2': [16],
        'dropout': [0.2],
        'learning_rate': [0.001],
        'batch_size': [16],
        'alpha_return': [0.5],
    }

    with patch('src.hyperparameter_tuning.LSTM_MULTITASK_PARAM_GRID', small_grid):
        results = tune_lstm_multitask(
            X_train, y_reg_train, y_cls_train,
            X_val, y_reg_val, y_cls_val,
            target_scaler=scaler,
            n_samples=1,
        )

    assert 'best_params' in results
    assert 'best_f1' in results
    assert 'best_threshold' in results

    tf.keras.backend.clear_session()


def test_tune_lstm_handles_training_error(sample_sequence_data):
    """Test LSTM tuning gracefully handles training errors."""
    X_seq, y_reg, y_cls = sample_sequence_data

    split = 40
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_reg_train, y_reg_val = y_reg[:split], y_reg[split:]
    y_cls_val = y_cls[split:]

    scaler = StandardScaler()
    scaler.fit(y_reg_train.reshape(-1, 1))

    # Mock build function to raise error
    def mock_build_error(*args, **kwargs):
        raise ValueError("Simulated training error")

    with patch('src.hyperparameter_tuning.build_lstm_with_params', side_effect=mock_build_error):
        results = tune_lstm(
            X_train, y_reg_train,
            X_val, y_reg_val, y_cls_val,
            target_scaler=scaler,
            n_samples=2,
        )

    # Should handle errors gracefully
    assert 'best_params' in results
    assert results['best_params'] is None  # No successful trials
    assert results['best_f1'] == -1.0


# ---------------------------------------------------------------------------
# Test Main Tuning Function
# ---------------------------------------------------------------------------

@patch('src.hyperparameter_tuning.prepare_tuning_data')
@patch('src.hyperparameter_tuning.tune_linear_regression_threshold')
@patch('src.hyperparameter_tuning.tune_random_forest')
@patch('src.hyperparameter_tuning.tune_xgboost')
@patch('src.hyperparameter_tuning.tune_lstm')
@patch('src.hyperparameter_tuning.create_sequences_for_tuning')
@patch('src.hyperparameter_tuning.save_best_params')
def test_tune_all_models_regressor_mode(
    mock_save, mock_create_seq, mock_lstm, mock_xgb, mock_rf, mock_lr, mock_data
):
    """Test tune_all_models in regressor mode."""
    # Mock data preparation
    n_samples = 100
    n_features = 10
    mock_data.return_value = (
        np.random.randn(n_samples, n_features),  # X_train
        np.random.randn(n_samples),  # y_train_reg
        np.random.randint(0, 2, n_samples),  # y_train_cls
        np.random.randn(50, n_features),  # X_val
        np.random.randn(50),  # y_val_reg
        np.random.randint(0, 2, 50),  # y_val_cls
        np.random.randn(50, n_features),  # X_test
        np.random.randn(50),  # y_test_reg
        np.random.randint(0, 2, 50),  # y_test_cls
        MinMaxScaler(),  # feature_scaler
        StandardScaler(),  # target_scaler
        np.random.randn(n_samples),  # y_train_reg_raw
        np.random.randn(50),  # y_val_reg_raw
        np.random.randn(50),  # y_test_reg_raw
    )

    # Mock sequence creation
    mock_create_seq.return_value = (
        np.random.randn(40, 10, 10),
        np.random.randn(40),
        np.random.randint(0, 2, 40),
    )

    # Mock tuning functions
    mock_lr.return_value = {'best_threshold': 0.0, 'best_f1': 0.60}
    mock_rf.return_value = {'best_params': {}, 'best_f1': 0.62, 'best_threshold': 0.0}
    mock_xgb.return_value = {'best_params': {}, 'best_f1': 0.63, 'best_threshold': 0.0}
    mock_lstm.return_value = {'best_params': {}, 'best_f1': 0.65, 'best_threshold': 0.0}

    # Import function
    from src.hyperparameter_tuning import tune_all_models

    results = tune_all_models(save_results=True, mode="regressor")

    # Check all models were tuned
    assert 'LinearRegression' in results
    assert 'RandomForest' in results
    assert 'XGBoost' in results
    assert 'LSTM' in results

    # Check save was called
    mock_save.assert_called_once()


@patch('src.hyperparameter_tuning.prepare_tuning_data')
@patch('src.hyperparameter_tuning.tune_linear_regression_threshold')
@patch('src.hyperparameter_tuning.tune_random_forest')
@patch('src.hyperparameter_tuning.tune_xgboost')
@patch('src.hyperparameter_tuning.tune_lstm_classifier')
@patch('src.hyperparameter_tuning.create_sequences_for_tuning')
def test_tune_all_models_classifier_mode(
    mock_create_seq, mock_lstm_cls, mock_xgb, mock_rf, mock_lr, mock_data
):
    """Test tune_all_models in classifier mode."""
    # Mock data
    n_samples = 100
    n_features = 10
    mock_data.return_value = (
        np.random.randn(n_samples, n_features),
        np.random.randn(n_samples),
        np.random.randint(0, 2, n_samples),
        np.random.randn(50, n_features),
        np.random.randn(50),
        np.random.randint(0, 2, 50),
        np.random.randn(50, n_features),
        np.random.randn(50),
        np.random.randint(0, 2, 50),
        MinMaxScaler(),
        StandardScaler(),
        np.random.randn(n_samples),
        np.random.randn(50),
        np.random.randn(50),
    )

    mock_create_seq.return_value = (
        np.random.randn(40, 10, 10),
        np.random.randn(40),
        np.random.randint(0, 2, 40),
    )

    mock_lr.return_value = {'best_threshold': 0.0, 'best_f1': 0.60}
    mock_rf.return_value = {'best_params': {}, 'best_f1': 0.62, 'best_threshold': 0.0}
    mock_xgb.return_value = {'best_params': {}, 'best_f1': 0.63, 'best_threshold': 0.0}
    mock_lstm_cls.return_value = {'best_params': {}, 'best_f1': 0.66, 'best_threshold': 0.5}

    from src.hyperparameter_tuning import tune_all_models

    results = tune_all_models(save_results=False, mode="classifier")

    assert 'LSTMClassifier' in results
    mock_lstm_cls.assert_called_once()


@patch('src.hyperparameter_tuning.prepare_tuning_data')
@patch('src.hyperparameter_tuning.tune_linear_regression_threshold')
@patch('src.hyperparameter_tuning.tune_random_forest')
@patch('src.hyperparameter_tuning.tune_xgboost')
@patch('src.hyperparameter_tuning.tune_lstm_multitask')
@patch('src.hyperparameter_tuning.create_sequences_for_tuning')
def test_tune_all_models_multitask_mode(
    mock_create_seq, mock_lstm_mt, mock_xgb, mock_rf, mock_lr, mock_data
):
    """Test tune_all_models in multitask mode."""
    # Mock data
    n_samples = 100
    n_features = 10
    mock_data.return_value = (
        np.random.randn(n_samples, n_features),
        np.random.randn(n_samples),
        np.random.randint(0, 2, n_samples),
        np.random.randn(50, n_features),
        np.random.randn(50),
        np.random.randint(0, 2, 50),
        np.random.randn(50, n_features),
        np.random.randn(50),
        np.random.randint(0, 2, 50),
        MinMaxScaler(),
        StandardScaler(),
        np.random.randn(n_samples),
        np.random.randn(50),
        np.random.randn(50),
    )

    mock_create_seq.return_value = (
        np.random.randn(40, 10, 10),
        np.random.randn(40),
        np.random.randint(0, 2, 40),
    )

    mock_lr.return_value = {'best_threshold': 0.0, 'best_f1': 0.60}
    mock_rf.return_value = {'best_params': {}, 'best_f1': 0.62, 'best_threshold': 0.0}
    mock_xgb.return_value = {'best_params': {}, 'best_f1': 0.63, 'best_threshold': 0.0}
    mock_lstm_mt.return_value = {'best_params': {}, 'best_f1': 0.67, 'best_threshold': 0.5}

    from src.hyperparameter_tuning import tune_all_models

    results = tune_all_models(save_results=False, mode="multitask")

    assert 'LSTMMultiTask' in results
    mock_lstm_mt.assert_called_once()


# ---------------------------------------------------------------------------
# Test Data Preparation
# ---------------------------------------------------------------------------

@patch('src.data_loader.build_and_save_feature_target_dataset')
@patch('src.data_loader.get_default_processed_csv_path')
def test_prepare_tuning_data(mock_get_path, mock_build_dataset, tmp_path):
    """Test data preparation for tuning."""
    import pandas as pd
    from datetime import datetime, timedelta

    # Create mock CSV file
    csv_path = tmp_path / "processed.csv"

    # Create fake data covering the date ranges
    dates = pd.date_range(start='2018-01-01', end='2024-12-31', freq='D')
    n_samples = len(dates)

    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'next_day_return': np.random.randn(n_samples) * 0.02,
        'next_day_direction': np.random.randint(0, 2, n_samples),
    }, index=dates)

    df.to_csv(csv_path)

    # Mock path to return our temp file
    mock_get_path.return_value = csv_path

    from src.hyperparameter_tuning import prepare_tuning_data

    result = prepare_tuning_data()

    # Unpack results
    (X_train, y_train_reg, y_train_cls,
     X_val, y_val_reg, y_val_cls,
     X_test, y_test_reg, y_test_cls,
     feature_scaler, target_scaler,
     y_train_reg_raw, y_val_reg_raw, y_test_reg_raw) = result

    # Check shapes
    assert X_train.shape[1] == 3  # 3 features
    assert X_val.shape[1] == 3
    assert X_test.shape[1] == 3

    # Check that splits are non-empty
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # Check that targets match features
    assert len(X_train) == len(y_train_reg) == len(y_train_cls)
    assert len(X_val) == len(y_val_reg) == len(y_val_cls)
    assert len(X_test) == len(y_test_reg) == len(y_test_cls)

    # Check that scalers were fit
    assert feature_scaler is not None
    assert target_scaler is not None

    # Check that features are scaled to [0, 1]
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0

    # Check that raw and scaled targets exist
    assert len(y_train_reg_raw) == len(y_train_reg)

    # Dataset shouldn't be built since file exists
    mock_build_dataset.assert_not_called()


@patch('src.data_loader.build_and_save_feature_target_dataset')
@patch('src.data_loader.get_default_processed_csv_path')
def test_prepare_tuning_data_builds_if_missing(mock_get_path, mock_build_dataset, tmp_path):
    """Test that data preparation builds dataset if CSV doesn't exist."""
    import pandas as pd

    # First return non-existent path
    csv_path = tmp_path / "nonexistent.csv"
    mock_get_path.return_value = csv_path

    # After build is called, create the file
    def create_csv_after_build():
        dates = pd.date_range(start='2018-01-01', end='2024-12-31', freq='D')
        n_samples = len(dates)
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'next_day_return': np.random.randn(n_samples) * 0.02,
            'next_day_direction': np.random.randint(0, 2, n_samples),
        }, index=dates)
        df.to_csv(csv_path)

    mock_build_dataset.side_effect = create_csv_after_build

    from src.hyperparameter_tuning import prepare_tuning_data

    result = prepare_tuning_data()

    # Should have called build function
    mock_build_dataset.assert_called_once()

    # Should still return valid data (13 arrays + 1 scaler object)
    assert len(result) == 14  # Correct count of returned values


# ---------------------------------------------------------------------------
# Test Edge Cases
# ---------------------------------------------------------------------------

def test_compute_f1_all_zeros():
    """Test F1 computation when all predictions are 0."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([-1.0, -0.5, -0.2, -0.1])  # All negative -> all 0

    f1 = compute_f1_from_predictions(y_true, y_pred, threshold=0.0)

    # Should handle zero_division gracefully
    assert isinstance(f1, float)
    assert f1 >= 0.0


def test_create_sequences_different_lookback():
    """Test sequence creation with different lookback values."""
    X = np.random.randn(50, 5)
    y_reg = np.random.randn(50)
    y_cls = np.random.randint(0, 2, 50)

    for lookback in [5, 10, 20]:
        X_seq, y_reg_seq, y_cls_seq = create_sequences_for_tuning(
            X, y_reg, y_cls, lookback=lookback
        )

        expected_samples = len(X) - lookback + 1
        assert X_seq.shape[0] == expected_samples
        assert X_seq.shape[1] == lookback
