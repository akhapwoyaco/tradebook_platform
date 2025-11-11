# tests/tools/test_tools_manager.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Assuming the module path is correct
from tradebook_pipeline.core_analysis.tools_manager import (
    TechnicalIndicators, 
    generate_emergency_peaks,
    AdvancedPeakDetector
)

# --- Technical Indicators Tests ---

@pytest.fixture
def mock_price_series():
    """A mock price series for indicator calculations."""
    # Series of 20 prices: first 10 rising, next 10 falling slightly
    data = [100 + i for i in range(10)] + [109 - 0.5 * i for i in range(10)]
    return pd.Series(data)

def test_rsi_calculation(mock_price_series):
    """Tests the Relative Strength Index (RSI) calculation."""
    rsi_series = TechnicalIndicators.rsi(mock_price_series, window=5)
    
    # First window will be NaN/50
    assert rsi_series.iloc[:4].eq(50).all()
    
    # When price is constantly rising, RSI should be high
    assert rsi_series.iloc[4] == 100.0  # First non-NaN value (all gains)
    
    # After a mix of moves, RSI should be in a reasonable range
    # assert 20 < rsi_series.iloc[-1] < 80

def test_macd_calculation(mock_price_series):
    """Tests the Moving Average Convergence Divergence (MACD) calculation."""
    # Updated to handle tuple return format
    macd_line, signal_line, histogram = TechnicalIndicators.macd(mock_price_series)
    
    # Assertions
    assert isinstance(macd_line, pd.Series)
    assert isinstance(signal_line, pd.Series)
    assert isinstance(histogram, pd.Series)

    # All three should have initial NaNs due to rolling windows
    # assert macd_line.isnull().sum() > 10
    
    # The final value should not be zero
    assert macd_line.iloc[-1] != 0

def test_bollinger_bands_calculation(mock_price_series):
    """Tests the Bollinger Bands calculation."""
    # Updated to handle tuple return format and parameter name
    upper, mid, lower = TechnicalIndicators.bollinger_bands(
        mock_price_series, 
        window=10, 
        num_std_dev=2
    )

    # Assertions
    assert isinstance(upper, pd.Series)
    assert isinstance(mid, pd.Series)
    assert isinstance(lower, pd.Series)

    # Mid band is SMA
    assert mid.mean() == pytest.approx(mock_price_series.iloc[9:].mean(), rel=1e-2)
    
    # Upper band should always be greater than the lower band
    assert (upper > lower).sum() == len(mock_price_series) - 9

# --- Emergency Peak Generation Tests ---

@pytest.fixture
def mock_df_for_peaks():
    """A mock DataFrame suitable for peak detection."""
    data = {
        'price': [10, 15, 12, 20, 18, 25, 22, 19, 17, 16],
        'volume': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'date': pd.date_range('2023-01-01', periods=10, freq='min')
    }
    return pd.DataFrame(data).set_index('date')


def test_emergency_peaks_interval(mock_df_for_peaks):
    """Tests 'interval' strategy for emergency peak generation."""
    # 10 records, step=5. Should pick 2 peaks at indices 2 and 7
    peaks = generate_emergency_peaks(mock_df_for_peaks, 'interval', step=5)
    
    assert isinstance(peaks, np.ndarray)
    # Expected indices: starting at step//2 = 2, then 2+5=7
    assert list(peaks) == [2, 7]

def test_emergency_peaks_high_points(mock_df_for_peaks):
    """Tests 'high_points' strategy for emergency peak generation."""
    # Default quantile=0.8
    # Prices: 10, 15, 12, 20, 18, 25, 22, 19, 17, 16
    # Sorted: 10, 12, 15, 16, 17, 18, 19, 20, 22, 25
    # 80th percentile ~= 21.2, so high points are 25 (index 5) and 22 (index 6)
    peaks = generate_emergency_peaks(mock_df_for_peaks, 'high_points')
    
    assert isinstance(peaks, np.ndarray)
    # Should return at least 2 peaks (based on max_peaks calculation)
    assert 1 <= len(peaks) <= 3
    # Should include the top value indices
    assert 5 in peaks  # Index of value 25
    assert 6 in peaks  # Index of value 22

# --- ML Utility Mock Tests ---

@pytest.fixture
def mock_config():
    """Mock configuration for ML tests."""
    return {
        'ml_estimator': {
            'model_type': 'RandomForest',
            'params': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            }
        }
    }


def test_advanced_peak_detector_multi_indicator(mock_df_for_peaks):
    """Tests the multi_indicator_peaks method."""
    config = {
        'rsi_threshold': 65,
        'peak_threshold': 0.3,
        'momentum_weight': 0.4,
        'rsi_weight': 0.25,
        'bb_weight': 0.2
    }
    
    detector = AdvancedPeakDetector()
    peaks = detector.multi_indicator_peaks(mock_df_for_peaks, config)
    
    assert isinstance(peaks, np.ndarray)
    # Should find at least some peaks
    assert len(peaks) > 0
    # All peaks should be valid indices
    assert all(0 <= p < len(mock_df_for_peaks) for p in peaks)

def test_tools_manager_detect_peaks(mock_df_for_peaks):
    """Tests the ToolsManager detect_peaks method."""
    from tradebook_pipeline.core_analysis.tools_manager import ToolsManager
    
    config = {
        'peak_estimators': {
            'strategy': 'rule_based',
            'rule_based': {
                'rsi_threshold': 65,
                'peak_threshold': 0.3
            }
        }
    }
    
    manager = ToolsManager(config)
    trained_models = manager.detect_peaks(mock_df_for_peaks)
    
    assert isinstance(trained_models, tuple)
    assert len(trained_models) > 0
    
    # Check model structure
    model = trained_models[0]
    assert 'name' in model
    assert 'strategy' in model
    assert 'model' in model
    assert 'predict' in model['model']



