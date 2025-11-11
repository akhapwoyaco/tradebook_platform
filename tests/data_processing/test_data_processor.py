# tests/data_processing/test_data_processor.py

import pytest
import pandas as pd
import numpy as np
from tradebook_pipeline.data_processing.data_processor import DataProcessor


def test_data_processor_initialization(mock_config):
    """Tests the DataProcessor initializes configuration settings correctly."""
    processor = DataProcessor(config=mock_config)
    # Access the nested data_processing config
    assert processor.config.get('volatility_window', 20) == 20
    assert processor.config.get('pressure_window', 5) == 5


def test_clean_data_removes_invalid_rows(mock_data_processor):
    """Tests that _clean_data removes rows with missing price or volume."""
    
    # Create data with NaNs in critical columns
    data = {
        'price': [100.0, 101.0, np.nan, 103.0],
        'volume': [10.0, np.nan, 12.0, 13.0]
    }
    df = pd.DataFrame(data)
    
    cleaned_df = mock_data_processor._clean_data(df)
    
    # Should only have 2 rows left (rows with both price and volume)
    assert len(cleaned_df) == 2
    assert cleaned_df['price'].iloc[0] == 100.0
    assert cleaned_df['price'].iloc[1] == 103.0
    assert not cleaned_df['price'].isnull().any()
    assert not cleaned_df['volume'].isnull().any()


def test_detect_data_type_tradebook(mock_data_processor):
    """Tests auto-detection of tradebook data type."""
    
    df = pd.DataFrame({
        'price': [100.0, 101.0],
        'volume': [10.0, 11.0],
        'type': ['a', 'b']
    })
    
    data_type = mock_data_processor._detect_data_type(df)
    assert data_type == 'tradebook'


def test_detect_data_type_orderbook(mock_data_processor):
    """Tests auto-detection of orderbook data type."""
    
    df = pd.DataFrame({
        'price': [100.0, 101.0],
        'volume': [10.0, 11.0],
        'level': [1, 2]
    })
    
    data_type = mock_data_processor._detect_data_type(df)
    assert data_type == 'orderbook'


def test_process_data_tradebook(mock_data_processor):
    """Tests processing of tradebook data creates expected features."""
    
    # Create sufficient data for rolling windows (at least 20 rows)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=30, freq='min')
    df = pd.DataFrame({
        'price': 100 + np.cumsum(np.random.randn(30) * 0.1),
        'volume': 10 + np.random.rand(30) * 5
    }, index=dates)
    
    processed_df = mock_data_processor.process_data(df.copy(), data_type='tradebook')
    
    # Check that expected features were created
    expected_cols = ['price', 'volume', 'price_ma', 'volume_ma', 
                     'log_returns', 'volatility', 'price_momentum']
    for col in expected_cols:
        assert col in processed_df.columns, f"Missing column: {col}"
    
    # Check that we didn't lose all data
    assert len(processed_df) > 0, "All data was dropped during processing"
    
    # Check that moving averages are calculated (not all NaN)
    assert processed_df['price_ma'].notna().sum() > 0
    assert processed_df['volume_ma'].notna().sum() > 0


def test_process_data_orderbook(mock_data_processor):
    """Tests processing of orderbook data creates expected features."""
    
    # Create orderbook data with 'type' column ('b' for bid, 's' for ask)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=60, freq='s')
    df = pd.DataFrame({
        'price': 100 + np.cumsum(np.random.randn(60) * 0.1),
        'volume': 10 + np.random.rand(60) * 5,
        'type': ['b', 's'] * 30,  # Alternating bid/ask
        'level': [1] * 60
    }, index=dates)
    
    processed_df = mock_data_processor.process_data(df.copy(), data_type='orderbook')
    
    # Check orderbook-specific features
    expected_cols = ['type_numeric', 'signed_volume', 'order_pressure', 
                     'price_ma', 'volume_ma']
    for col in expected_cols:
        assert col in processed_df.columns, f"Missing column: {col}"
    
    # Verify type_numeric mapping
    assert (processed_df['type_numeric'].isin([1, -1])).all()
    
    # Check that we didn't lose all data
    assert len(processed_df) > 0, "All data was dropped during processing"


def test_process_tradebook_features_calculations(mock_data_processor):
    """Tests that tradebook feature calculations are reasonable."""
    
    # Create simple predictable data
    dates = pd.date_range('2023-01-01', periods=25, freq='min')
    df = pd.DataFrame({
        'price': [100.0] * 25,  # Constant price
        'volume': [10.0] * 25   # Constant volume
    }, index=dates)
    
    processed_df = mock_data_processor._process_tradebook_data(df.copy())
    
    # With constant price, moving average should equal price (where not NaN)
    valid_ma = processed_df['price_ma'].dropna()
    assert len(valid_ma) > 0
    assert np.allclose(valid_ma, 100.0), "Price MA should be 100 for constant prices"
    
    # Log returns should be 0 for constant prices
    valid_returns = processed_df['log_returns'].dropna()
    assert len(valid_returns) > 0
    assert np.allclose(valid_returns, 0.0, atol=1e-10), "Log returns should be 0 for constant prices"



def test_process_data_empty_input(mock_data_processor):
    """Tests that process_data handles empty DataFrame gracefully."""
    
    empty_df = pd.DataFrame()
    result = mock_data_processor.process_data(empty_df, data_type='tradebook')
    
    assert result is None, "Should return None for empty DataFrame"


def test_process_data_with_insufficient_data(mock_data_processor):
    """Tests processing with very few rows (less than window size)."""
    
    # Create data with only 5 rows (less than default window of 20)
    df = pd.DataFrame({
        'price': [100.0, 101.0, 102.0, 103.0, 104.0],
        'volume': [10.0, 11.0, 12.0, 13.0, 14.0]
    })
    
    processed_df = mock_data_processor.process_data(df.copy(), data_type='tradebook')
    
    # Should still return a DataFrame, but with NaN values in rolling features
    assert processed_df is not None
    assert len(processed_df) > 0
    # Most rolling features will be NaN due to insufficient window
    assert processed_df['price_ma'].isna().sum() > 0
