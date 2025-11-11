# tests/conftest.py
# This file holds fixtures that can be shared across multiple test files.

import pytest
import pandas as pd
from unittest.mock import MagicMock
from pathlib import Path
from tradebook_pipeline.data_ingestion.ingestion_manager import IngestionManager
from tradebook_pipeline.data_processing.data_processor import DataProcessor

# --- Mock Configuration Fixture ---

@pytest.fixture(scope="session")
def mock_config():
    """Returns a comprehensive, mock configuration dictionary."""
    return {
        'data_schema': {'time_col_name': 'date', 'entity_col_name': 'type_encoded', 'event_cols': ['price', 'amount', 'type_encoded']},
        'paths': {
            'raw_data_dir': 'data/raw/', 
            'synthetic_output_dir': 'data/synthetic/datasets/', 
            'models_dir': 'models/', 
            'logs_dir': 'logs/'
        },
        'data_ingestion': {
            'data_sources': {'binance': 'binance', 'local': 'local_file'},
            'tokens_to_monitor': [{'symbol': 'BTC/USDT', 'exchange': 'binance'}]
        },
        'data_processing': {
            'price_ma_window': 5,
            'pressure_window': 5
        },
        'synthetic_data': {
            'generator_model_type': 'Dummy',
            'feature_cols': ['price', 'volume'],
            'seq_len': 24,
            'training_params': {'model_save_path': 'models/synthetic_data/'}
        },
        'backtesting': {
            'initial_capital': 10000,
            'trade_size_pct': 10,
        },
        'live_trading': {
            'api_key': 'mock_key',
            'api_secret': 'mock_secret',
            'exchange': 'binance',
            'symbol': 'BTC/USDT',
            'initial_capital': 10000,
            'max_position_value': 5000,
            'max_daily_loss': 1000,
        },
        'logging': {'log_file_path': 'logs/test.log', 'level': 'INFO'},
        'pipeline': {'monitoring_interval': 1}
    }


# --- Mock Data Fixtures ---

@pytest.fixture
def mock_tradebook_df():
    """Returns a mock DataFrame representing processed tradebook data."""
    data = {
        'date': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00', '2023-01-01 00:03:00', '2023-01-01 00:04:00']),
        'price': [100.0, 101.0, 102.0, 103.0, 104.0],
        'volume': [10.0, 12.0, 8.0, 15.0, 9.0],
        'type_encoded': [1, -1, 1, -1, 1], # 1 for buy, -1 for sell
        'signal': [0, 1, 0, -1, 0] # 1=BUY, -1=SELL, 0=HOLD/PEAK, needs to be changed for backtester
    }
    df = pd.DataFrame(data).set_index('date')
    return df

@pytest.fixture
def mock_tradebook_raw_df():
    """Returns a mock DataFrame simulating raw ingested data."""
    data = {
        'date': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00']),
        'price': [100.0, 101.0, 102.0],
        'amount': [1.0, 1.2, 0.8],
        'type': ['buy', 'sell', 'buy'],
        'symbol': ['BTC/USDT', 'BTC/USDT', 'BTC/USDT']
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def mock_orderbook_raw_df():
    """Returns a mock DataFrame simulating raw orderbook data."""
    data = {
        'date': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:01', '2023-01-01 00:00:02']),
        'price': [100.0, 99.9, 100.1],
        'volume': [5.0, 3.0, 4.0],
        'type': ['b', 's', 'b'], # 'b' for bid/buy, 's' for ask/sell
        'symbol': ['BTC/USDT', 'BTC/USDT', 'BTC/USDT']
    }
    df = pd.DataFrame(data)
    return df

# --- Mock Service Fixtures ---

@pytest.fixture
def mock_ingestion_manager(mock_config):
    """Mocks the IngestionManager instance."""
    manager = IngestionManager(config=mock_config)
    manager._fetch_data_ccxt = MagicMock() # Mock the actual fetching method
    manager._fetch_local_file = MagicMock()
    return manager

@pytest.fixture
def mock_data_processor(mock_config):
    """Mocks the DataProcessor instance."""
    return DataProcessor(config=mock_config)


@pytest.fixture
def temp_model_path(tmp_path):
    """Creates a temporary path for model saving/loading."""
    model_dir = tmp_path / "models/synthetic_data"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
