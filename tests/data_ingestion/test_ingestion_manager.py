# tests/data_ingestion/test_ingestion_manager.py

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
import ccxt
from tradebook_pipeline.data_ingestion.ingestion_manager import IngestionManager


@pytest.fixture
def mock_config():
    """Fixture providing a complete mock configuration matching the expected structure."""
    return {
        'data_ingestion': {
            'data_sources': {
                'binance': 'binance',
                'local': 'local_file'
            },
            'tokens_to_monitor': [
                {
                    'symbol': 'BTC/USDT',
                    'source': 'binance'
                }
            ],
            'google_drive': {
                'enabled': False
            }
        },
        'exchanges': {
            'binance': {
                'apiKey': 'test_key',
                'secret': 'test_secret'
            }
        }
    }


@pytest.fixture
def ingestion_manager_instance(mock_config):
    """Returns an initialized IngestionManager instance."""
    return IngestionManager(config=mock_config)

# 
# def test_ingestion_manager_initialization(ingestion_manager_instance, mock_config):
#     """Tests the IngestionManager initializes core properties correctly."""
#     # Access the data_ingestion config section
#     data_ingestion_config = mock_config['data_ingestion']
#     
#     # Verify data sources are accessible
#     assert 'binance' in data_ingestion_config['data_sources']
#     assert 'local' in data_ingestion_config['data_sources']
#     
#     # Verify tokens are loaded
#     tokens = data_ingestion_config['tokens_to_monitor']
#     assert len(tokens) == 1
#     assert tokens[0]['symbol'] == 'BTC/USDT'
#     
#     # Verify the manager stored the config correctly
#     assert ingestion_manager_instance.config is not None
# 
# 
# @patch('tradebook_pipeline.data_ingestion.ingestion_manager.ccxt.binance')
# def test_fetch_data_ccxt_success(mock_ccxt_binance, ingestion_manager_instance):
#     """Tests data fetching from CCXT exchanges."""
#     
#     # Create mock exchange instance
#     mock_exchange_instance = MagicMock()
#     mock_ccxt_binance.return_value = mock_exchange_instance
#     
#     # Mock the fetch_trades method to return dummy CCXT trade data
#     mock_exchange_instance.fetch_trades.return_value = [
#         {
#             'timestamp': 1672531200000,
#             'datetime': '2023-01-01T00:00:00.000Z',
#             'symbol': 'BTC/USDT',
#             'price': 100.0,
#             'amount': 1.0,
#             'side': 'buy',
#             'id': '1'
#         },
#         {
#             'timestamp': 1672531260000,
#             'datetime': '2023-01-01T00:01:00.000Z',
#             'symbol': 'BTC/USDT',
#             'price': 101.0,
#             'amount': 2.0,
#             'side': 'sell',
#             'id': '2'
#         },
#     ]
#     
#     # Mock exchange properties
#     mock_exchange_instance.id = 'binance'
#     mock_exchange_instance.rateLimit = 1200
#     mock_exchange_instance.has = {'fetchTrades': True}
#     
#     start = datetime(2023, 1, 1)
#     end = datetime(2023, 1, 2)
#     
#     # Call the actual method that exists in IngestionManager
#     df = ingestion_manager_instance._fetch_from_exchange(
#         symbol='BTC/USDT',
#         start_time=start,
#         end_time=end,
#         data_type='tradebook',
#         limit=1000
#     )
#     
#     # Assertions
#     assert isinstance(df, pd.DataFrame)
#     assert len(df) == 2
#     assert 'price' in df.columns
#     assert df['price'].iloc[0] == 100.0
#     # Note: 'side' column is renamed to match the implementation
#     if 'side' in df.columns:
#         assert df['side'].iloc[1] == 'sell'
# 
# 
# @patch('tradebook_pipeline.data_ingestion.ingestion_manager.ccxt.binance')
# def test_fetch_data_ccxt_failure_and_retry(mock_ccxt_binance, ingestion_manager_instance):
#     """Tests CCXT network errors trigger the retry mechanism."""
#     
#     mock_exchange_instance = MagicMock()
#     mock_ccxt_binance.return_value = mock_exchange_instance
#     
#     # Mock exchange properties
#     mock_exchange_instance.id = 'binance'
#     mock_exchange_instance.rateLimit = 1200
#     mock_exchange_instance.has = {'fetchTrades': True}
#     
#     # Make the first two calls fail with a network error, and the third succeed
#     mock_exchange_instance.fetch_trades.side_effect = [
#         ccxt.NetworkError('Timeout'),
#         ccxt.NetworkError('Connection Lost'),
#         [{'timestamp': 1672531200000, 'price': 100.0, 'amount': 1.0, 'side': 'buy', 'symbol': 'BTC/USDT'}]
#     ]
#     
#     start = datetime(2023, 1, 1)
#     end = datetime(2023, 1, 2)
#     
#     # We expect the call to succeed on the 3rd attempt
#     df = ingestion_manager_instance._fetch_from_exchange(
#         symbol='BTC/USDT',
#         start_time=start,
#         end_time=end,
#         data_type='tradebook',
#         limit=1000
#     )
#     
#     # Verify retry happened (3 attempts total)
#     assert mock_exchange_instance.fetch_trades.call_count == 3
#     assert len(df) == 1
# 
# 
# @patch('tradebook_pipeline.data_ingestion.ingestion_manager.IngestionManager._fetch_from_local_file')
# @patch('tradebook_pipeline.data_ingestion.ingestion_manager.IngestionManager._fetch_from_exchange')
# def test_fetch_selective_data_combines_sources(
#     mock_fetch_exchange,
#     mock_fetch_local,
#     ingestion_manager_instance
# ):
#     """Tests the main orchestration method combines data from multiple sources."""
#     
#     start = datetime(2024, 1, 1)
#     end = datetime(2025, 1, 2)
#     
#     # Mock data from exchange (Source: binance, Type: exchange)
#     exchange_df = pd.DataFrame({
#         'price': [100.0],
#         'volume': [1.0],
#         'symbol': ['BTC/USDT'],
#         'source': ['binance']
#     })
#     exchange_df.index = pd.DatetimeIndex([start + timedelta(minutes=1)])
#     exchange_df.index.name = 'date'
#     mock_fetch_exchange.return_value = exchange_df
#     
#     # Mock data from local file (Source: local, Type: local_file)
#     local_df = pd.DataFrame({
#         'price': [101.0],
#         'volume': [2.0],
#         'symbol': ['BTC/USDT'],
#         'source': ['local']
#     })
#     local_df.index = pd.DatetimeIndex([start + timedelta(minutes=2)])
#     local_df.index.name = 'date'
#     mock_fetch_local.return_value = local_df
#     
#     # Update the manager's config to include both sources
#     ingestion_manager_instance.config['data_sources']['local_source'] = 'local_file'
#     ingestion_manager_instance.config['tokens_to_monitor'].append({
#         'symbol': 'BTC/USDT',
#         'source': 'local_source'
#     })
#     
#     # Call fetch_selective_data with proper parameters
#     combined_df = ingestion_manager_instance.fetch_selective_data(
#         start_time=start,
#         end_time=end,
#         data_type='tradebook',
#         source_type='exchange',  # Primary source type
#         symbols=['BTC/USDT'],
#         source_types=['exchange', 'local_file']
#     )
#     
#     # Assertions
#     assert isinstance(combined_df, pd.DataFrame)
#     assert len(combined_df) == 2  # One record from each mock
#     
#     # Verify both fetch methods were called
#     assert mock_fetch_exchange.call_count >= 1
#     assert mock_fetch_local.call_count >= 1
# 
# 
# def test_get_source_type(ingestion_manager_instance):
#     """Test the _get_source_type method correctly identifies source types."""
#     # Test exchange identification
#     assert ingestion_manager_instance._get_source_type('binance') == 'exchange'
#     
#     # Test local file identification
#     ingestion_manager_instance.config['data_sources']['local_test'] = 'local_file'
#     assert ingestion_manager_instance._get_source_type('local_test') == 'local_file'
# 
# 
# def test_fetch_single_token(ingestion_manager_instance):
#     """Test the convenience method for fetching single token data."""
#     start = datetime(2023, 1, 1)
#     end = datetime(2023, 1, 2)
#     
#     with patch.object(ingestion_manager_instance, '_fetch_from_exchange') as mock_fetch:
#         mock_df = pd.DataFrame({
#             'price': [100.0],
#             'volume': [1.0],
#             'symbol': ['BTC/USDT']
#         })
#         mock_df.index = pd.DatetimeIndex([start])
#         mock_fetch.return_value = mock_df
#         
#         result = ingestion_manager_instance.fetch_single_token(
#             symbol='BTC/USDT',
#             start_time=start,
#             end_time=end,
#             data_type='tradebook'
#         )
#         
#         assert result is not None
#         assert len(result) == 1
#         mock_fetch.assert_called_once()
# 
# 
# def test_validate_inputs(ingestion_manager_instance):
#     """Test input validation logic."""
#     start = datetime(2023, 1, 1)
#     end = datetime(2023, 1, 2)
#     
#     # Valid inputs
#     assert ingestion_manager_instance._validate_inputs('BTC/USDT', start, end, 'tradebook') is True
#     
#     # Invalid symbol format
#     assert ingestion_manager_instance._validate_inputs('BTCUSDT', start, end, 'tradebook') is False
    
    # Invalid data type
    assert ingestion_manager_instance._validate_inputs('BTC/USDT', start, end, 'invalid_type') is False
    
    # Invalid time range (start >= end)
    assert ingestion_manager_instance._validate_inputs('BTC/USDT', end, start, 'tradebook') is False
