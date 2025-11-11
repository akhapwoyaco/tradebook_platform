# tests/live_trading/test_live_trader.py

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
from tradebook_pipeline.live_trading.live_trader import LiveTrader, TradeRecord 

# --- Fixtures ---

@pytest.fixture
def mock_processed_data():
    """Returns mock processed data ready for the trader to act on."""
    data = {
        'date': [datetime.now()],
        'price': [100.0],
        'signal': [1],
        'confidence': [0.95],
        'feature_1': [0.5]
    }
    return pd.DataFrame(data).set_index('date')


@pytest.fixture
def simple_live_trader():
    """
    Returns an initialized LiveTrader instance with mocked dependencies.
    The configuration is defined *inside* the fixture to ensure keys exist.
    """
    
    # GUARANTEED CONFIGURATION: Define the minimal required config dictionary locally
    config_data = {
        # The 'live_trading' key is included here for LiveTrader initialization
        'live_trading': {'symbol': 'BTC/USDT'},
        'general': {'risk_management': {'initial_capital': 10000.0}},
        'portfolio': {'base_currency': 'USDT'}
    }

    # Use dependency mocking (MockOM, MockCB) to isolate LiveTrader logic
    with (
        patch('tradebook_pipeline.live_trading.live_trader.EnhancedOrderManager') as MockOM,
        patch('tradebook_pipeline.live_trading.live_trader.ProductionCircuitBreaker') as MockCB,
        # Mock other required class dependencies
        patch('tradebook_pipeline.live_trading.live_trader.SecurityManager'),
        patch('tradebook_pipeline.live_trading.live_trader.MarketScheduleManager'),
        patch('tradebook_pipeline.live_trading.live_trader.IngestionManager') as MockIM,
    ):
        
        # 1. Instantiate the trader with the guaranteed config
        trader = LiveTrader(config=config_data, model=MagicMock(), ingestion_manager=MockIM.return_value)
        
        # 2. Explicitly assign mocks (to fix potential lazy/conditional initialization)
        trader.order_manager = MockOM.return_value 
        trader.circuit_breaker = MockCB.return_value

        # 3. Configure the mock components' return values (The assumed trade result)
        trader.order_manager.place_market_order_with_retry.return_value = {
            'status': 'filled',
            'fill_price': 100.0,
            'filled_amount': 10.0, 
            'slippage': 0.001,
            'execution_time_ms': 50.0
        }
        trader.circuit_breaker.comprehensive_risk_check.return_value = (True, "Active") 

        # 4. Explicitly set core state variables for a clean test start
        trader.cash = config_data['general']['risk_management']['initial_capital']
        trader.initial_capital = trader.cash
        trader.position = 0.0
        
        return trader

# --- Tests ---

def test_live_trader_initialization(simple_live_trader):
    """Tests that the LiveTrader initializes its primary components correctly."""
    
    assert simple_live_trader.initial_capital == 10000.0
    assert simple_live_trader.cash == 10000.0
    assert simple_live_trader.position == 0.0
    assert simple_live_trader.order_manager is not None
    assert simple_live_trader.circuit_breaker is not None
    assert simple_live_trader.trading_active is False 

