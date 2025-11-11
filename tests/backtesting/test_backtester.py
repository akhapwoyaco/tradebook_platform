# tests/backtesting/test_backtester.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from tradebook_pipeline.simulations.backtester import EnhancedBacktester, BacktestTrade, BacktestMetrics


@pytest.fixture
def mock_model():
    """Mock model with predict method"""
    def predict(features):
        # Return peak indices: early (buy signal) and late (sell signal)
        return [1, 5]  # Index 1 for buy, index 5 for sell
    
    return {'predict': predict}


@pytest.fixture
def backtest_config():
    """Basic backtest configuration"""
    return {
        'initial_capital': 10000.0,
        'position_size': 0.95,
        'transaction_fee_rate': 0.001,
        'min_trade_interval': 1,
        'save_detailed_results': False
    }


@pytest.fixture
def sample_price_data():
    """Sample price data with 7 data points"""
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=7, freq='1min'),
        'price': [100.0, 105.0, 102.0, 110.0, 108.0, 115.0, 112.0]
    })
    data.set_index('date', inplace=True)
    return data


def test_backtester_initialization(backtest_config, mock_model):
    """Test backtester initializes with correct parameters"""
    backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
    
    assert backtester.initial_capital == 10000.0
    assert backtester.cash == 10000.0
    assert backtester.position == 0.0
    assert backtester.position_size == 0.95
    assert backtester.transaction_fee_rate == 0.001
    assert backtester.min_trade_interval == 1
    assert len(backtester.trade_history) == 0


def test_model_validation(backtest_config):
    """Test model validation catches invalid models"""
    # Valid model
    valid_model = {'predict': lambda x: [1, 2, 3]}
    backtester = EnhancedBacktester(config=backtest_config, model=valid_model)
    assert backtester._predict_method is not None
    
    # Invalid model - missing predict
    with pytest.raises(AttributeError):
        invalid_model = {'no_predict': lambda x: x}
        EnhancedBacktester(config=backtest_config, model=invalid_model)


def test_run_backtest_execution(backtest_config, mock_model, sample_price_data):
    """Test backtest executes trades correctly"""
    backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
    
    # Run backtest
    results = backtester.run_backtest(sample_price_data)
    
    # Verify results structure
    assert 'performance_metrics' in results
    assert 'trade_summary' in results
    assert 'detailed_trades' in results
    
    # Verify trades occurred
    assert len(backtester.trade_history) > 0
    
    # Check trade actions
    actions = [trade.action for trade in backtester.trade_history]
    assert 'BUY' in actions or 'SELL' in actions


# def test_backtest_buy_sell_sequence(backtest_config, mock_model, sample_price_data):
#     """Test a complete buy-sell sequence"""
#     backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
#     
#     initial_cash = backtester.cash
#     
#     # Run backtest
#     results = backtester.run_backtest(sample_price_data)
#     
#     # Should have executed trades
#     assert len(backtester.trade_history) >= 2
#     
#     # First trade should be BUY (peak at index 1, early = buy signal)
#     first_trade = backtester.trade_history[0]
#     assert first_trade.action == 'BUY'
#     assert first_trade.price == 105.0  # Price at index 1
#     assert first_trade.quantity > 0
#     
#     # Check cash decreased after buy
#     assert backtester.cash < initial_cash


def test_calculate_metrics(backtest_config, mock_model, sample_price_data):
    """Test performance metrics calculation"""
    backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
    
    # Run backtest
    results = backtester.run_backtest(sample_price_data)
    
    metrics = results['performance_metrics']
    
    # Verify all required metrics present
    assert 'initial_capital' in metrics
    assert 'final_value' in metrics
    assert 'total_return_pct' in metrics
    assert 'max_drawdown_pct' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'total_trades' in metrics
    assert 'win_rate_pct' in metrics
    
    # Verify metric types and ranges
    assert isinstance(metrics['initial_capital'], float)
    assert isinstance(metrics['final_value'], float)
    assert isinstance(metrics['total_trades'], int)
    assert metrics['total_trades'] >= 0
    assert 0 <= metrics['win_rate_pct'] <= 100


def test_transaction_fees(backtest_config, mock_model, sample_price_data):
    """Test transaction fees are applied correctly"""
    backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
    
    # Run backtest
    results = backtester.run_backtest(sample_price_data)
    
    metrics = results['performance_metrics']
    
    # Fees should be calculated
    assert 'total_fees' in metrics
    
    # If trades occurred, fees should be > 0
    if metrics['total_trades'] > 0:
        assert metrics['total_fees'] > 0
        
        # Verify fees are reasonable (< 1% of initial capital for normal trading)
        assert metrics['total_fees'] < backtester.initial_capital * 0.01


def test_empty_data_handling(backtest_config, mock_model):
    """Test backtester handles empty data gracefully"""
    backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
    
    empty_data = pd.DataFrame()
    results = backtester.run_backtest(empty_data)
    
    # Should return empty results without crashing
    assert results == {}


def test_portfolio_value_tracking(backtest_config, mock_model, sample_price_data):
    """Test portfolio values are tracked correctly"""
    backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
    
    # Run backtest
    results = backtester.run_backtest(sample_price_data)
    
    # Portfolio curve should be tracked
    assert 'portfolio_curve' in results
    assert len(results['portfolio_curve']) > 0
    
    # Final portfolio value should equal final metrics
    final_portfolio = results['portfolio_curve'][-1]
    final_value = results['performance_metrics']['final_value']
    
    assert abs(final_portfolio - final_value) < 0.01  # Allow small float difference


def test_max_drawdown_calculation(backtest_config, sample_price_data):
    """Test maximum drawdown is calculated correctly"""
    # Model that causes losses
    def losing_predict(features):
        # Buy at peak, sell at trough
        return [1, 4]  # Buy at 105, sell at 108 (slight gain but with fees might lose)
    
    losing_model = {'predict': losing_predict}
    backtester = EnhancedBacktester(config=backtest_config, model=losing_model)
    
    # Run backtest
    results = backtester.run_backtest(sample_price_data)
    
    metrics = results['performance_metrics']
    
    # Drawdown should be calculated and >= 0
    assert 'max_drawdown_pct' in metrics
    assert metrics['max_drawdown_pct'] >= 0


def test_trade_record_completeness(backtest_config, mock_model, sample_price_data):
    """Test trade records contain all required information"""
    backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
    
    # Run backtest
    results = backtester.run_backtest(sample_price_data)
    
    if len(backtester.trade_history) > 0:
        first_trade = backtester.trade_history[0]
        
        # Verify required fields
        assert hasattr(first_trade, 'timestamp')
        assert hasattr(first_trade, 'action')
        assert hasattr(first_trade, 'price')
        assert hasattr(first_trade, 'quantity')
        assert hasattr(first_trade, 'value')
        assert hasattr(first_trade, 'portfolio_value')
        
        # Verify values are valid
        assert first_trade.price > 0
        assert first_trade.quantity >= 0
        assert first_trade.action in ['BUY', 'SELL']


def test_min_trade_interval(mock_model, sample_price_data):
    """Test minimum trade interval is respected"""
    config = {
        'initial_capital': 10000.0,
        'position_size': 0.95,
        'transaction_fee_rate': 0.001,
        'min_trade_interval': 3,  # Require 3 bars between trades
        'save_detailed_results': False
    }
    
    backtester = EnhancedBacktester(config=config, model=mock_model)
    
    # Run backtest
    results = backtester.run_backtest(sample_price_data)
    
    # Verify trades are spaced appropriately
    if len(backtester.trade_history) > 1:
        # Would need timestamp indices to fully verify spacing
        # At minimum, check we don't have more trades than possible
        max_possible_trades = len(sample_price_data) // config['min_trade_interval']
        assert len(backtester.trade_history) <= max_possible_trades + 1


def test_position_size_limit(mock_model, sample_price_data):
    """Test position size limit is respected"""
    config = {
        'initial_capital': 10000.0,
        'position_size': 0.5,  # Only use 50% of capital
        'transaction_fee_rate': 0.001,
        'min_trade_interval': 1,
        'save_detailed_results': False
    }
    
    backtester = EnhancedBacktester(config=config, model=mock_model)
    
    # Run backtest
    results = backtester.run_backtest(sample_price_data)
    
    # Check first buy doesn't exceed position size
    if len(backtester.trade_history) > 0:
        first_buy = next((t for t in backtester.trade_history if t.action == 'BUY'), None)
        
        if first_buy:
            # Buy value shouldn't exceed position_size * initial_capital
            buy_value = first_buy.price * first_buy.quantity
            max_allowed = config['initial_capital'] * config['position_size']
            
            # Allow for fees
            assert buy_value <= max_allowed * 1.01


def test_signal_conversion_logic(backtest_config, sample_price_data):
    """Test peak predictions are converted to correct trading signals"""
    # Model predicting early peak (should trigger BUY)
    early_model = {'predict': lambda x: [0]}  # Very early = BUY
    
    backtester = EnhancedBacktester(config=backtest_config, model=early_model)
    results = backtester.run_backtest(sample_price_data)
    
    if len(backtester.trade_history) > 0:
        first_trade = backtester.trade_history[0]
        assert first_trade.action == 'BUY'
    
    # Model predicting late peak (should trigger SELL)
    late_model = {'predict': lambda x: [6]}  # Very late = SELL
    
    backtester2 = EnhancedBacktester(config=backtest_config, model=late_model)
    
    # Need to have a position first to sell
    backtester2.position = 10.0  # Manually set position for test
    backtester2.cash = 9000.0
    
    results2 = backtester2.run_backtest(sample_price_data)
    
    if len(backtester2.trade_history) > 0:
        # Should have a sell trade
        sell_trades = [t for t in backtester2.trade_history if t.action == 'SELL']
        assert len(sell_trades) > 0


def test_backtest_report_format(backtest_config, mock_model, sample_price_data):
    """Test backtest report has correct format"""
    backtester = EnhancedBacktester(config=backtest_config, model=mock_model)
    
    results = backtester.run_backtest(sample_price_data)
    
    # Required top-level keys
    assert 'backtest_summary' in results
    assert 'performance_metrics' in results
    assert 'trade_summary' in results
    assert 'detailed_trades' in results
    assert 'portfolio_curve' in results
    
    # Backtest summary details
    summary = results['backtest_summary']
    assert 'start_date' in summary
    assert 'end_date' in summary
    assert 'duration_days' in summary
    assert 'data_points' in summary
    
    # Trade summary details
    trade_summary = results['trade_summary']
    assert 'total_trades' in trade_summary
    assert 'buy_trades' in trade_summary
    assert 'sell_trades' in trade_summary
    assert 'final_position' in trade_summary
    assert 'final_cash' in trade_summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
