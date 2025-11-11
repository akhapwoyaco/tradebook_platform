import pandas as pd
from loguru import logger
from typing import Any, Dict
from tradebook_pipeline.data_processing.data_processor import DataProcessor

import pandas as pd
from loguru import logger
from typing import Any, Dict



import pandas as pd
import numpy as np
from loguru import logger
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

@dataclass
class BacktestTrade:
    """Data class for individual backtest trades"""
    timestamp: str
    action: str  # 'BUY', 'SELL'
    price: float
    quantity: float
    value: float
    portfolio_value: float
    signal_strength: Optional[float] = None
    peak_index: Optional[int] = None

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    initial_capital: float
    final_value: float
    total_return_pct: float
    profit_loss: float
    max_drawdown_pct: float
    sharpe_ratio: float
    volatility: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_trade_return_pct: float
    avg_winning_trade_pct: float
    avg_losing_trade_pct: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_fees: float

class EnhancedBacktester:
    """
    Enhanced backtesting system that properly handles peak detection models
    and provides comprehensive performance analytics.
    """

    def __init__(self, config: Dict[str, Any], model: Any):
        """
        Initialize the Enhanced Backtester.
        
        Args:
            config: Backtesting configuration
            model: Trained model with predict method
        """
        self.config = config
        self.model = model
        
        # Trading parameters
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.position_size = self.config.get('position_size', 0.95)  # Use 95% of capital
        self.transaction_fee_rate = self.config.get('transaction_fee_rate', 0.001)  # 0.1%
        self.min_trade_interval = self.config.get('min_trade_interval', 1)  # Minimum bars between trades
        
        # State tracking
        self.cash = self.initial_capital
        self.position = 0.0  # Number of tokens held
        self.trade_history: List[BacktestTrade] = []
        self.portfolio_values: List[float] = []
        self.peak_portfolio_value = self.initial_capital
        self.last_trade_index = -1
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
        # Output configuration
        self.save_detailed_results = self.config.get('save_detailed_results', True)
        self.output_dir = self.config.get('output_directory', './backtest_outputs')
        self._setup_output_directory()
        
        self._validate_model()
        logger.info(f"Enhanced Backtester initialized - Capital: ${self.initial_capital:,.2f}, "
                   f"Position Size: {self.position_size:.2%}, Fee Rate: {self.transaction_fee_rate:.3%}")

    def _setup_output_directory(self):
        """Create output directory for detailed results"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Backtest output directory: {self.output_dir}")
        except Exception as e:
            logger.warning(f"Could not create output directory: {e}")
            self.save_detailed_results = False

    def _validate_model(self):
        """Validate model has required predict method"""
        if isinstance(self.model, dict) and callable(self.model.get('predict')):
            self._predict_method = self.model.get('predict')
        elif hasattr(self.model, 'predict') and callable(self.model.predict):
            self._predict_method = self.model.predict
        else:
            raise AttributeError("Model does not have a callable 'predict' method.")

    def run_backtest(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive backtest simulation.
        
        Args:
            historical_data: Processed historical data with features and prices
            
        Returns:
            Dictionary with backtest results and metrics
        """
        if historical_data.empty:
            logger.error("Historical data is empty. Cannot run backtest.")
            return {}

        logger.info(f"Starting enhanced backtest with {len(historical_data)} data points")
        logger.info(f"Data period: {historical_data.index[0]} to {historical_data.index[-1]}")
        
        # Strategy 1: Batch prediction approach (recommended)
        return self._run_batch_backtest(historical_data)

    def _run_batch_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest using batch predictions (more efficient and realistic).
        
        This approach gets all predictions at once, then simulates trading
        through the historical data with those signals.
        """
        logger.info("Running batch prediction backtest...")
        
        try:
            # Get all predictions at once
            features = data.drop(columns=['price'], errors='ignore')
            all_predictions = self._predict_method(features)
            
            if len(all_predictions) == 0:
                logger.warning("Model returned no predictions")
                return self._generate_empty_report()
            
            # Convert predictions to trading signals
            trading_signals = self._convert_predictions_to_signals(all_predictions, len(data))
            logger.info(f"Generated {len(trading_signals)} trading signals from {len(all_predictions)} predictions")
            
            # Execute trades based on signals
            self._execute_backtest_trades(data, trading_signals)
            
        except Exception as e:
            logger.error(f"Error in batch backtest: {e}")
            return self._generate_empty_report()
        
        return self._generate_comprehensive_report(data)

    def _convert_predictions_to_signals(self, predictions: List[int], data_length: int) -> List[Dict]:
        """
        Convert peak predictions to actionable trading signals.
        
        Args:
            predictions: List of peak indices from model
            data_length: Length of the dataset
            
        Returns:
            List of trading signals with timestamps and actions
        """
        signals = []
        
        for peak_index in predictions:
            if 0 <= peak_index < data_length:
                # Determine signal type based on peak position
                position_pct = peak_index / data_length
                
                if position_pct > 0.8:  # Recent peak - sell signal
                    signal_type = 'SELL'
                    strength = min(1.0, (position_pct - 0.8) / 0.2)  # Stronger as more recent
                elif position_pct < 0.3:  # Early peak - buy signal
                    signal_type = 'BUY'  
                    strength = min(1.0, (0.3 - position_pct) / 0.3)  # Stronger as earlier
                else:
                    continue  # Skip mid-range peaks
                
                signals.append({
                    'index': peak_index,
                    'action': signal_type,
                    'strength': strength,
                    'position_pct': position_pct
                })
        
        # Sort signals by index to process chronologically
        signals.sort(key=lambda x: x['index'])
        
        logger.info(f"Signal breakdown: {sum(1 for s in signals if s['action'] == 'BUY')} BUY, "
                   f"{sum(1 for s in signals if s['action'] == 'SELL')} SELL")
        
        return signals

    def _execute_backtest_trades(self, data: pd.DataFrame, signals: List[Dict]):
        """
        Execute trades based on generated signals.
        
        Args:
            data: Historical price data
            signals: List of trading signals
        """
        signal_idx = 0
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['price']
            portfolio_value = self.cash + (self.position * current_price)
            self.portfolio_values.append(portfolio_value)
            
            # Update peak portfolio value and calculate drawdown
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value
            
            # Check if we have a signal for this timestamp
            if signal_idx < len(signals) and signals[signal_idx]['index'] == i:
                signal = signals[signal_idx]
                
                # Check minimum trade interval
                if i - self.last_trade_index >= self.min_trade_interval:
                    self._execute_signal_trade(signal, current_price, timestamp, portfolio_value)
                    self.last_trade_index = i
                else:
                    logger.debug(f"Skipping trade at index {i} - minimum interval not met")
                
                signal_idx += 1

    def _execute_signal_trade(self, signal: Dict, price: float, timestamp: str, portfolio_value: float):
        """
        Execute individual trade based on signal.
        
        Args:
            signal: Trading signal dictionary
            price: Current price
            timestamp: Current timestamp
            portfolio_value: Current portfolio value
        """
        action = signal['action']
        strength = signal['strength']
        
        if action == 'SELL' and self.position > 0:
            # Sell all position
            sell_value = self.position * price
            fee = sell_value * self.transaction_fee_rate
            net_value = sell_value - fee
            
            trade = BacktestTrade(
                timestamp=str(timestamp),
                action='SELL',
                price=price,
                quantity=self.position,
                value=net_value,
                portfolio_value=portfolio_value,
                signal_strength=strength,
                peak_index=signal['index']
            )
            
            self.cash += net_value
            self.position = 0
            self.trade_history.append(trade)
            
            logger.info(f"BACKTEST SELL: {trade.quantity:.6f} tokens at ${price:.4f}, "
                       f"Value: ${net_value:,.2f}, Strength: {strength:.2f}")
        
        elif action == 'BUY' and self.position == 0 and self.cash > 0:
            # Buy with specified position size
            buy_amount = self.cash * self.position_size
            fee = buy_amount * self.transaction_fee_rate
            net_buy_amount = buy_amount - fee
            quantity = net_buy_amount / price
            
            if quantity > 0:
                trade = BacktestTrade(
                    timestamp=str(timestamp),
                    action='BUY',
                    price=price,
                    quantity=quantity,
                    value=net_buy_amount,
                    portfolio_value=portfolio_value,
                    signal_strength=strength,
                    peak_index=signal['index']
                )
                
                self.cash -= buy_amount
                self.position = quantity
                self.trade_history.append(trade)
                
                logger.info(f"BACKTEST BUY: {quantity:.6f} tokens at ${price:.4f}, "
                           f"Cost: ${buy_amount:,.2f}, Strength: {strength:.2f}")

    def _calculate_performance_metrics(self, data: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        
        if not self.portfolio_values:
            return self._get_empty_metrics()
        
        final_value = self.portfolio_values[-1]
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100
        profit_loss = final_value - self.initial_capital
        
        # Calculate returns for Sharpe ratio and volatility
        portfolio_returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            portfolio_returns.append(daily_return)
        
        # Risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252) if portfolio_returns else 0  # Annualized
        sharpe_ratio = (np.mean(portfolio_returns) * 252) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        max_drawdown_pct = 0
        if self.portfolio_values:
            peak = self.portfolio_values[0]
            for value in self.portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                if drawdown > max_drawdown_pct:
                    max_drawdown_pct = drawdown
        
        # Trade statistics
        winning_trades = 0
        losing_trades = 0
        trade_returns = []
        
        for i, trade in enumerate(self.trade_history):
            if trade.action == 'SELL' and i > 0:
                # Find corresponding buy
                for j in range(i-1, -1, -1):
                    if self.trade_history[j].action == 'BUY':
                        trade_return = (trade.price - self.trade_history[j].price) / self.trade_history[j].price * 100
                        trade_returns.append(trade_return)
                        if trade_return > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1
                        break
        
        total_trades = len([t for t in self.trade_history if t.action in ['BUY', 'SELL']])
        win_rate_pct = (winning_trades / len(trade_returns) * 100) if trade_returns else 0
        avg_trade_return_pct = np.mean(trade_returns) if trade_returns else 0
        
        winning_returns = [r for r in trade_returns if r > 0]
        losing_returns = [r for r in trade_returns if r <= 0]
        
        avg_winning_trade_pct = np.mean(winning_returns) if winning_returns else 0
        avg_losing_trade_pct = np.mean(losing_returns) if losing_returns else 0
        
        # Calculate total fees
        total_fees = sum(trade.value * self.transaction_fee_rate for trade in self.trade_history)
        
        return BacktestMetrics(
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            profit_loss=profit_loss,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate_pct,
            avg_trade_return_pct=avg_trade_return_pct,
            avg_winning_trade_pct=avg_winning_trade_pct,
            avg_losing_trade_pct=avg_losing_trade_pct,
            max_consecutive_wins=self.max_consecutive_wins,
            max_consecutive_losses=self.max_consecutive_losses,
            total_fees=total_fees
        )

    def _generate_comprehensive_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        
        metrics = self._calculate_performance_metrics(data)
        
        # Log summary
        logger.info("=" * 60)
        logger.info("ENHANCED BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Period: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Initial Capital: ${metrics.initial_capital:,.2f}")
        logger.info(f"Final Value: ${metrics.final_value:,.2f}")
        logger.info(f"Total Return: {metrics.total_return_pct:.2f}%")
        logger.info(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        logger.info(f"Volatility: {metrics.volatility:.2f}%")
        logger.info(f"Total Trades: {metrics.total_trades}")
        logger.info(f"Win Rate: {metrics.win_rate_pct:.2f}%")
        logger.info(f"Avg Trade Return: {metrics.avg_trade_return_pct:.2f}%")
        logger.info(f"Total Fees: ${metrics.total_fees:.2f}")
        logger.info("=" * 60)
        
        # Create comprehensive report
        report = {
            'backtest_summary': {
                'start_date': str(data.index[0]),
                'end_date': str(data.index[-1]),
                'duration_days': len(data),
                'data_points': len(data)
            },
            'performance_metrics': asdict(metrics),
            'trade_summary': {
                'total_trades': len(self.trade_history),
                'buy_trades': len([t for t in self.trade_history if t.action == 'BUY']),
                'sell_trades': len([t for t in self.trade_history if t.action == 'SELL']),
                'final_position': self.position,
                'final_cash': self.cash
            },
            'detailed_trades': [asdict(trade) for trade in self.trade_history],
            'portfolio_curve': self.portfolio_values
        }
        
        # Save detailed results if enabled
        if self.save_detailed_results:
            self._save_backtest_results(report)
        
        return report

    def _save_backtest_results(self, report: Dict[str, Any]):
        """Save detailed backtest results to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main report
            report_file = os.path.join(self.output_dir, f'backtest_report_{timestamp}.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save trades as CSV
            if self.trade_history:
                trades_df = pd.DataFrame([asdict(trade) for trade in self.trade_history])
                trades_file = os.path.join(self.output_dir, f'backtest_trades_{timestamp}.csv')
                trades_df.to_csv(trades_file, index=False)
                
                logger.info(f"Backtest results saved to {report_file}")
                logger.info(f"Trade details saved to {trades_file}")
            
        except Exception as e:
            logger.warning(f"Could not save detailed backtest results: {e}")

    def _get_empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics for failed backtests."""
        return BacktestMetrics(
            initial_capital=self.initial_capital,
            final_value=self.initial_capital,
            total_return_pct=0.0,
            profit_loss=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            volatility=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            avg_trade_return_pct=0.0,
            avg_winning_trade_pct=0.0,
            avg_losing_trade_pct=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            total_fees=0.0
        )

    def _generate_empty_report(self) -> Dict[str, Any]:
        """Generate empty report for failed backtests."""
        return {
            'performance_metrics': asdict(self._get_empty_metrics()),
            'trade_summary': {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'final_position': 0,
                'final_cash': self.initial_capital
            },
            'detailed_trades': [],
            'portfolio_curve': [self.initial_capital]
        }

# Integration helper for your existing pipeline
def create_enhanced_backtester(config: Dict[str, Any], model: Any) -> EnhancedBacktester:
    """
    Factory function to create enhanced backtester with your existing configuration.
    
    Args:
        config: Your existing simulations config
        model: Trained model from your pipeline
        
    Returns:
        Configured EnhancedBacktester instance
    """
    # Map your existing config to enhanced backtester format
    enhanced_config = {
        'initial_capital': config.get('initial_capital', 100000),
        'position_size': config.get('position_size', 0.95),
        'transaction_fee_rate': config.get('transaction_fee_rate', 0.001),
        'min_trade_interval': config.get('min_trade_interval', 1),
        'save_detailed_results': config.get('save_detailed_results', True),
        'output_directory': config.get('output_directory', './backtest_outputs')
    }
    
    return EnhancedBacktester(enhanced_config, model)
  
