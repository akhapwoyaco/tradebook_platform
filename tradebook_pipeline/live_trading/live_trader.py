"""
Production-Ready Live Trading System for Cryptocurrency Trading
================================================================

This module provides a complete, production-grade live trading system with:
- Real exchange integration via CCXT
- Comprehensive order management with retry logic
- Advanced circuit breakers and risk management
- Position synchronization and reconciliation
- Network resilience and error recovery
- Security features (API key encryption)
- Market hours and maintenance detection
- Full monitoring and reporting

License: MIT
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Any, Dict, Optional, List, Tuple, Union
import time
import json
import os
from datetime import datetime, timedelta
from collections import deque
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import pickle
import ccxt
from enum import Enum
import uuid
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import socket
import requests
from cryptography.fernet import Fernet

# Import project modules
from tradebook_pipeline.data_processing.data_processor import DataProcessor
from tradebook_pipeline.data_ingestion.ingestion_manager import IngestionManager


import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# SECTION 1: DATA CLASSES AND ENUMERATIONS
# ============================================================================

@dataclass
class TradeRecord:
    """Enhanced trade record with full exchange integration"""
    timestamp: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    quantity: float
    cash_after: float
    position_after: float
    prediction: int
    confidence: Optional[float] = None
    features_hash: Optional[str] = None
    order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    slippage: Optional[float] = None
    execution_time_ms: Optional[float] = None
    order_status: Optional[str] = None
    partial_fills: Optional[List[Dict]] = None
    retry_count: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking"""
    total_trades: int = 0
    successful_trades: int = 0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_return: float = 0.0
    volatility: float = 0.0
    total_fees: float = 0.0
    avg_slippage: float = 0.0
    execution_latency_ms: float = 0.0
    order_rejection_rate: float = 0.0
    network_error_rate: float = 0.0
    position_sync_accuracy: float = 1.0


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"
    CANCELLING = "cancelling"


class ExchangeStatus(Enum):
    """Exchange connectivity status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    RATE_LIMITED = "rate_limited"


# ============================================================================
# SECTION 2: SECURITY AND INFRASTRUCTURE
# ============================================================================

class SecurityManager:
    """Handles secure API key management and encryption"""
    
    def __init__(self):
        self.cipher_suite = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize Fernet encryption for sensitive data"""
        try:
            key_file = ".trader_key"
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
            
            self.cipher_suite = Fernet(key)
            logger.info("Security manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.cipher_suite = None
    
    def encrypt_api_credentials(self, api_key: str, secret: str, 
                               passphrase: str = None) -> Dict[str, str]:
        """Encrypt API credentials"""
        try:
            if not self.cipher_suite:
                return {"api_key": api_key, "secret": secret, "passphrase": passphrase}
            
            encrypted = {
                "api_key": self.cipher_suite.encrypt(api_key.encode()).decode(),
                "secret": self.cipher_suite.encrypt(secret.encode()).decode()
            }
            
            if passphrase:
                encrypted["passphrase"] = self.cipher_suite.encrypt(passphrase.encode()).decode()
            
            return encrypted
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return {"api_key": api_key, "secret": secret, "passphrase": passphrase}
    
    def decrypt_api_credentials(self, encrypted_data: Dict[str, str]) -> Dict[str, str]:
        """Decrypt API credentials"""
        try:
            if not self.cipher_suite:
                return encrypted_data
            
            decrypted = {
                "api_key": self.cipher_suite.decrypt(encrypted_data["api_key"].encode()).decode(),
                "secret": self.cipher_suite.decrypt(encrypted_data["secret"].encode()).decode()
            }
            
            if "passphrase" in encrypted_data:
                decrypted["passphrase"] = self.cipher_suite.decrypt(encrypted_data["passphrase"].encode()).decode()
            
            return decrypted
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data


class NetworkManager:
    """Manages network connectivity and quality monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.network_stats = {
            'total_requests': 0,
            'failed_requests': 0,
            'last_successful_request': None,
            'consecutive_failures': 0
        }
        self._response_times = deque(maxlen=100)
    
    def test_connectivity(self, exchange_url: str) -> Tuple[bool, float]:
        """Test network connectivity to exchange"""
        start_time = time.time()
        
        try:
            response = requests.get(f"https://{exchange_url}", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            self._response_times.append(response_time)
            self.network_stats['total_requests'] += 1
            self.network_stats['last_successful_request'] = datetime.now()
            self.network_stats['consecutive_failures'] = 0
            
            return True, response_time
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.network_stats['total_requests'] += 1
            self.network_stats['failed_requests'] += 1
            self.network_stats['consecutive_failures'] += 1
            
            logger.warning(f"Connectivity test failed: {e}")
            return False, response_time
    
    def get_network_quality(self) -> Dict[str, Any]:
        """Get current network quality metrics"""
        total = self.network_stats['total_requests']
        failed = self.network_stats['failed_requests']
        
        return {
            'success_rate': 1.0 - (failed / max(1, total)),
            'avg_response_time_ms': np.mean(self._response_times) if self._response_times else 0,
            'consecutive_failures': self.network_stats['consecutive_failures'],
            'last_successful_request': self.network_stats['last_successful_request'],
            'quality_score': self._calculate_quality_score()
        }
    
    def _calculate_quality_score(self) -> float:
        """Calculate network quality score (0-1)"""
        if self.network_stats['total_requests'] < 10:
            return 0.5
        
        success_rate = 1.0 - (self.network_stats['failed_requests'] / self.network_stats['total_requests'])
        avg_response = np.mean(self._response_times) if self._response_times else 1000
        
        success_score = success_rate * 0.7
        speed_score = max(0, 1.0 - (avg_response / 5000)) * 0.3
        
        return min(1.0, success_score + speed_score)


class MarketScheduleManager:
    """Manages trading hours and market maintenance windows"""
    
    def __init__(self, exchange_id: str):
        self.exchange_id = exchange_id
        self.market_hours = self._get_market_hours()
        self.current_status = "unknown"
    
    def _get_market_hours(self) -> Dict[str, Any]:
        """Get market hours configuration"""
        schedules = {
            'binance': {'24_7': True, 'maintenance_day': 'wednesday', 'maintenance_hour': 8},
            'kraken': {'24_7': True},
            'coinbasepro': {'24_7': True},
            'bybit': {'24_7': True, 'maintenance_day': 'friday', 'maintenance_hour': 8},
        }
        return schedules.get(self.exchange_id, {'24_7': True})
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        if self._is_maintenance_window(datetime.now()):
            return False
        return self.market_hours.get('24_7', True)
    
    def _is_maintenance_window(self, timestamp: datetime) -> bool:
        """Check if in maintenance window"""
        if 'maintenance_day' not in self.market_hours:
            return False
        
        day = self.market_hours['maintenance_day']
        hour = self.market_hours.get('maintenance_hour', 8)
        
        if timestamp.strftime('%A').lower() == day:
            if hour <= timestamp.hour < hour + 2:
                return True
        return False
    
    def get_next_market_open(self) -> Optional[datetime]:
        """Get next market opening time"""
        if self.is_market_open():
            return None
        
        now = datetime.now()
        if self._is_maintenance_window(now):
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
        return None


# ============================================================================
# SECTION 3: ORDER MANAGEMENT SYSTEM
# ============================================================================

class EnhancedOrderManager:
    """Production-grade order management with lifecycle tracking"""
    
    def __init__(self, exchange_config: Dict[str, Any], security_manager: SecurityManager):
        self.exchange_config = exchange_config
        self.security_manager = security_manager
        self.exchange = None
        self.pending_orders = {}
        self.order_history = []
        self.partial_fills = {}
        self.last_heartbeat = None
        self._lock = threading.Lock()
        
        # Tracking metrics
        self.execution_times = deque(maxlen=1000)
        self.slippage_history = deque(maxlen=1000)
        self.rejection_count = 0
        self.total_orders = 0
        self.network_errors = 0
        
        # Configuration
        self.max_retries = exchange_config.get('max_order_retries', 3)
        self.order_timeout = exchange_config.get('order_timeout_seconds', 30)
        self.slippage_tolerance = exchange_config.get('slippage_tolerance_pct', 0.5) / 100
        
        # Position tracking
        self.exchange_positions = {}
        self.last_position_sync = None
        self.position_sync_errors = 0
        
        # Network management
        self.network_manager = NetworkManager(exchange_config)
    
    def initialize_exchange(self) -> bool:
        """Initialize exchange with comprehensive testing"""
        try:
            exchange_name = self.exchange_config.get('name', 'binance')
            
            if not hasattr(ccxt, exchange_name):
                logger.error(f"Exchange {exchange_name} not supported")
                return False
            
            exchange_class = getattr(ccxt, exchange_name)
            
            # Decrypt credentials
            encrypted_creds = {
                'api_key': self.exchange_config.get('api_key', ''),
                'secret': self.exchange_config.get('secret', ''),
                'passphrase': self.exchange_config.get('passphrase', '')
            }
            
            if encrypted_creds['api_key'] and encrypted_creds['secret']:
                creds = self.security_manager.decrypt_api_credentials(encrypted_creds)
            else:
                creds = encrypted_creds
            
            # Build configuration
            config = {
                'apiKey': creds['api_key'],
                'secret': creds['secret'],
                'timeout': self.exchange_config.get('timeout', 30000),
                'rateLimit': self.exchange_config.get('rate_limit', 1200),
                'enableRateLimit': True,
                'sandbox': self.exchange_config.get('sandbox', True),
                'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
            }
            
            if creds.get('passphrase'):
                config['password'] = creds['passphrase']
            
            self.exchange = exchange_class(config)
            
            return self._comprehensive_connection_test()
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
            return False
    
    def _comprehensive_connection_test(self) -> bool:
        """Test exchange connectivity comprehensively"""
        try:
            # Test 1: Load markets
            logger.info("Testing market data access...")
            self.exchange.load_markets()
            
            # Test 2: Network connectivity
            exchange_urls = {
                'binance': 'api.binance.com',
                'kraken': 'api.kraken.com',
                'coinbasepro': 'api.exchange.coinbase.com'
            }
            
            url = exchange_urls.get(self.exchange.id, f'api.{self.exchange.id}.com')
            connected, response_time = self.network_manager.test_connectivity(url)
            
            if not connected:
                logger.error("Network test failed")
                return False
            
            logger.info(f"Network test passed - {response_time:.1f}ms")
            
            # Test 3: Authentication (if API keys provided)
            if self.exchange_config.get('api_key'):
                logger.info("Testing authentication...")
                try:
                    balance = self.exchange.fetch_balance()
                    logger.info("Authentication successful")
                except ccxt.AuthenticationError as e:
                    logger.error(f"Authentication failed: {e}")
                    return False
            
            self.last_heartbeat = datetime.now()
            logger.info(f"Exchange {self.exchange.id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def synchronize_positions(self, symbol: str) -> Dict[str, float]:
        """Synchronize positions with exchange"""
        try:
            if not self.exchange:
                raise Exception("Exchange not initialized")
            
            balance = self.exchange.fetch_balance()
            base = symbol.split('/')[0]
            quote = symbol.split('/')[1]
            
            position = {
                'base_free': balance.get(base, {}).get('free', 0.0),
                'base_total': balance.get(base, {}).get('total', 0.0),
                'quote_free': balance.get(quote, {}).get('free', 0.0),
                'quote_total': balance.get(quote, {}).get('total', 0.0),
            }
            
            self.exchange_positions[symbol] = position
            self.last_position_sync = datetime.now()
            
            return position
        except Exception as e:
            self.position_sync_errors += 1
            logger.error(f"Position sync failed: {e}")
            return {}
    
    def place_market_order_with_retry(self, symbol: str, side: str, amount: float,
                                     expected_price: Optional[float] = None,
                                     retry_count: int = 0) -> Dict[str, Any]:
        """Place market order with comprehensive retry logic"""
        order_id = str(uuid.uuid4())
        start_time = time.time()
        
        order_result = {
            'order_id': order_id,
            'status': OrderStatus.FAILED.value,
            'exchange_order_id': None,
            'filled_amount': 0.0,
            'fill_price': None,
            'slippage': None,
            'execution_time_ms': 0.0,
            'error': None,
            'retry_count': retry_count,
            'partial_fills': []
        }
        
        try:
            with self._lock:
                self.total_orders += 1
                
                # Validate order
                if not self._validate_order(symbol, side, amount, expected_price):
                    raise Exception("Order validation failed")
                
                # Check slippage tolerance
                if expected_price:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    est_slippage = abs(current_price - expected_price) / expected_price
                    
                    if est_slippage > self.slippage_tolerance:
                        raise Exception(f"Slippage {est_slippage:.4%} exceeds tolerance")
                
                # Record as pending
                self.pending_orders[order_id] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'expected_price': expected_price,
                    'timestamp': datetime.now(),
                    'status': OrderStatus.SUBMITTED.value,
                    'retry_count': retry_count
                }
                
                logger.info(f"Placing {side} order: {amount:.6f} {symbol}")
                
                # Place order
                exchange_order = self.exchange.create_market_order(symbol, side, amount)
                exchange_order_id = exchange_order.get('id')
                order_result['exchange_order_id'] = exchange_order_id
                order_result['status'] = OrderStatus.SUBMITTED.value
                
                # Fetch fill details
                if exchange_order_id:
                    filled_order = self._get_order_status_with_retry(exchange_order_id, symbol)
                    
                    if filled_order:
                        order_result = self._process_order_fills(order_result, filled_order, expected_price)
                
                # Record execution time
                execution_time = (time.time() - start_time) * 1000
                order_result['execution_time_ms'] = execution_time
                self.execution_times.append(execution_time)
                
                # Update tracking
                self.pending_orders[order_id].update(order_result)
                
                if order_result['status'] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value]:
                    self.order_history.append(self.pending_orders[order_id])
                    del self.pending_orders[order_id]
                
                logger.info(f"Order {order_id}: {order_result['status']} in {execution_time:.1f}ms")
                return order_result
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            order_result['execution_time_ms'] = execution_time
            order_result['error'] = str(e)
            
            # Retry logic
            if retry_count < self.max_retries and self._should_retry(e):
                logger.warning(f"Order failed, retrying... ({retry_count + 1}/{self.max_retries})")
                backoff = min(2 ** retry_count, 10)
                time.sleep(backoff)
                return self.place_market_order_with_retry(symbol, side, amount, expected_price, retry_count + 1)
            
            # Final failure
            with self._lock:
                self.rejection_count += 1
                if order_id in self.pending_orders:
                    self.pending_orders[order_id]['status'] = OrderStatus.FAILED.value
                    self.order_history.append(self.pending_orders[order_id])
                    del self.pending_orders[order_id]
            
            logger.error(f"Order failed after {retry_count + 1} attempts: {e}")
            return order_result
    
    def _should_retry(self, error: Exception) -> bool:
        """Determine if error is retryable"""
        retry_types = [ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout]
        no_retry_types = [ccxt.AuthenticationError, ccxt.InsufficientFunds, ccxt.InvalidOrder]
        
        if any(isinstance(error, t) for t in no_retry_types):
            return False
        if any(isinstance(error, t) for t in retry_types):
            return True
        
        error_str = str(error).lower()
        return any(kw in error_str for kw in ['timeout', 'network', 'connection', 'temporary'])
    
    def _get_order_status_with_retry(self, order_id: str, symbol: str, max_retries: int = 3) -> Optional[Dict]:
        """Get order status with retry"""
        for attempt in range(max_retries):
            try:
                return self.exchange.fetch_order(order_id, symbol)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Order status fetch failed: {e}")
        return None
    
    def _process_order_fills(self, order_result: Dict, filled_order: Dict, 
                           expected_price: Optional[float]) -> Dict:
        """Process order fill information"""
        try:
            order_result['status'] = self._map_status(filled_order.get('status', 'unknown'))
            order_result['filled_amount'] = filled_order.get('filled', 0.0)
            order_result['fill_price'] = filled_order.get('average') or filled_order.get('price')
            
            # Calculate slippage
            if expected_price and order_result['fill_price']:
                slippage = abs(order_result['fill_price'] - expected_price) / expected_price
                order_result['slippage'] = slippage
                self.slippage_history.append(slippage)
            
            return order_result
        except Exception as e:
            logger.error(f"Error processing fills: {e}")
            return order_result
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            if order_id not in self.pending_orders:
                return False
            
            order = self.pending_orders[order_id]
            self.exchange.cancel_order(order['exchange_order_id'], order['symbol'])
            
            order['status'] = OrderStatus.CANCELLED.value
            self.order_history.append(order)
            del self.pending_orders[order_id]
            
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all pending orders"""
        count = 0
        for order_id in list(self.pending_orders.keys()):
            order = self.pending_orders[order_id]
            if symbol and order.get('symbol') != symbol:
                continue
            if self.cancel_order(order_id):
                count += 1
        return count
    
    def _validate_order(self, symbol: str, side: str, amount: float, 
                       expected_price: Optional[float]) -> bool:
        """Comprehensive order validation"""
        try:
            if side not in ['buy', 'sell'] or amount <= 0:
                return False
            
            if symbol not in self.exchange.markets:
                return False
            
            market = self.exchange.markets[symbol]
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            
            if amount < min_amount:
                logger.error(f"Amount below minimum: {amount} < {min_amount}")
                return False
            
            # Balance check for sell
            if side == 'sell':
                positions = self.synchronize_positions(symbol)
                available = positions.get('base_free', 0.0)
                if amount > available:
                    logger.error(f"Insufficient balance: {amount} > {available}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _map_status(self, exchange_status: str) -> str:
        """Map exchange status to OrderStatus"""
        mapping = {
            'open': OrderStatus.SUBMITTED.value,
            'closed': OrderStatus.FILLED.value,
            'canceled': OrderStatus.CANCELLED.value,
            'rejected': OrderStatus.REJECTED.value,
            'filled': OrderStatus.FILLED.value,
            'partial': OrderStatus.PARTIALLY_FILLED.value
        }
        return mapping.get(exchange_status.lower(), OrderStatus.FAILED.value)
    
    def monitor_partial_fills(self):
        """Monitor and complete partial fills"""
        completed = []
        for order_id, order in self.partial_fills.items():
            try:
                current = self._get_order_status_with_retry(order['exchange_order_id'], order['symbol'])
                if current and current.get('status') == 'closed':
                    order['status'] = OrderStatus.FILLED.value
                    self.order_history.append(order)
                    completed.append(order_id)
            except Exception as e:
                logger.error(f"Partial fill monitor error: {e}")
        
        for order_id in completed:
            del self.partial_fills[order_id]
    
    def get_execution_metrics(self) -> Dict[str, float]:
        """Get execution performance metrics"""
        with self._lock:
            network = self.network_manager.get_network_quality()
            return {
                'avg_execution_time_ms': np.mean(self.execution_times) if self.execution_times else 0.0,
                'avg_slippage': np.mean(self.slippage_history) if self.slippage_history else 0.0,
                'order_rejection_rate': self.rejection_count / max(1, self.total_orders),
                'pending_orders_count': len(self.pending_orders),
                'completed_orders_count': len(self.order_history),
                'network_quality_score': network['quality_score']
            }


# ============================================================================
# SECTION 4: CIRCUIT BREAKER AND RISK MANAGEMENT
# ============================================================================

class ProductionCircuitBreaker:
    """Advanced circuit breaker with multiple risk controls"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk thresholds
        self.max_daily_loss = config.get('max_daily_loss', 1000.0)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.10)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        self.max_hourly_trades = config.get('max_hourly_trades', 10)
        self.volatility_threshold = config.get('volatility_threshold', 0.05)
        self.max_slippage_threshold = config.get('max_slippage_threshold', 0.02)
        self.min_network_quality = config.get('min_network_quality', 0.7)
        
        # State tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.consecutive_errors = 0
        self.hourly_trades = deque(maxlen=100)
        self.recent_volatility = deque(maxlen=20)
        self.recent_slippage = deque(maxlen=10)
        
        # Circuit breaker state
        self.trading_halted = False
        self.halt_reason = None
        self.halt_timestamp = None
        self.last_reset = datetime.now().date()
    
    def comprehensive_risk_check(self, portfolio_value: float, initial_capital: float,
                                current_price: float, last_price: Optional[float] = None,
                                recent_slippage: Optional[float] = None,
                                network_quality: Optional[float] = None,
                                last_trade_profitable: Optional[bool] = None) -> Tuple[bool, str]:
        """Run comprehensive risk assessment"""
        
        self._check_daily_reset()
        
        # 1. Daily loss limit
        daily_pnl = portfolio_value - initial_capital
        if daily_pnl < -self.max_daily_loss:
            return self._trigger_halt("DAILY_LOSS", f"Loss ${abs(daily_pnl):.2f} exceeds ${self.max_daily_loss:.2f}")
        
        # 2. Maximum drawdown
        drawdown = max(0, (initial_capital - portfolio_value) / initial_capital)
        if drawdown > self.max_drawdown_pct:
            return self._trigger_halt("MAX_DRAWDOWN", f"Drawdown {drawdown:.2%} exceeds {self.max_drawdown_pct:.2%}")
        
        # 3. Consecutive losses
        if last_trade_profitable is not None:
            self.consecutive_losses = self.consecutive_losses + 1 if not last_trade_profitable else 0
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            return self._trigger_halt("CONSECUTIVE_LOSSES", f"{self.consecutive_losses} losses in a row")
        
        # 4. Trading frequency
        self.hourly_trades.append(datetime.now())
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_count = sum(1 for t in self.hourly_trades if t > hour_ago)
        
        if recent_count > self.max_hourly_trades:
            return self._trigger_halt("EXCESSIVE_TRADING", f"{recent_count} trades/hour exceeds {self.max_hourly_trades}")
        
        # 5. Market volatility
        if last_price and current_price:
            change = abs(current_price - last_price) / last_price
            self.recent_volatility.append(change)
            
            avg_vol = np.mean(self.recent_volatility)
            if avg_vol > self.volatility_threshold:
                return self._trigger_halt("HIGH_VOLATILITY", f"Volatility {avg_vol:.2%} exceeds {self.volatility_threshold:.2%}")
        
        # 6. Slippage monitoring
        if recent_slippage is not None:
            self.recent_slippage.append(recent_slippage)
            avg_slip = np.mean(self.recent_slippage)
            
            if avg_slip > self.max_slippage_threshold:
                return self._trigger_halt("HIGH_SLIPPAGE", f"Slippage {avg_slip:.2%} exceeds {self.max_slippage_threshold:.2%}")
        
        # 7. Network quality
        if network_quality is not None and network_quality < self.min_network_quality:
            return self._trigger_halt("POOR_NETWORK", f"Network quality {network_quality:.2f} below {self.min_network_quality:.2f}")
        
        # 8. System errors
        if self.consecutive_errors >= 5:
            return self._trigger_halt("SYSTEM_ERRORS", f"{self.consecutive_errors} consecutive errors")
        
        return not self.trading_halted, self.halt_reason or "Active"
    
    def _check_daily_reset(self):
        """Reset daily counters"""
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.hourly_trades.clear()
            self.last_reset = current_date
            
            if self.trading_halted and self.halt_reason != "EMERGENCY_STOP":
                self.resume_trading("New day auto-resume")
    
    def _trigger_halt(self, reason: str, message: str) -> Tuple[bool, str]:
        """Trigger circuit breaker"""
        if not self.trading_halted:
            self.trading_halted = True
            self.halt_reason = reason
            self.halt_timestamp = datetime.now()
            logger.critical(f"CIRCUIT BREAKER: {reason} - {message}")
        return False, f"{reason}: {message}"
    
    def log_system_error(self):
        """Log system error"""
        self.consecutive_errors += 1
    
    def log_system_success(self):
        """Reset error counter"""
        self.consecutive_errors = 0
    
    def emergency_stop(self, reason: str = "Emergency"):
        """Manual emergency stop"""
        self._trigger_halt("EMERGENCY_STOP", reason)
    
    def resume_trading(self, reason: str = "Manual"):
        """Resume trading"""
        if self.trading_halted:
            logger.warning(f"Resuming: {reason} (was: {self.halt_reason})")
            self.trading_halted = False
            self.halt_reason = None
            self.halt_timestamp = None
            self.consecutive_errors = 0
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get risk status"""
        return {
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'halt_timestamp': self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_errors': self.consecutive_errors
        }


# ============================================================================
# SECTION 5: PRODUCTION LIVE TRADER
# ============================================================================

class ProductionLiveTrader:
    """Production-ready live trader with comprehensive safety systems"""
    
    def __init__(self, config: Dict[str, Any], model: Any, ingestion_manager: Any = None):
        """Initialize production trader"""
        
        # Core components
        self.config = config.get('live_trading', {})
        self.model = model
        self.data_processor = DataProcessor(config)
        self.ingestion_manager = ingestion_manager
        
        # Trading state
        self.trading_active = False
        self.position = 0.0
        self.cash = self.config.get('initial_capital', 10000)
        self.initial_capital = self.cash
        
        self.current_portfolio_value = self.cash  # ADD THIS LINE


        # Tracking
        self.trade_history: List[TradeRecord] = []
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_successful_trade = None
        self.session_start_time = None
        
        # Production components
        self.paper_trading = self.config.get('paper_trading', True)
        self.security_manager = SecurityManager()
        self.order_manager = None
        self.circuit_breaker = None
        self.market_schedule = None
        
        # Risk parameters
        self.max_position_size = self.config.get('max_position_size', 0.8)
        self.transaction_fee_rate = self.config.get('transaction_fee_rate', 0.001)
        self.stop_loss_threshold = self.config.get('stop_loss_threshold', 0.05)
        self.min_trade_interval = self.config.get('min_trade_interval_seconds', 30)
        
        # Performance tracking
        self.peak_portfolio_value = self.cash
        self.max_drawdown = 0.0
        self.portfolio_values = deque(maxlen=2000)
        self.returns = deque(maxlen=2000)
        self.price_history = deque(maxlen=100)
        self.last_price = None
        
        # Output
        self.output_dir = self.config.get('output_directory', './trading_outputs')
        self.save_trades = self.config.get('save_trades', True)
        self.save_performance = self.config.get('save_performance', True)
        
        # Threading
        self._lock = threading.RLock()
        self._performance_timer = None
        self._monitoring_active = False
        
        # Initialize
        self._initialize_all_systems()
        
        if not self.ingestion_manager:
            raise ValueError("ingestion_manager required")
        
        # Initialize performance tracker if available
        self.performance_tracker = None
        if PerformanceTracker and self.config.get('model_retraining', {}).get('enabled', False):
            tracker_config = {'model_retraining': self.config.get('model_retraining', {})}
            self.performance_tracker = PerformanceTracker(
                config=tracker_config,
                output_dir=os.path.join(self.output_dir, 'monitoring')
            )
            logger.info("Performance tracking enabled")
            
        mode = "LIVE" if not self.paper_trading else "PAPER"
        logger.info(f"Production trader initialized - {mode} mode")
    
    def _initialize_all_systems(self):
        """Initialize all systems"""
        try:
            self._setup_directories()
            self._validate_model()
            
            # Live trading only
            if not self.paper_trading:
                exchange_config = self.config.get('exchange', {})
                if not exchange_config:
                    raise ValueError("Exchange config required for live trading")
                
                self.order_manager = EnhancedOrderManager(exchange_config, self.security_manager)
                if not self.order_manager.initialize_exchange():
                    raise Exception("Exchange initialization failed")
                
                self.market_schedule = MarketScheduleManager(exchange_config.get('name', 'binance'))
            
            # Circuit breaker
            risk_config = self.config.get('risk_management', {})
            self.circuit_breaker = ProductionCircuitBreaker(risk_config)
            
            logger.info("All systems initialized")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def _setup_directories(self):
        """Setup output directories"""
        try:
            dirs = [
                self.output_dir,
                os.path.join(self.output_dir, 'trades'),
                os.path.join(self.output_dir, 'performance'),
                os.path.join(self.output_dir, 'monitoring')
            ]
            for d in dirs:
                os.makedirs(d, exist_ok=True)
        except Exception as e:
            logger.warning(f"Directory setup failed: {e}")
    
    def _validate_model(self):
        """Validate model"""
        if isinstance(self.model, dict) and callable(self.model.get('predict')):
            self._predict_method = self.model.get('predict')
        elif hasattr(self.model, 'predict') and callable(self.model.predict):
            self._predict_method = self.model.predict
        else:
            raise AttributeError("Model missing predict method")
    
    def start_trading(self):
        """Start trading session"""
        with self._lock:
            if self.trading_active:
                logger.warning("Already active")
                return
            
            # Pre-flight checks
            if not self.paper_trading:
                if self.market_schedule and not self.market_schedule.is_market_open():
                    logger.error("Market closed")
                    return
                
                if not self.order_manager or not self.order_manager.exchange:
                    logger.error("Exchange not initialized")
                    return
            
            self.trading_active = True
            self.session_start_time = datetime.now()
            self.circuit_breaker.log_system_success()
        
        mode = "LIVE" if not self.paper_trading else "PAPER"
        logger.info(f"=== {mode} SESSION STARTED ===")
        logger.info(f"Capital: ${self.initial_capital:,.2f}")
        
        if not self.paper_trading:
            logger.warning("⚠️  LIVE TRADING - REAL MONEY AT RISK ⚠️")
    
    def stop_trading(self):
        """Stop trading"""
        with self._lock:
            if not self.trading_active:
                return
            self.trading_active = False
        
        logger.info("=== STOPPING SESSION ===")
        
        # Cancel orders
        if not self.paper_trading and self.order_manager:
            try:
                count = self.order_manager.cancel_all_orders()
                logger.info(f"Cancelled {count} orders")
            except Exception as e:
                logger.error(f"Cancel failed: {e}")
        
        # Final report
        self._generate_session_report()
        logger.info("Session stopped")
    
    def emergency_stop(self, reason: str = "Emergency"):
        """Emergency stop"""
        logger.critical(f"🚨 EMERGENCY STOP: {reason}")
        
        if self.circuit_breaker:
            self.circuit_breaker.emergency_stop(reason)
        
        if not self.paper_trading and self.order_manager:
            try:
                self.order_manager.cancel_all_orders()
            except Exception as e:
                logger.critical(f"Emergency cancel failed: {e}")
        
        self.stop_trading()
    
    def run_trading_loop(self, symbol: str, data_type: str):
        """Main trading loop"""
        logger.info("Starting trading loop")
        iteration = 0
        
        while self.trading_active:
            iteration += 1
            
            try:
                with self._error_handling(f"iteration_{iteration}"):
                    
                    # Market check
                    if not self.paper_trading and self.market_schedule:
                        if not self.market_schedule.is_market_open():
                            time.sleep(300)
                            continue
                    
                    # Fetch data
                    current_price, features = self._fetch_and_process(symbol, data_type)
                    if current_price is None:
                        continue
                    
                    self.price_history.append(current_price)
                    
                    # Risk check
                    network_quality = None
                    if not self.paper_trading and self.order_manager:
                        network_quality = self.order_manager.network_manager.get_network_quality()['quality_score']
                    
                    allowed, msg = self.circuit_breaker.comprehensive_risk_check(
                        portfolio_value=self._calc_value(current_price),
                        initial_capital=self.initial_capital,
                        current_price=current_price,
                        last_price=self.last_price,
                        network_quality=network_quality
                    )
                    
                    if not allowed:
                        logger.warning(f"Trading blocked: {msg}")
                        time.sleep(60)
                        continue
                    
                    # Trade
                    prediction, confidence = self._get_prediction(features)
                    
                    if self._should_trade(prediction, current_price):
                        self._execute_trade(prediction, current_price, confidence, features, symbol)
                    
                    # Update
                    self._update_tracking(current_price)
                    self.last_price = current_price
                    
                    # Monitor partials
                    if not self.paper_trading and self.order_manager:
                        self.order_manager.monitor_partial_fills()
                    
                    self.circuit_breaker.log_system_success()
                    
                    if iteration % 10 == 0:
                        self._log_status(current_price)
            
            except Exception as e:
                self.circuit_breaker.log_system_error()
                logger.error(f"Loop error: {e}")
                time.sleep(5)
            
            time.sleep(self.config.get('polling_interval', 60))
    
    def _fetch_and_process(self, symbol: str, data_type: str) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
        """Fetch and process data"""
        try:
            end = datetime.now()
            start = end - timedelta(minutes=2)
            
            data = self.ingestion_manager.fetch_single_token(symbol, start, end, data_type)
            
            if data is None or data.empty:
                return None, None
            
            # Preprocess
            if 'side' in data.columns:
                data['side'] = data['side'].map({'ask': 'a', 'bid': 'b'})
                data = data.rename(columns={'side': 'type'})
            
            if 'timestamp_ms' in data.columns:
                data.drop('timestamp_ms', axis=1, inplace=True)
            
            processed = self.data_processor.process_data(data, data_type=data_type)
            
            if processed is None or processed.empty:
                return None, None
            
            features = processed.drop(columns=['price'], errors='ignore')
            price = float(processed['price'].iloc[0])
            
            if not self._valid_price(price):
                return None, None
            
            return price, features
        except Exception as e:
            logger.error(f"Fetch/process error: {e}")
            return None, None
    
    def _execute_trade(self, prediction: int, price: float, confidence: Optional[float],
                      features: Optional[pd.DataFrame], symbol: str):
        """Execute trade"""
        
        feat_len = len(features) if features is not None else 951
        is_sell = prediction > (feat_len * 0.8)
        is_buy = prediction < (feat_len * 0.2)
        
        action = "HOLD"
        quantity = 0.0
        order_result = None
        
        with self._lock:
            # SELL
            if is_sell and self.position > 0:
                action = "SELL"
                quantity = self.position
                
                if not self.paper_trading and self.order_manager:
                    order_result = self.order_manager.place_market_order_with_retry(
                        symbol, 'sell', quantity, price
                    )
                    
                    if order_result['status'] == OrderStatus.FILLED.value:
                        fill_price = order_result.get('fill_price', price)
                        fill_qty = order_result.get('filled_amount', quantity)
                        
                        value = fill_qty * fill_price
                        fee = value * self.transaction_fee_rate
                        
                        self.cash += value - fee
                        self.position -= fill_qty
                        
                        logger.info(f"✓ SELL: {fill_qty:.6f} @ ${fill_price:.4f}")
                    else:
                        action = "HOLD"
                else:
                    # Paper
                    value = self.position * price
                    fee = value * self.transaction_fee_rate
                    self.cash += value - fee
                    self.position = 0
                    logger.info(f"PAPER SELL: {quantity:.6f} @ ${price:.4f}")
            
            # BUY
            elif is_buy and self.position == 0 and self.cash > 0:
                max_buy = min(self.cash, self.initial_capital * self.max_position_size)
                
                if max_buy > (price * 0.001):
                    action = "BUY"
                    
                    if not self.paper_trading and self.order_manager:
                        net_value = max_buy / (1 + self.transaction_fee_rate)
                        quantity = net_value / price
                        
                        order_result = self.order_manager.place_market_order_with_retry(
                            symbol, 'buy', quantity, price
                        )
                        
                        if order_result['status'] == OrderStatus.FILLED.value:
                            fill_price = order_result.get('fill_price', price)
                            fill_qty = order_result.get('filled_amount', quantity)
                            
                            cost = fill_qty * fill_price
                            fee = cost * self.transaction_fee_rate
                            
                            self.position += fill_qty
                            self.cash -= (cost + fee)
                            
                            logger.info(f"✓ BUY: {fill_qty:.6f} @ ${fill_price:.4f}")
                        else:
                            action = "HOLD"
                    else:
                        # Paper
                        net_value = max_buy / (1 + self.transaction_fee_rate)
                        quantity = net_value / price
                        fee = net_value * self.transaction_fee_rate
                        
                        self.position = quantity
                        self.cash -= (net_value + fee)
                        logger.info(f"PAPER BUY: {quantity:.6f} @ ${price:.4f}")
            
            # Record
            if action in ["BUY", "SELL"]:
                self.last_successful_trade = datetime.now()
                
                trade = TradeRecord(
                    timestamp=datetime.now().isoformat(),
                    action=action,
                    price=price,
                    quantity=quantity,
                    cash_after=self.cash,
                    position_after=self.position,
                    prediction=prediction,
                    confidence=confidence,
                    features_hash=self._hash(features),
                    order_id=order_result.get('order_id') if order_result else None,
                    exchange_order_id=order_result.get('exchange_order_id') if order_result else None,
                    fill_price=order_result.get('fill_price') if order_result else price,
                    slippage=order_result.get('slippage') if order_result else None,
                    execution_time_ms=order_result.get('execution_time_ms') if order_result else None,
                    order_status=order_result.get('status') if order_result else 'SIMULATED'
                )
                
                self.trade_history.append(trade)
                
                if self.save_trades:
                    self._save_trade(trade)
                    
            # Record prediction for performance tracking
            if self.performance_tracker and action in ["BUY", "SELL"]:
                trade_id = trade.trade_id if hasattr(trade, 'trade_id') else trade.timestamp
                self.performance_tracker.record_prediction(
                    trade_id=str(trade_id),
                    prediction=prediction,
                    predicted_direction=action,
                    confidence=confidence,
                    entry_price=price,
                    execution_latency_ms=order_result.get('execution_time_ms') if order_result else None,
                    features=features
                )
                
    
    def close_position_with_tracking(self, current_price: float):
        """Close position and record outcome for performance tracking"""
        if self.position == 0:
            return
        
        # Get last trade that opened this position
        last_trade = None
        for trade in reversed(self.trade_history):
            if trade.action == "BUY" and trade.position_after > 0:
                last_trade = trade
                break
        
        if last_trade and self.performance_tracker:
            # Calculate outcome
            actual_return = (current_price - last_trade.price) / last_trade.price
            prediction_correct = actual_return > 0 if last_trade.action == "BUY" else actual_return < 0
            
            # Record trade close
            self.performance_tracker.record_trade_close(
                trade_id=str(last_trade.timestamp),
                exit_price=current_price,
                slippage_pct=last_trade.slippage
            )
            
            
    def _get_prediction(self, features) -> Tuple[int, Optional[float]]:
        """Get model prediction"""
        try:
            result = self._predict_method(features)
            if len(result) == 0:
                return 0, None
            
            prediction = result[0]
            confidence = None
            
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(features)
                    if len(proba) > 0:
                        confidence = float(np.max(proba[0]))
                except:
                    pass
            
            return int(prediction), confidence
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, None
    
    def _should_trade(self, prediction: int, price: float) -> bool:
        """Check if should trade"""
        if self.circuit_breaker and self.circuit_breaker.trading_halted:
            return False
        
        if self.last_successful_trade:
            elapsed = (datetime.now() - self.last_successful_trade).total_seconds()
            if elapsed < self.min_trade_interval:
                return False
        
        value = self._calc_value(price)
        if value < self.initial_capital * (1 - self.stop_loss_threshold):
            if self.position > 0:
                return True
            return False
        
        return True
    
    def _calc_value(self, price: float) -> float:
        """Calculate portfolio value"""
        return self.cash + (self.position * price)
    
    def _update_tracking(self, price: float):
        """Update tracking"""
        value = self._calc_value(price)
        self.portfolio_values.append(value)
        
        if value > self.peak_portfolio_value:
            self.peak_portfolio_value = value
        
        drawdown = (self.peak_portfolio_value - value) / self.peak_portfolio_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        if len(self.portfolio_values) > 1:
            ret = (value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns.append(ret)
    
    def _log_status(self, price: float):
        """Log status"""
        value = self._calc_value(price)
        ret = (value - self.initial_capital) / self.initial_capital
        
        logger.info(f"Portfolio: ${value:,.2f} | Return: {ret:+.2%} | Trades: {len(self.trade_history)}")
    
    def _valid_price(self, price: float) -> bool:
        """Validate price"""
        return isinstance(price, (int, float)) and not np.isnan(price) and price > 0
    
    def _hash(self, features: pd.DataFrame) -> str:
        """Hash features"""
        try:
            return str(hash(pd.util.hash_pandas_object(features).sum()))
        except:
            return None
    
    def _save_trade(self, trade: TradeRecord):
        """Save trade"""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.output_dir, 'trades', f'trade_{ts}.json')
            with open(path, 'w') as f:
                json.dump(asdict(trade), f, indent=2)
        except Exception as e:
            logger.warning(f"Save failed: {e}")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics"""
        total = len([t for t in self.trade_history if t.action in ["BUY", "SELL"]])
        
        if total == 0:
            return PerformanceMetrics()
        
        value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        total_return = (value - self.initial_capital) / self.initial_capital
        
        successful = 0
        returns = []
        
        for i, trade in enumerate(self.trade_history):
            if trade.action == "SELL" and i > 0:
                for j in range(i-1, -1, -1):
                    if self.trade_history[j].action == "BUY":
                        ret = (trade.price - self.trade_history[j].price) / self.trade_history[j].price
                        returns.append(ret)
                        if ret > 0:
                            successful += 1
                        break
        
        win_rate = successful / len(returns) if returns else 0
        avg_return = np.mean(returns) if returns else 0
        volatility = np.std(list(self.returns)) if len(self.returns) > 1 else 0
        sharpe = (np.mean(list(self.returns)) / volatility) if volatility > 0 else 0
        
        fees = sum(t.price * t.quantity * self.transaction_fee_rate 
                  for t in self.trade_history if t.action in ["BUY", "SELL"])
        
        return PerformanceMetrics(
            total_trades=total,
            successful_trades=successful,
            total_return=total_return,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            avg_trade_return=avg_return,
            volatility=volatility,
            total_fees=fees
        )
    
    def _generate_session_report(self):
        """Generate session report"""
        try:
            duration = (datetime.now() - self.session_start_time).total_seconds() / 60 if self.session_start_time else 0
            metrics = self.get_performance_metrics()
            value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
            
            logger.info("=" * 60)
            logger.info(f"{'LIVE' if not self.paper_trading else 'PAPER'} SESSION REPORT")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration:.1f} minutes")
            logger.info(f"Initial: ${self.initial_capital:,.2f}")
            logger.info(f"Final: ${value:,.2f}")
            logger.info(f"Return: {metrics.total_return:+.2%}")
            logger.info(f"Drawdown: {metrics.max_drawdown:.2%}")
            logger.info(f"Trades: {metrics.total_trades} | Win Rate: {metrics.win_rate:.1%}")
            logger.info(f"Fees: ${metrics.total_fees:.2f}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Report error: {e}")
    
    # def get_current_status(self) -> Dict[str, Any]:
    #     """Get current status"""
    #     value = self.current_portfolio_value if hasattr(self, 'current_portfolio_value') else \
    #             (self.portfolio_values[-1] if self.portfolio_values else self.initial_capital)
    #     
    #     return {
    #         'active': self.trading_active,
    #         'mode': 'LIVE' if not self.paper_trading else 'PAPER',
    #         'value': value,
    #         'current_portfolio_value': value,  # ADD explicit key for monitoring
    #         'return_pct': ((value - self.initial_capital) / self.initial_capital) * 100,
    #         'position': self.position,
    #         'cash': self.cash,
    #         'trades': len(self.trade_history)
    #     }
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status"""
        value = self.current_portfolio_value if hasattr(self, 'current_portfolio_value') else \
                (self.portfolio_values[-1] if self.portfolio_values else self.initial_capital)
        
        return {
            'active': self.trading_active,
            'mode': 'LIVE' if not self.paper_trading else 'PAPER',
            'value': value,
            'current_portfolio_value': value,
            'return_pct': ((value - self.initial_capital) / self.initial_capital) * 100,
            'position': self.position,
            'cash': self.cash,
            'trades': len(self.trade_history),
            'total_trades': len(self.trade_history),
            'max_drawdown': self.max_drawdown,
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'session_duration_minutes': self._get_session_duration_minutes(),
            'last_trade_time': self.trade_history[-1].timestamp if self.trade_history else None
        }
        
    def _get_session_duration_minutes(self) -> float:
        """Get current session duration in minutes"""
        if self.session_start_time:
            duration_seconds = (datetime.now() - self.session_start_time).total_seconds()
            return duration_seconds / 60
        return 0.0
      
      
    @contextmanager
    def _error_handling(self, operation: str):
        """Error handling context"""
        try:
            yield
            self.consecutive_errors = 0
        except Exception as e:
            self.error_count += 1
            self.consecutive_errors += 1
            logger.error(f"Error in {operation}: {e}")
            
            if self.consecutive_errors >= 5:
                self.circuit_breaker.emergency_stop("Excessive errors")
                self.stop_trading()
            raise
      

    
    def export_trade_history(self, filename: Optional[str] = None) -> str:
        """
        Export complete trade history to CSV file.
        
        Args:
            filename: Optional custom filename. If None, auto-generates timestamp-based name.
            
        Returns:
            str: Full path to the exported CSV file.
        """
        if not filename:
            filename = f'trade_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        if self.trade_history:
            # Convert trade records to DataFrame
            df = pd.DataFrame([asdict(trade) for trade in self.trade_history])
            df.to_csv(filepath, index=False)
            logger.info(f"Trade history exported: {filepath} ({len(df)} trades)")
        else:
            logger.warning("No trade history to export")
            # Create empty CSV with headers
            empty_df = pd.DataFrame(columns=[f.name for f in TradeRecord.__dataclass_fields__.values()])
            empty_df.to_csv(filepath, index=False)
        
        return filepath
      

    def get_live_metrics_summary(self) -> Dict[str, Any]:
        """Get condensed live metrics summary for monitoring"""
        current_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        total_trades = len([t for t in self.trade_history if t.action in ["BUY", "SELL"]])
        
        # Calculate recent win rate (last 10 trades)
        recent_trades = []
        if len(self.trade_history) >= 2:
            for i in range(len(self.trade_history) - 1, max(-1, len(self.trade_history) - 21), -1):
                if self.trade_history[i].action == "SELL":
                    for j in range(i-1, -1, -1):
                        if self.trade_history[j].action == "BUY":
                            trade_return = ((self.trade_history[i].price - self.trade_history[j].price) 
                                          / self.trade_history[j].price)
                            recent_trades.append(trade_return)
                            break
        
        recent_win_rate = sum(1 for r in recent_trades if r > 0) / len(recent_trades) if recent_trades else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'trading_active': self.trading_active,
            'trading_mode': 'LIVE' if not self.paper_trading else 'PAPER',
            'portfolio_value': current_value,
            'total_return_pct': ((current_value - self.initial_capital) / self.initial_capital) * 100,
            'position': self.position,
            'cash': self.cash,
            'total_trades': total_trades,
            'recent_win_rate_pct': recent_win_rate * 100,
            'max_drawdown_pct': self.max_drawdown * 100,
            'session_minutes': (datetime.now() - self.session_start_time).total_seconds() / 60 if self.session_start_time else 0,
            'consecutive_errors': self.consecutive_errors,
            'last_trade': self.trade_history[-1].timestamp if self.trade_history else None
        }
        
        # Add live trading metrics
        if not self.paper_trading and self.order_manager:
            exec_metrics = self.order_manager.get_execution_metrics()
            summary.update({
                'avg_execution_time_ms': exec_metrics['avg_execution_time_ms'],
                'avg_slippage_pct': exec_metrics['avg_slippage'] * 100,
                'order_rejection_rate_pct': exec_metrics['order_rejection_rate'] * 100,
                'pending_orders': exec_metrics['pending_orders_count'],
                'network_quality': exec_metrics['network_quality_score']
            })
        
        return summary
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value (cash + position value)"""
        value = self.cash + (self.position * current_price)
        # Store for monitoring access
        self.current_portfolio_value = value  # ADD THIS LINE
        return value
    
    
    

# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

LiveTrader = ProductionLiveTrader


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example:
    
    config = load_config('config.yaml')
    trader = ProductionLiveTrader(config, model, ingestion_manager)
    trader.start_trading()
    trader.run_trading_loop(symbol='BTC/USDT', data_type='tradebook')
    """
    pass
 
 
