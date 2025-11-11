import pandas as pd
import ccxt
import os
import gzip
import tarfile
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
import gdown


def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying failed requests with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                except Exception as e:
                    # Don't retry for non-network errors
                    logger.error(f"Non-retryable error: {e}")
                    raise e
            raise last_exception
        return wrapper
    return decorator


class IngestionManager:
    """
    Manages data ingestion from multiple sources, including exchanges, local files,
    and external APIs.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the IngestionManager with the project configuration.
        
        Args:
            config (Dict[str, Any]): The data ingestion configuration section
        """
        # FIXED: Accept the full config, then extract data_ingestion section
        if 'data_ingestion' in config:
            self.config = config['data_ingestion']
            self.full_config = config
        else:
            self.config = config
            self.full_config = config
      
        # Add these missing attributes:
        self.use_google_drive = self.config.get('google_drive', {}).get('enabled', False)
         
        # Import gdown check
        try:
           import gdown
           import zipfile
           import shutil
           import random
           self.gdown = gdown
           self.zipfile = zipfile
           self.shutil = shutil
           self.random = random
        except ImportError:
           self.use_google_drive = False
           logger.warning("Google Drive dependencies not available")
        
        logger.info("IngestionManager initialized.")
        logger.debug(f"Config keys: {list(self.config.keys())}")
  
         

    def fetch_historical_data(self, start_time: datetime, end_time: datetime, data_type: str, source_type: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical data for all configured tokens over a specified period.
        
        Args:
            start_time (datetime): The start of the time range for data retrieval.
            end_time (datetime): The end of the time range for data retrieval.
            data_type (str): The type of data to fetch ('tradebook' or 'orderbook').

        Returns:
            Optional[pd.DataFrame]: A concatenated DataFrame containing all fetched data.
        """
        logger.info(f"Starting data fetch for {data_type} from {start_time} to {end_time}")
        
        all_data = []
        tokens = self.config.get('tokens_to_monitor', [])
        
        if not tokens:
            logger.error("No tokens configured in 'tokens_to_monitor'. Check your configuration.")
            return None
        
        logger.info(f"Found {len(tokens)} tokens to process: {[t.get('symbol') for t in tokens]}")
        
        for token_info in tokens:
            symbol = token_info.get('symbol')
            source = token_info.get('source')
            
            if not symbol:
                logger.warning(f"Token missing 'symbol' field: {token_info}")
                continue
                
            if not source:
                logger.warning(f"Token {symbol} missing 'source' field")
                continue

            # FIXED: Look up source type correctly
            # source_type = self._get_source_type(source)
            logger.info(f"Processing {symbol} from {source} (type: {source_type})")

            # Route the request based on the data source type
            try:
                if source_type == 'exchange':
                    # FIXED: Pass the actual symbol and parameters correctly
                    data = self._fetch_from_exchange(symbol, start_time, end_time, data_type)
                    
                elif source_type == 'local_file':
                    data = self._fetch_from_local_file(symbol, start_time, end_time, data_type)
                
                elif source_type == 'google_drive':
                    gdrive_config = token_info.get('gdrive_config', {})
                    gdrive_url = gdrive_config.get('url') or self.config.get('google_drive', {}).get('default_url')
                    sample_ratio = gdrive_config.get('sample_ratio', 0.0001)
                    
                    if gdrive_url:
                        data = self._fetch_from_google_drive(gdrive_url, symbol, start_time, end_time, data_type, sample_ratio)
                    else:
                        logger.error(f"No Google Drive URL configured for {symbol}")
                        continue
                  
                else:
                    logger.warning(f"Unsupported source type '{source_type}' for symbol '{symbol}'. Skipping.")
                    continue
                
                if data is not None and not data.empty:
                    all_data.append(data)
                    logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} from {source}: {e}")
                continue
        
        if not all_data:
            logger.error("No data was fetched from any source.")
            return None
        
        # Combine all data into a single DataFrame
        logger.info(f"Combining data from {len(all_data)} sources...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp if 'date' column exists
        if 'date' in combined_df.columns:
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            
        logger.info(f"Combined dataset has {len(combined_df)} total records")
        return combined_df

    def _get_source_type(self, source: str) -> str:
        """
        Determine the source type based on configuration.
        
        Args:
            source (str): The source identifier
            
        Returns:
            str: The source type ('exchange', 'local_file', etc.)
        """
        # Check in data_sources first
        data_sources = self.config.get('data_sources', {})
        if source in data_sources:
            source_value = data_sources[source]
            # If it's a known exchange name, it's an exchange
            if source_value in ['binance', 'coinbase', 'kraken', 'bybit', 'okx']:
                return 'exchange'
            # If it contains 'exchange' in the name
            elif 'exchange' in str(source_value).lower():
                return 'exchange'
            # g drive
            if source_value == 'google_drive':  # Add this exact check
                return 'google_drive'
         
            # If it's a file path or 'local'
            elif 'local' in str(source_value).lower() or 'file' in str(source_value).lower():
                return 'local_file'
            else:
                return str(source_value)
        
        # Fallback: assume it's an exchange if it's a known exchange name
        known_exchanges = ['binance', 'coinbase', 'kraken', 'bybit', 'okx', 'bitfinex']
        if source.lower() in known_exchanges:
            return 'exchange'
            
        # Default fallback
        logger.warning(f"Could not determine source type for '{source}', defaulting to 'exchange'")
        return 'exchange'

    def _fetch_from_local_file(self, symbol: str, start_time: datetime, end_time: datetime, data_type: str) -> Optional[pd.DataFrame]:
        """
        Fetches compressed data from local file structure with improved error handling.
        """
        logger.info(f"Fetching {data_type} for {symbol} from local files")
        
        # Construct the file path based on the directory structure
        base_path = self.config.get('local_data_path', 'data/raw')
        data_path = Path(base_path) / data_type
        
        if not data_path.exists():
            logger.warning(f"Data directory does not exist: {data_path}")
            return None
        
        # Generate date range
        dates = pd.date_range(start_time.date(), end_time.date(), freq='D')
        logger.info(f"Searching for files across {len(dates)} days")
        
        all_data = []
        files_processed = 0
        
        for date in dates:
            date_str = date.strftime('%Y_%m_%d')
            
            # Look for various file patterns
            patterns = [
                f"*{symbol.replace('/', '')}*{data_type}*{date_str}*.tar*",
                f"*{symbol.replace('/', '')}*{date_str}*.tar*",
                f"*{data_type}*{date_str}*.tar*",
                f"*{symbol.replace('/', '')}*{data_type}*{date_str}*.csv.gz",
                f"*{symbol.replace('/', '')}*{date_str}*.csv.gz",
            ]
            
            files_found = []
            for pattern in patterns:
                files_found.extend(list(data_path.glob(pattern)))
            
            # Remove duplicates
            files_found = list(set(files_found))
            
            if not files_found:
                logger.debug(f"No files found for {symbol} on {date_str}")
                continue
                
            logger.info(f"Found {len(files_found)} files for {date_str}: {[f.name for f in files_found]}")
            
            for file_path in files_found:
                try:
                    df = self._process_file(file_path, symbol, data_type)
                    if df is not None and not df.empty:
                        all_data.append(df)
                        files_processed += 1
                        logger.info(f"Successfully processed {file_path.name} - {len(df)} records")
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
        
        logger.info(f"Processed {files_processed} files successfully")
        
        if not all_data:
            logger.warning(f"No data found for {symbol}")
            return None
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Filter by time range if date column exists
        if 'date' in combined_df.columns:
            mask = (combined_df['date'] >= start_time) & (combined_df['date'] <= end_time)
            combined_df = combined_df[mask]
            
        logger.info(f"Final dataset for {symbol}: {len(combined_df)} records")
        return combined_df

    def _process_file(self, file_path: Path, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """
        Process a single file (tar, tar.gz, or csv.gz).
        """
        try:
            if file_path.suffix in ['.tar'] or '.tar' in file_path.name:
                return self._process_tar_file(file_path, symbol, data_type)
            elif file_path.suffix == '.gz' and '.csv' in file_path.name:
                return self._process_csv_gz_file(file_path, symbol, data_type)
            else:
                logger.warning(f"Unknown file type: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def _process_tar_file(self, file_path: Path, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """Process tar or tar.gz files containing CSV data."""
        all_data = []
        
        with tarfile.open(file_path, 'r:*') as tar:
            # Find CSV or CSV.gz files in the tar
            csv_members = [m for m in tar.getmembers() 
                          if (m.name.endswith('.csv.gz') or m.name.endswith('.csv')) and m.isfile()]
            
            logger.debug(f"Found {len(csv_members)} CSV files in {file_path.name}")
            
            for member in csv_members:
                try:
                    extracted_file = tar.extractfile(member)
                    if extracted_file:
                        if member.name.endswith('.csv.gz'):
                            # Decompress gzip content
                            with gzip.open(extracted_file, 'rt') as f:
                                df = pd.read_csv(f)
                        else:
                            # Direct CSV
                            df = pd.read_csv(extracted_file)
                        
                        df = self._standardize_dataframe(df, symbol, data_type, 'local_file')
                        if df is not None:
                            all_data.append(df)
                            
                except Exception as e:
                    logger.error(f"Error processing {member.name} from {file_path}: {e}")
                    continue
        
        return pd.concat(all_data, ignore_index=True) if all_data else None

    def _process_csv_gz_file(self, file_path: Path, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """Process individual CSV.gz files."""
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f)
        
        return self._standardize_dataframe(df, symbol, data_type, 'local_file')

    def _standardize_dataframe(self, df: pd.DataFrame, symbol: str, data_type: str, source: str) -> Optional[pd.DataFrame]:
        """
        Standardize column names and formats across different data sources.
        """
        if df is None or df.empty:
            return None
            
        try:
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Add metadata columns
            df['source'] = source
            df['symbol'] = symbol
            
            # Standardize column names based on data type
            if data_type == 'tradebook':
                # Common tradebook column mappings
                column_mapping = {
                    'timestamp': 'timestamp',
                    'time': 'timestamp', 
                    'date': 'timestamp',
                    'amount': 'volume',
                    'qty': 'volume',
                    'quantity': 'volume',
                    'size': 'volume'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    # Try different timestamp formats
                    if df['timestamp'].dtype == 'int64':
                        # Assume Unix timestamp - try both seconds and milliseconds
                        if df['timestamp'].max() > 1e10:  # Likely milliseconds
                            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                        else:  # Likely seconds
                            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                    else:
                        df['date'] = pd.to_datetime(df['timestamp'])
                        
            elif data_type == 'orderbook':
                # Common orderbook column mappings
                column_mapping = {
                    'timestamp': 'timestamp',
                    'time': 'timestamp',
                    'date': 'timestamp',
                    'amount': 'volume',
                    'qty': 'volume',
                    'quantity': 'volume',
                    'size': 'volume',
                    'type': 'side',
                    'order_type': 'side'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    if df['timestamp'].dtype == 'int64':
                        # Orderbook data often uses milliseconds
                        if df['timestamp'].max() > 1e10:  # Likely milliseconds
                            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                        else:  # Likely seconds
                            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                    else:
                        df['date'] = pd.to_datetime(df['timestamp'])
            
            # Ensure required columns exist
            required_cols = ['date', 'symbol', 'price', 'volume', 'source']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing required columns {missing_cols} in data for {symbol}")
                # Try to create missing columns with defaults
                for col in missing_cols:
                    if col not in df.columns:
                        df[col] = None
                        
            # Select only the columns we need
            available_cols = [col for col in required_cols if col in df.columns]
            if 'side' in df.columns:
                available_cols.append('side')
            if 'level' in df.columns:
                available_cols.append('level')
                
            df = df[available_cols]
            
            # Set date as index
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
            
            logger.debug(f"Standardized dataframe: {len(df)} rows, columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error standardizing dataframe for {symbol}: {e}")
            return None

    def _fetch_from_exchange(self, symbol: str, start_time: datetime, end_time: datetime, 
                           data_type: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetches tradebook or orderbook data from a crypto exchange using ccxt.
        """
        # FIXED: Get the correct exchange configuration
        exchange_id = self._get_exchange_id(symbol)
        
        # Validate inputs
        if not self._validate_inputs(symbol, start_time, end_time, data_type):
            return None
        
        try:
            # Initialize exchange
            exchange = self._initialize_exchange(exchange_id)
            if not exchange:
                return None
                
            logger.info(f"Fetching {data_type} for {symbol} from {exchange.id}")
            
            # Route to appropriate data fetching method
            if data_type.lower() == 'orderbook':
                return self._fetch_orderbook_data(exchange, symbol, limit)
            elif data_type.lower() == 'tradebook':
                if start_time and end_time:
                    logger.info(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
                return self._fetch_trade_data(exchange, symbol, start_time, end_time, limit)
            else:
                logger.error(f"Unsupported data_type: {data_type}. Use 'tradebook' or 'orderbook'")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error fetching {data_type} from {exchange_id} for {symbol}: {e}")
            return None

    def _get_exchange_id(self, symbol: str) -> str:
        """
        Get the exchange ID for a given symbol.
        """
        # Look through tokens_to_monitor to find the source for this symbol
        tokens = self.config.get('tokens_to_monitor', [])
        for token in tokens:
            if token.get('symbol') == symbol:
                source = token.get('source')
                if source:
                    # Look up the actual exchange name
                    data_sources = self.config.get('data_sources', {})
                    exchange_id = data_sources.get(source, source)
                    return exchange_id
        
        # Fallback to default
        default_exchange = self.config.get('data_sources', {}).get('binance', 'binance')
        logger.warning(f"Could not find specific exchange for {symbol}, using default: {default_exchange}")
        return default_exchange

    def _validate_inputs(self, symbol: str, start_time: datetime, end_time: datetime, data_type: str) -> bool:
        """Validate input parameters"""
        if not symbol or '/' not in symbol:
            logger.error(f"Invalid symbol format: {symbol}. Expected format: 'BASE/QUOTE'")
            return False
        
        if data_type.lower() not in ['tradebook', 'orderbook']:
            logger.error(f"Invalid data_type: {data_type}. Must be 'tradebook' or 'orderbook'")
            return False
        
        # Only validate time range for tradebook data
        if data_type.lower() == 'tradebook' and start_time and end_time:
            if start_time >= end_time:
                logger.error(f"Invalid time range: start_time ({start_time}) >= end_time ({end_time})")
                return False
            
            # Check if time range is reasonable
            now = datetime.now()
            if end_time > now:
                logger.warning(f"End time is in the future, may not return expected data")
            
            if (now - start_time).days > 365:
                logger.warning(f"Requesting data older than 1 year, this might not be available")
        
        return True

    def _initialize_exchange(self, exchange_id: str) -> Optional[ccxt.Exchange]:
        """Initialize exchange with proper configuration and error handling"""
        try:
            if not hasattr(ccxt, exchange_id):
                logger.error(f"Exchange '{exchange_id}' not supported by ccxt")
                return None
            
            exchange_class = getattr(ccxt, exchange_id)
            
            # Configure exchange with rate limiting and other options
            exchange_config = {
                'rateLimit': 1200,  # Be conservative with rate limiting
                'enableRateLimit': True,
                'timeout': 30000,   # 30 second timeout
                'sandbox': False,   # Set to True for testing
            }
            
            # Add API credentials if available
            exchanges_config = self.full_config.get('exchanges', {})
            api_config = exchanges_config.get(exchange_id, {})
            exchange_config.update(api_config)
            
            exchange = exchange_class(exchange_config)
            
            # Verify exchange supports required operations
            if not exchange.has.get('fetchTrades', False) and not exchange.has.get('fetchOrderBook', False):
                logger.error(f"Exchange {exchange_id} doesn't support required operations")
                return None
                
            logger.info(f"Successfully initialized {exchange.id} exchange (rateLimit: {exchange.rateLimit}ms)")
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {e}")
            return None
    
    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def _fetch_orderbook_data(self, exchange: ccxt.Exchange, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch current orderbook data with bid/ask levels"""
        try:
            if not exchange.has.get('fetchOrderBook', False):
                logger.error(f"Exchange {exchange.id} doesn't support order book fetching")
                return None
            
            logger.info(f"Fetching orderbook for {symbol} with limit {limit}")
            orderbook = exchange.fetch_order_book(symbol, limit)
            
            # Print all keys in the orderbook dictionary
            # print(f"Orderbook keys: {list(orderbook.keys())}")
            
            # Or more detailed:
            # logger.info(f"Available orderbook fields: {', '.join(orderbook.keys())}")
            
            if not orderbook or ('bids' not in orderbook and 'asks' not in orderbook):
                logger.warning(f"Empty or invalid orderbook received for {symbol}")
                return None
            
            # Handle timestamp extraction more robustly
            timestamp_ms = orderbook.get('timestamp')
            
            if timestamp_ms is None:
                # Fallback to current time if exchange doesn't provide timestamp
                timestamp_ms = int(time.time() * 1000)
                logger.debug(f"No timestamp in orderbook, using current time: {timestamp_ms}")
            
            # Convert to pandas datetime
            date = pd.to_datetime(timestamp_ms, unit='ms')
            
            # Convert to readable datetime for logging
            timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000)
            
            logger.info(f"Order book for {symbol}:")
            logger.info(f"Timestamp (ms): {timestamp_ms}")
            logger.info(f"Timestamp (readable): {timestamp_dt}")
            logger.info(f"Pandas datetime: {date}")
            
            # Create separate records for bids and asks
            orderbook_data = []
            
            # Process bids - handle different exchange formats
            for i, level_data in enumerate(orderbook.get('bids', [])[:limit]):
                if len(level_data) >= 2:
                    price, volume = level_data[0], level_data[1]
                    # Some exchanges (like Kraken) include timestamp as 3rd element, ignore it
                    orderbook_data.append({
                        'date': date,
                        'symbol': symbol,
                        'side': 'bid',
                        'price': float(price),
                        'volume': float(volume),
                        'level': i + 1,
                        'source': exchange.id,
                        'timestamp_ms': timestamp_ms  # Include raw timestamp if needed
                    })
            
            # Process asks
            for i, level_data in enumerate(orderbook.get('asks', [])[:limit]):
                if len(level_data) >= 2:
                    price, volume = level_data[0], level_data[1]
                    orderbook_data.append({
                        'date': date,
                        'symbol': symbol,
                        'side': 'ask',
                        'price': float(price),
                        'volume': float(volume),
                        'level': i + 1,
                        'source': exchange.id,
                        'timestamp_ms': timestamp_ms
                    })
            
            if not orderbook_data:
                logger.warning(f"No orderbook data found for {symbol}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(orderbook_data)
            df.set_index('date', inplace=True)
            
            # Calculate and log market metrics
            bids = [item for item in orderbook_data if item['side'] == 'bid']
            asks = [item for item in orderbook_data if item['side'] == 'ask']
            
            if bids and asks:
                best_bid = max(bids, key=lambda x: x['price'])['price']
                best_ask = min(asks, key=lambda x: x['price'])['price']
                spread = best_ask - best_bid
                spread_pct = (spread / best_ask) * 100
                mid_price = (best_bid + best_ask) / 2
                
                total_bid_volume = sum(item['volume'] for item in bids)
                total_ask_volume = sum(item['volume'] for item in asks)
                
                logger.info(f"Orderbook metrics - Best bid: {best_bid}, Best ask: {best_ask}")
                logger.info(f"Spread: {spread:.6f} ({spread_pct:.4f}%), Mid price: {mid_price:.6f}")
                logger.info(f"Total bid volume: {total_bid_volume:.6f}, Total ask volume: {total_ask_volume:.6f}")
            
            logger.info(f"Successfully fetched orderbook with {len(df)} price levels ({len(bids)} bids, {len(asks)} asks)")
            # # logger.info(f"Sample data:\n{df.columns}")
            # logger.info(f"Sample data:\n{df.head(2)}")
            # 
            return df
            
        except ccxt.BaseError as e:
            logger.error(f"CCXT error fetching orderbook for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching orderbook for {symbol}: {e}")
            raise
          
    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def _fetch_trade_data(self, exchange: ccxt.Exchange, symbol: str, start_time: datetime, 
                         end_time: datetime, limit: int) -> Optional[pd.DataFrame]:
        """Fetch historical trade data with pagination support"""
        try:
            if not exchange.has.get('fetchTrades', False):
                logger.error(f"Exchange {exchange.id} doesn't support trade fetching")
                return None
            
            all_trades = []
            since = int(start_time.timestamp() * 1000) if start_time else None
            end_timestamp = int(end_time.timestamp() * 1000) if end_time else None
            
            logger.info(f"Starting trade data fetch with pagination (limit: {limit} per request)")
            request_count = 0
            
            # Paginate through trades
            while True:
                request_count += 1
                logger.debug(f"Request #{request_count}: Fetching trades" + 
                           (f" since {pd.to_datetime(since, unit='ms')}" if since else ""))
                
                trades = exchange.fetch_trades(symbol, since=since, limit=limit)
                
                if not trades:
                    logger.info("No more trades available, ending pagination")
                    break
                
                # Filter trades within time range if specified
                if end_timestamp:
                    filtered_trades = [
                        trade for trade in trades 
                        if trade['timestamp'] <= end_timestamp
                    ]
                else:
                    filtered_trades = trades
                
                all_trades.extend(filtered_trades)
                logger.debug(f"Retrieved {len(trades)} trades, {len(filtered_trades)} within range")
                
                # Update since timestamp for pagination
                if trades:
                    last_timestamp = trades[-1]['timestamp']
                    if last_timestamp <= (since or 0):
                        # Prevent infinite loop if we get the same timestamp
                        since = last_timestamp + 1
                    else:
                        since = last_timestamp
                else:
                    break
                
                # Stop if we've reached the end time
                if end_timestamp and since >= end_timestamp:
                    break
                    
                # Rate limiting protection
                if exchange.rateLimit > 0:
                    time.sleep(exchange.rateLimit / 1000)
                
                # Safety check to prevent infinite loops and excessive data
                if len(all_trades) > 100000:
                    logger.warning(f"Trade limit reached ({len(all_trades)} trades), stopping fetch")
                    break
            
            if not all_trades:
                logger.warning(f"No trades found for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_trades)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['source'] = exchange.id
            df['symbol'] = symbol
            
            # Ensure numeric types for price and amount
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            # Select and rename columns for consistency
            columns_to_keep = ['date', 'symbol', 'price', 'amount', 'side', 'source']
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns].rename(columns={'amount': 'volume'})
            df.set_index('date', inplace=True)
            
            # Remove duplicates and sort
            initial_count = len(df)
            df = df.drop_duplicates().sort_index()
            final_count = len(df)
            
            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} duplicate trades")
            
            # Calculate trade statistics
            if not df.empty:
                buy_trades = df[df['side'] == 'buy'] if 'side' in df.columns else df
                sell_trades = df[df['side'] == 'sell'] if 'side' in df.columns else pd.DataFrame()
                
                total_volume = df['volume'].sum()
                avg_price = df['price'].mean()
                price_range = df['price'].max() - df['price'].min()
                
                logger.info(f"Trade statistics:")
                logger.info(f"  Total trades: {len(df)} ({len(buy_trades)} buys, {len(sell_trades)} sells)")
                logger.info(f"  Total volume: {total_volume:.6f}")
                logger.info(f"  Avg price: {avg_price:.6f}, Price range: {price_range:.6f}")
                logger.info(f"  Time range: {df.index.min()} to {df.index.max()}")
            
            logger.info(f"Successfully fetched {len(df)} trades from {exchange.id} in {request_count} requests")
            logger.info(f"{df.head(2)}")
            return df
            
        except ccxt.BaseError as e:
            logger.error(f"CCXT error fetching trades for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching trades for {symbol}: {e}")
            raise

    # =====================
    # GOOGLE DRIVE METHODS
    # =====================
    
    def _fetch_from_google_drive(self, 
                               gdrive_url: str,
                               symbol: str, 
                               start_time: datetime, 
                               end_time: datetime, 
                               data_type: str,
                               sample_ratio: float = 0.00001) -> Optional[pd.DataFrame]:
        """
        Fetch and process data from Google Drive folder.
        
        Args:
            gdrive_url (str): Google Drive folder URL
            symbol (str): Trading symbol
            start_time (datetime): Start time for data filtering
            end_time (datetime): End time for data filtering  
            data_type (str): Type of data ('tradebook' or 'orderbook')
            sample_ratio (float): Ratio of files to sample
            
        Returns:
            Optional[pd.DataFrame]: Combined DataFrame from sampled files
        """
        if not self.use_google_drive or not gdown:
            logger.error("Google Drive functionality not available")
            return None
        
        logger.info(f"Fetching {data_type} data for {symbol} from Google Drive")
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"Sample ratio: {sample_ratio*100:.9f}%")
        
        # Download sample files
        sampled_file_paths = self._download_gdrive_sample(gdrive_url, sample_ratio)
        
        if not sampled_file_paths:
            logger.warning("No files downloaded from Google Drive")
            return None
        
        # Process files
        all_dataframes = []
        extract_dir = Path("temp_extracted")
        extract_dir.mkdir(exist_ok=True)
        
        for file_path in sampled_file_paths:
            try:
                # Extract if compressed
                if file_path.suffix in ['.gz', '.zip'] or '.tar' in file_path.name:
                    extracted_files = self._extract_compressed_file(file_path, extract_dir)
                else:
                    extracted_files = [file_path]
                
                # Process each extracted file
                for extracted_file in extracted_files:
                    if extracted_file.suffix in ['.csv'] or extracted_file.name.endswith('.csv.gz'):
                        df = self._process_gdrive_data_file(extracted_file, data_type, symbol)
                        if df is not None:
                            # Filter by date range if date index is available
                            if not df.empty and hasattr(df.index, 'to_series'):
                                try:
                                    df = df[(df.index >= start_time) & (df.index <= end_time)]
                                except:
                                    logger.warning(f"Could not filter by date range for {extracted_file}")
                            
                            if not df.empty:
                                all_dataframes.append(df)
                                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Combine all dataframes
        if not all_dataframes:
            logger.warning("No valid data found in sampled files")
            return None
        
        logger.info(f"Combining {len(all_dataframes)} dataframes")
        combined_df = pd.concat(all_dataframes, ignore_index=False)
        
        # Sort by date if index is datetime
        try:
            combined_df = combined_df.sort_index()
        except:
            logger.warning("Could not sort by date index")
        
        logger.info(f"Successfully combined data: {len(combined_df)} total records")
        logger.info(f"Date range in data: {combined_df.index.min()} to {combined_df.index.max()}")
        logger.info(f"Data shape: {combined_df.shape}")
        
        # Clean up temporary files
        try:
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            temp_gdrive_dir = Path("temp_gdrive_list")
            if temp_gdrive_dir.exists():
                shutil.rmtree(temp_gdrive_dir)
        except:
            logger.warning("Could not clean up temporary files")
        
        return combined_df

    def _list_gdrive_files(self, gdrive_url: str) -> List[Dict]:
        """
        List files in a Google Drive folder using gdown.
        
        Args:
            gdrive_url (str): Google Drive folder URL
            
        Returns:
            List[Dict]: List of file information dictionaries
        """
        if not gdown:
            logger.error("gdown not available. Cannot list Google Drive files.")
            return []
        
        try:
            # Extract folder ID from URL
            folder_id = self._extract_folder_id(gdrive_url)
            if not folder_id:
                logger.error("Could not extract folder ID from URL")
                return []
            
            # Create a temporary directory for downloading file list
            temp_dir = Path("temp_gdrive_list")
            temp_dir.mkdir(exist_ok=True)
            
            # Download folder structure (this will show us what's available)
            # Note: gdown has limitations (50 files max per folder)
            folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
            
            # Try to download folder to see what files are available
            try:
                gdown.download_folder(
                    folder_url, 
                    output=str(temp_dir),
                    quiet=False,
                    use_cookies=False,
                    remaining_ok=True  # Continue even if some files fail
                )
                
                # List downloaded files
                files_info = []
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        files_info.append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'size': file_path.stat().st_size,
                            'type': file_path.suffix,
                            'is_compressed': file_path.suffix in ['.gz', '.zip', '.tar', '.bz2']
                        })
                
                logger.info(f"Found {len(files_info)} files in Google Drive folder")
                return files_info
                
            except Exception as e:
                logger.error(f"Failed to download folder: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing Google Drive files: {e}")
            return []

    def _extract_folder_id(self, url: str) -> Optional[str]:
        """
        Extract folder ID from Google Drive URL.
        
        Args:
            url (str): Google Drive URL
            
        Returns:
            Optional[str]: Folder ID if found
        """
        # Handle different URL formats
        if "/folders/" in url:
            # Format: https://drive.google.com/drive/folders/FOLDER_ID
            return url.split("/folders/")[1].split("?")[0].split("/")[0]
        elif "id=" in url:
            # Format: https://drive.google.com/...?id=FOLDER_ID
            return url.split("id=")[1].split("&")[0]
        else:
            logger.warning(f"Could not extract folder ID from URL: {url}")
            return None

    def _download_gdrive_sample(self, gdrive_url: str, sample_ratio: float = 0.0001) -> List[Path]:
        """
        Download a sample of files from Google Drive folder.
        
        Args:
            gdrive_url (str): Google Drive folder URL
            sample_ratio (float): Ratio of files to download (default: 0.1% = 0.001)
            
        Returns:
            List[Path]: List of downloaded file paths
        """
        if not gdown:
            logger.error("gdown not available")
            return []
        
        # First, get the list of available files
        files_info = self._list_gdrive_files(gdrive_url)
        
        if not files_info:
            logger.warning("No files found in Google Drive folder")
            return []
        
        # Filter for relevant data files (csv, gz, tar, etc.)
        data_files = [f for f in files_info if any(ext in f['name'].lower() 
                     for ext in ['.csv', '.gz', '.tar', '.zip', 'trade', 'order'])]
        
        logger.info(f"Found {len(data_files)} potential data files")
        
        # Sample files
        n_sample = max(1, int(len(data_files) * sample_ratio))
        sampled_files = random.sample(data_files, min(n_sample, len(data_files)))
        
        logger.info(f"Sampling {len(sampled_files)} files ({sample_ratio*100:.3f}% of {len(data_files)} files)")
        
        # Print sampled files info
        logger.info("=== SAMPLED FILES ===")
        for i, file_info in enumerate(sampled_files, 1):
            logger.info(f"{i:2d}. {file_info['name']} ({file_info['size']/1024/1024:.1f} MB)")
        logger.info("=" * 50)
        
        return [Path(f['path']) for f in sampled_files]

    def _extract_compressed_file(self, file_path: Path, extract_dir: Path) -> List[Path]:
        """
        Extract compressed files (zip, tar, gz) and return paths to extracted files.
        
        Args:
            file_path (Path): Path to compressed file
            extract_dir (Path): Directory to extract files to
            
        Returns:
            List[Path]: List of extracted file paths
        """
        extracted_files = []
        
        try:
            if file_path.suffix == '.zip':
                # Handle ZIP files
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    extracted_files = [extract_dir / name for name in zip_ref.namelist()]
                    
            elif '.tar' in file_path.name:
                # Handle TAR files (tar, tar.gz, tar.bz2)
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    extracted_files = [extract_dir / member.name for member in tar_ref.getmembers() 
                                     if member.isfile()]
                                     
            elif file_path.suffix == '.gz' and not '.tar' in file_path.name:
                # Handle standalone gzip files
                extracted_path = extract_dir / file_path.stem
                with gzip.open(file_path, 'rb') as gz_file:
                    with open(extracted_path, 'wb') as out_file:
                        out_file.write(gz_file.read())
                extracted_files = [extracted_path]
                
            else:
                # File is not compressed or unknown format
                extracted_files = [file_path]
                
            logger.info(f"Extracted {len(extracted_files)} files from {file_path}")
            return extracted_files
            
        except Exception as e:
            logger.error(f"Failed to extract {file_path}: {e}")
            return [file_path]  # Return original file if extraction fails

    def _process_gdrive_data_file(self, file_path: Path, data_type: str, symbol: str) -> Optional[pd.DataFrame]:
        """
        Process a single data file from Google Drive and return standardized DataFrame.
        
        Args:
            file_path (Path): Path to data file
            data_type (str): Type of data ('tradebook' or 'orderbook')
            symbol (str): Trading symbol
            
        Returns:
            Optional[pd.DataFrame]: Processed DataFrame or None
        """
        try:
            # Read the file
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f, index_col=False)
            else:
                df = pd.read_csv(file_path, index_col=False)
            
            if df.empty:
                logger.warning(f"Empty file: {file_path}")
                return None
            
            # Use the existing standardization method
            df = self._standardize_dataframe(df, symbol, data_type, 'google_drive')
            
            if df is not None:
                # Add Google Drive specific metadata
                df['file_origin'] = str(file_path.name)
                logger.info(f"Successfully processed {len(df)} records from {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
          
          
    def fetch_selective_data(self, 
                        start_time: datetime, 
                        end_time: datetime, 
                        data_type: str, 
                        source_type: str,
                        symbols: Optional[List[str]] = None,
                        sources: Optional[List[str]] = None,
                        source_types: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Fetch data with selective filtering to avoid processing all configured tokens.
        
        Args:
            start_time (datetime): Start of time range
            end_time (datetime): End of time range  
            data_type (str): 'tradebook' or 'orderbook'
            source_type (str): Primary source type for routing
            symbols (Optional[List[str]]): Specific symbols to fetch (e.g., ['BTC/USDT', 'ETH/USDT'])
            sources (Optional[List[str]]): Specific sources to use (e.g., ['binance', 'kraken'])
            source_types (Optional[List[str]]): Specific source types (e.g., ['exchange', 'local_file'])
            
        Returns:
            Optional[pd.DataFrame]: Filtered and combined data
        """
        logger.info(f"Starting selective data fetch for {data_type}")
        
        # Get filtered token list
        filtered_tokens = self._filter_tokens(symbols, sources, source_types)
        
        if not filtered_tokens:
            logger.warning("No tokens match the specified filters")
            return None
        
        logger.info(f"Processing {len(filtered_tokens)} filtered tokens: {[t['symbol'] for t in filtered_tokens]}")
        
        # Use existing fetch logic with filtered tokens
        return self._process_tokens(filtered_tokens, start_time, end_time, data_type, source_type)
    
    def _filter_tokens(self, symbols: Optional[List[str]] = None,
                       sources: Optional[List[str]] = None, 
                       source_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Filter tokens based on provided criteria.
        
        Returns:
            List[Dict]: Filtered list of token configurations
        """
        all_tokens = self.config.get('tokens_to_monitor', [])
        filtered_tokens = []
        
        for token in all_tokens:
            # Check symbol filter
            if symbols and token.get('symbol') not in symbols:
                continue
                
            # Check source filter  
            if sources and token.get('source') not in sources:
                continue
                
            # Check source type filter
            if source_types:
                token_source_type = self._get_source_type(token.get('source', ''))
                if token_source_type not in source_types:
                    continue
                    
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def _process_tokens(self, tokens: List[Dict], start_time: datetime, 
                       end_time: datetime, data_type: str, source_type: str) -> Optional[pd.DataFrame]:
        """
        Process a list of tokens (extracted from fetch_historical_data for reuse).
        """
        all_data = []
        
        for token_info in tokens:
            symbol = token_info.get('symbol')
            source = token_info.get('source')
            
            if not symbol or not source:
                logger.warning(f"Skipping invalid token: {token_info}")
                continue
    
            actual_source_type = self._get_source_type(source)
            logger.info(f"Processing {symbol} from {source} (type: {actual_source_type})")
    
            try:
                # Route based on actual source type
                if actual_source_type == 'exchange':
                    data = self._fetch_from_exchange(symbol, start_time, end_time, data_type)
                elif actual_source_type == 'local_file':
                    data = self._fetch_from_local_file(symbol, start_time, end_time, data_type)
                elif actual_source_type == 'google_drive':
                    gdrive_config = token_info.get('gdrive_config', {})
                    gdrive_url = gdrive_config.get('url') or self.config.get('google_drive', {}).get('default_url')
                    sample_ratio = gdrive_config.get('sample_ratio', 0.0001)
                    
                    if gdrive_url:
                        data = self._fetch_from_google_drive(gdrive_url, symbol, start_time, end_time, data_type, sample_ratio)
                    else:
                        logger.error(f"No Google Drive URL configured for {symbol}")
                        continue
                else:
                    logger.warning(f"Unsupported source type '{actual_source_type}' for {symbol}")
                    continue
                
                if data is not None and not data.empty:
                    all_data.append(data)
                    logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        if not all_data:
            return None
            
        # Combine and return
        combined_df = pd.concat(all_data, ignore_index=True)
        if 'date' in combined_df.columns:
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            
        logger.info(f"Combined selective dataset: {len(combined_df)} records from {len(all_data)} sources")
        return combined_df
    
    # Convenience methods for common use cases
    def fetch_single_token(self, symbol: str, start_time: datetime, end_time: datetime, 
                          data_type: str, source_type: str = 'exchange') -> Optional[pd.DataFrame]:
        """Fetch data for a single token."""
        return self.fetch_selective_data(
            start_time, end_time, data_type, source_type, 
            symbols=[symbol]
        )
    
    def fetch_exchange_data_only(self, start_time: datetime, end_time: datetime, 
                                data_type: str) -> Optional[pd.DataFrame]:
        """Fetch data only from exchange sources."""
        return self.fetch_selective_data(
            start_time, end_time, data_type, 'exchange',
            source_types=['exchange']
        )
    
    def fetch_specific_sources(self, sources: List[str], start_time: datetime, 
                              end_time: datetime, data_type: str, source_type: str = 'exchange') -> Optional[pd.DataFrame]:
        """Fetch data from specific sources only."""
        return self.fetch_selective_data(
            start_time, end_time, data_type, source_type,
            sources=sources
        )
 
