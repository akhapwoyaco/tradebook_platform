import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional

class DataProcessor:
    """
    Handles all data cleaning, standardization, and feature engineering.

    This class processes raw, ingested data from multiple sources and formats,
    transforming it into a clean and feature-rich DataFrame suitable for
    analysis and model training. The methods are designed to be robust to
    different data schemas.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataProcessor with a configuration dictionary.

        Args:
            config (Dict[str, Any]): The configuration for the data processing
                                     stage, including window sizes for features.
        """
        self.config = config.get('data_processing', {})
        logger.info("DataProcessor initialized.")

    def process_data(self, raw_data: pd.DataFrame, data_type: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Orchestrates the data processing workflow based on the data type.
        This is the main entry point for the class. It delegates the processing
        to a specific method based on whether the data is a tradebook, orderbook,
        or another format.
        Args:
            raw_data (pd.DataFrame): The raw DataFrame to be processed.
            data_type (Optional[str]): The type of data ('tradebook', 'orderbook', etc.).
                                     If None, will attempt to auto-detect the data type.
        Returns:
            Optional[pd.DataFrame]: The processed DataFrame, or None if the data
                                    is invalid or an error occurs.
        """
        if raw_data is None or raw_data.empty:
            logger.warning("Input raw data is empty. Returning None.")
            return None
        
        # Clean the data first, handling any missing or invalid values
        cleaned_data = self._clean_data(raw_data)
        
        # logger.info(f"CLEANED DATA: {cleaned_data.head(2)}")
        
        # Auto-detect data type if not provided
        if data_type is None:
            data_type = self._detect_data_type(cleaned_data)
            if data_type is None:
                logger.error("Could not auto-detect data type.")
                return None
        
        # Dispatch to a specific processing method based on the data type
        if data_type == 'tradebook':
            processed_df = self._process_tradebook_data(cleaned_data)
        elif data_type == 'orderbook':
            processed_df = self._process_orderbook_data(cleaned_data)
        # Future expansion: Add more data types here, e.g., 'social_media', 'blockchain'
        else:
            logger.error(f"Unsupported data type for processing: {data_type}.")
            return None
        
        return processed_df
    
    def _detect_data_type(self, data: pd.DataFrame) -> Optional[str]:
        """
        Attempts to auto-detect the data type based on column names or patterns.
        """
        columns = set(data.columns.str.lower())
        
        # Example detection logic - adjust based on your actual data structure
        # if {'trade_id', 'price', 'quantity', 'timestamp'}.issubset(columns):
        #     return 'tradebook'
        if {'level'}.issubset(columns): #type', 'price', 'volume
            return 'orderbook'
        else:
            return 'tradebook'
        
        return None
      
      

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic data cleaning steps.

        This method handles missing values and ensures essential columns are
        in the correct format (e.g., numeric types).

        Args:
            df (pd.DataFrame): The DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logger.info("Cleaning data: handling NaNs and correcting dtypes.")
        
        # Drop rows where 'price' or 'volume' is missing, as these are critical
        df.dropna(subset=['price', 'volume'], inplace=True)
        
        # Ensure numerical columns are of the correct type
        for col in ['price', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['price', 'volume'], inplace=True)
        
        return df

    def _process_tradebook_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering specific to tradebook data.
        This method calculates financial indicators like volatility, moving averages,
        and momentum, which are key for predicting price movements.
        Args:
            df (pd.DataFrame): A cleaned tradebook DataFrame.
        Returns:
            pd.DataFrame: The DataFrame with new features.
        """
        logger.info(f"Processing tradebook data and engineering features. DataFrame shape: {df.shape}")
        
        # # Debug: Check initial data quality
        # logger.info(f"Initial NA count per column:\n{df.isnull().sum()}")
        # logger.info(f"Initial data types:\n{df.dtypes}")
        
        # Check if we have required columns
        required_cols = ['price', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df
        
        df = df.sort_index()
        
        # Calculate moving averages for price and volume
        vol_window = self.config.get('volatility_window', 20)
        vol_ma_window = self.config.get('volume_ma_window', 20)
        momentum_window = self.config.get('momentum_window', 10)
        
        # logger.info(f"Processing tradebook data: Volatility window {vol_window}, MA {vol_ma_window}, Momentum {momentum_window}")
        # 
        # # Debug: Check data before calculations
        # logger.info(f"Price column stats:\n{df['price'].describe()}")
        # logger.info(f"Volume column stats:\n{df['volume'].describe()}")
        
        # Calculate moving averages for price and volume
        df['price_ma'] = df['price'].rolling(window=vol_window).mean()
        df['volume_ma'] = df['volume'].rolling(window=vol_ma_window).mean()
        
        # Debug: Check after moving averages
        # logger.info(f"After moving averages - NA count:\n{df.isnull().sum()}")
        
        # Calculate rolling volatility as the standard deviation of returns
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        df['volatility'] = df['log_returns'].rolling(window=vol_window).std()
        
        # Debug: Check after volatility calculation
        # logger.info(f"After volatility calc - NA count:\n{df.isnull().sum()}")
        
        # Calculate price momentum
        df['price_momentum'] = df['price'].diff(momentum_window)
        
        # # Debug: Final NA count before dropna
        # logger.info(f"Before dropna - NA count per column:\n{df.isnull().sum()}")
        # logger.info(f"Before dropna - DataFrame shape: {df.shape}")
        # logger.info(f"Rows with ANY NA values: {df.isnull().any(axis=1).sum()}")
        # logger.info(f"Rows with ALL NA values: {df.isnull().all(axis=1).sum()}")
        
        # More conservative approach: only drop rows where ALL values are NaN
        # or drop rows where key columns are NaN
        initial_shape = df.shape[0]
        
        # Option 1: Drop only rows where all new feature columns are NaN
        feature_cols = ['price_ma', 'volume_ma', 'log_returns', 'volatility', 'price_momentum']
        df_cleaned = df.dropna(subset=feature_cols, how='all')
        
        # Option 2: More selective - only drop if critical features are missing
        # df_cleaned = df.dropna(subset=['price_ma', 'volatility'])
        
        rows_dropped = initial_shape - df_cleaned.shape[0]
        # logger.info(f"Dropped {rows_dropped} rows out of {initial_shape} ({rows_dropped/initial_shape*100:.1f}%)")
        
        # Debug: Final state
        # logger.info(f"Final NA count per column:\n{df_cleaned.isnull().sum()}")
        # logger.info(f"Tradebook feature engineering complete. DataFrame shape: {df_cleaned.shape}")
        
        # Additional warning if we lost too much data
        if df_cleaned.shape[0] < initial_shape * 0.5:
            logger.warning(f"Lost more than 50% of data during processing. Consider adjusting window sizes or data quality.")
        
        return df_cleaned
      

    def _process_orderbook_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering specific to orderbook data.

        This method focuses on creating features that capture market sentiment and
        liquidity dynamics, such as order pressure, from the 'type', 'price', and
        'amount' columns.

        Args:
            df (pd.DataFrame): A cleaned orderbook DataFrame with columns `date`,
                               `type`, `price`, and `volume`.

        Returns:
            pd.DataFrame: The DataFrame with new features.
        """
        logger.info("Processing orderbook data and engineering features.")
        
        # The 'type' column from the old format: 'b' for buy, 's' for sell
        # We need to map this to numerical values for processing
        df['type_numeric'] = df['type'].map({'b': 1, 's': -1})
        df = df.dropna(subset=['type_numeric'])
        
        # Calculate a signed volume based on order type
        df['signed_volume'] = df['volume'] * df['type_numeric']
        
        # Calculate order pressure as a rolling sum of signed volume
        pressure_window = self.config.get('pressure_window', 50)
        df['order_pressure'] = df['signed_volume'].rolling(window=pressure_window).sum()
        
        # Also include price and volume features similar to tradebook data
        df['price_ma'] = df['price'].rolling(window=pressure_window).mean()
        df['volume_ma'] = df['volume'].rolling(window=pressure_window).mean()

        df.dropna(inplace=True)
        
        logger.info(f"Orderbook feature engineering complete. DataFrame shape: {df.shape}")
        return df
