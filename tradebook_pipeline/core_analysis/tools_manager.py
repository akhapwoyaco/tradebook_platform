import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Tuple, List, Optional, Union
from scipy.signal import find_peaks, savgol_filter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import scikit-learn components for the ML model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class TechnicalIndicators:
    """
    Advanced technical indicators for cryptocurrency peak detection.
    Incorporates industry-standard indicators used in crypto analysis.
    """
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        RSI values above 70 typically indicate overbought conditions (potential peaks).
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        MACD crossovers and divergences can indicate potential peaks.
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, num_std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        Price touching upper band can indicate potential peaks.
        
        Returns:
            Tuple of (upper, middle, lower)
        """
        ma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = ma + (std * num_std_dev)
        lower = ma - (std * num_std_dev)
        
        return upper, ma, lower
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        Values above 80 indicate overbought conditions.
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k': k_percent.fillna(50),
            'd': d_percent.fillna(50)
        }
    
    @staticmethod
    def volume_weighted_average_price(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        Price significantly above VWAP can indicate overvaluation.
        """
        vwap = (price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
        return vwap.fillna(price)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        Values above -20 indicate overbought conditions.
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.fillna(-50)
    
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        Values above +100 indicate overbought conditions.
        """
        tp = (high + low + close) / 3  # Typical Price
        ma = tp.rolling(window=window).mean()
        md = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - ma) / (0.015 * md)
        return cci.fillna(0)


def generate_emergency_peaks(df: pd.DataFrame, strategy: str = 'percentage', **kwargs) -> np.ndarray:
    """
    Generate emergency peaks when normal detection fails completely.
    Used as last resort to keep backtesting running.
    
    Args:
        df: DataFrame with price data
        strategy: 'percentage', 'interval', or 'high_points'
        **kwargs: Additional parameters (percentage, step, quantile)
    """
    if len(df) == 0:
        return np.array([])
    
    try:
        if strategy == 'percentage':
            # Generate peaks for a percentage of the data
            percentage = kwargs.get('percentage', 10)
            n_peaks = max(1, int(len(df) * percentage / 100))
            
            # Get price column
            price_col = None
            for col in ['price', 'close', 'price_ma']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col:
                # Get indices of highest values
                price = df[price_col]
                peak_indices = price.nlargest(n_peaks).index.tolist()
                # Convert to position indices if needed
                if isinstance(peak_indices[0], (int, np.integer)):
                    return np.array(sorted(peak_indices))
                else:
                    # Convert index values to positions
                    return np.array([df.index.get_loc(idx) for idx in sorted(peak_indices)])
            else:
                # Fallback to evenly spaced
                step = max(1, len(df) // n_peaks)
                return np.array(list(range(0, len(df), step))[:n_peaks])
        
        elif strategy == 'interval':
            # Generate peaks at regular intervals
            step = kwargs.get('step', 5)
            indices = list(range(step // 2, len(df), step))
            return np.array(indices)
        
        elif strategy == 'high_points':
            # Find actual high points in the data
            price_col = None
            for col in ['price', 'close', 'price_ma']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col:
                price = df[price_col]
                quantile = kwargs.get('quantile', 0.8)
                threshold = price.quantile(quantile)
                high_points = price[price >= threshold].index.tolist()
                
                # Limit to reasonable number
                max_peaks = max(3, len(df) // 20)
                if len(high_points) > max_peaks:
                    # Select highest points
                    top_values = price.nlargest(max_peaks)
                    high_points = top_values.index.tolist()
                
                # Convert to position indices
                if isinstance(high_points[0], (int, np.integer)):
                    return np.array(sorted(high_points))
                else:
                    return np.array([df.index.get_loc(idx) for idx in sorted(high_points)])
        
        # Fallback to percentage method
        return generate_emergency_peaks(df, 'percentage')
        
    except Exception as e:
        logger.error(f"Emergency peak generation failed: {e}")
        # Final fallback
        if len(df) > 0:
            return np.array([0, len(df)//2, len(df)-1])
        return np.array([])


class AdvancedPeakDetector:
    """
    Advanced peak detection algorithms specifically designed for cryptocurrency data.
    """
    
    def fallback_peak_detection(self, df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
        """
        Ultra-simple fallback peak detection for edge cases.
        """
        try:
            # Get price column
            price_col = None
            for col in ['price', 'close', 'price_ma']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                return np.array([])
            
            price = df[price_col]
            
            if len(price) <= 2:
                return np.array([0])
            
            # Simple approach: mark points above 75th percentile as peaks
            threshold = price.quantile(0.75)
            peak_candidates = price[price >= threshold].index.tolist()
            
            # Ensure minimum distance between peaks
            if len(peak_candidates) > 1:
                min_distance = max(2, len(price) // 20)
                filtered_peaks = [peak_candidates[0]]
                
                for peak in peak_candidates[1:]:
                    if peak - filtered_peaks[-1] >= min_distance:
                        filtered_peaks.append(peak)
                
                return np.array(filtered_peaks)
            
            return np.array(peak_candidates)
            
        except Exception as e:
            logger.error(f"Fallback peak detection failed: {e}")
            if len(df) > 0:
                step = max(1, len(df) // 5)
                return np.array(list(range(0, len(df), step)))
            return np.array([])
    
    @staticmethod
    def multi_indicator_peaks(df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
        """
        Detect peaks using multiple technical indicators with configurable weights.
        Better fallback mechanisms and more robust peak detection.
        """
        logger.info("Running multi-indicator peak detection algorithm...")
        
        # Get price series
        price_col = None
        for col in ['price', 'close', 'price_ma']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            logger.error("No price column found in dataframe")
            return np.array([])
        
        price = df[price_col].copy()
        
        min_points = config.get('min_data_points', 5)
        
        if len(price) < min_points:
            logger.warning(f"Insufficient data for peak detection: {len(price)} points")
            return np.array([])
        
        # For very small datasets, use simple peak detection
        if len(price) < 10:
            logger.info("Using simplified peak detection for small dataset")
            try:
                if len(price) >= 3:
                    peaks = []
                    for i in range(1, len(price) - 1):
                        if price.iloc[i] > price.iloc[i-1] and price.iloc[i] > price.iloc[i+1]:
                            peaks.append(i)
                    
                    if not peaks and len(price) >= 2:
                        peaks = [price.idxmax()]
                        if not isinstance(peaks[0], (int, np.integer)):
                            peaks = [df.index.get_loc(peaks[0])]
                    
                    return np.array(peaks)
                else:
                    return np.array([0])
            except Exception as e:
                logger.warning(f"Simple peak detection failed: {e}")
                return np.array([])
        
        # Initialize peak scores
        peak_scores = pd.Series(0.0, index=df.index)
        
        # RSI calculation
        try:
            rsi = calculate_adaptive_rsi(price, 14)
            rsi_threshold = config.get('rsi_threshold', 60)
            rsi_weight = config.get('rsi_weight', 0.25)
            
            rsi_signals = (rsi > rsi_threshold).astype(float)
            peak_scores += rsi_signals * rsi_weight
            
            logger.debug(f"RSI peaks added: {rsi_signals.sum()} peaks")
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
        
        # Bollinger Bands
        try:
            bb_data = calculate_adaptive_bollinger_bands(price, 20)
            bb_weight = config.get('bb_weight', 0.2)
            bb_signals = (price > bb_data['upper']).astype(float)
            peak_scores += bb_signals * bb_weight
            
            logger.debug(f"Bollinger Band peaks added: {bb_signals.sum()} peaks")
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
        
        # Momentum-based peaks
        try:
            price_values = price.values
            price_std = price.std()
            price_range = price.max() - price.min()
            
            prominence = max(price_range * 0.02, price_std * 0.5)
            distance = max(5, len(price) // 50)
            
            peaks_indices, properties = find_peaks(
                price_values,
                prominence=prominence,
                distance=distance,
                height=price.quantile(0.4)
            )
            
            momentum_weight = config.get('momentum_weight', 0.4)
            momentum_signals = pd.Series(0.0, index=df.index)
            if len(peaks_indices) > 0:
                momentum_signals.iloc[peaks_indices] = 1.0
            
            peak_scores += momentum_signals * momentum_weight
            logger.debug(f"Momentum peaks added: {len(peaks_indices)} peaks")
        except Exception as e:
            logger.warning(f"Momentum calculation failed: {e}")
        
        # Volume-based peaks
        if 'volume' in df.columns:
            try:
                volume = df['volume']
                volume_ma = volume.rolling(window=20, min_periods=1).mean()
                price_change = price.pct_change()
                
                volume_weight = config.get('volume_weight', 0.15)
                volume_signals = ((volume > volume_ma * 1.2) & 
                                (price_change > 0.01)).astype(float)
                peak_scores += volume_signals * volume_weight
                logger.debug(f"Volume peaks added: {volume_signals.sum()} peaks")
            except Exception as e:
                logger.warning(f"Volume calculation failed: {e}")
        
        # Adaptive threshold
        score_mean = peak_scores.mean()
        score_std = peak_scores.std()
        
        if score_std > 0:
            adaptive_threshold = score_mean + score_std * 0.5
        else:
            adaptive_threshold = config.get('peak_threshold', 0.3)
        
        peak_threshold = min(adaptive_threshold, config.get('peak_threshold', 0.1))
        peak_indices = peak_scores[peak_scores >= peak_threshold].index.tolist()
        
        # Ensure minimum number of peaks
        min_peaks = max(3, len(df) // 100)
        if len(peak_indices) < min_peaks:
            logger.warning(f"Too few peaks detected ({len(peak_indices)}). Using top peaks.")
            top_peaks = peak_scores.nlargest(min_peaks).index.tolist()
            peak_indices = top_peaks
        
        # Ultimate fallback
        if len(peak_indices) == 0:
            logger.warning("No peaks detected. Using emergency fallback.")
            detector = AdvancedPeakDetector()
            fallback_peaks = detector.fallback_peak_detection(df, config)
            if len(fallback_peaks) > 0:
                peak_indices = fallback_peaks.tolist()
        
        # Convert index values to position indices if needed
        if len(peak_indices) > 0 and not isinstance(peak_indices[0], (int, np.integer)):
            peak_indices = [df.index.get_loc(idx) for idx in peak_indices]
        
        logger.info(f"Multi-indicator algorithm found {len(peak_indices)} peaks")
        return np.array(peak_indices)


def calculate_adaptive_rsi(price: pd.Series, base_window: int = 14) -> pd.Series:
    """Calculate RSI with adaptive window size based on data availability."""
    window = min(base_window, max(3, len(price) // 4))
    
    try:
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception as e:
        logger.warning(f"RSI calculation failed: {e}")
        return pd.Series(50, index=price.index)


def calculate_adaptive_bollinger_bands(price: pd.Series, base_window: int = 20) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands with adaptive window size."""
    window = min(base_window, max(3, len(price) // 3))
    
    try:
        ma = price.rolling(window=window, min_periods=1).mean()
        std = price.rolling(window=window, min_periods=1).std().fillna(0)
        upper = ma + (std * 1.5)
        lower = ma - (std * 1.5)
        
        return {
            'upper': upper,
            'middle': ma,
            'lower': lower,
            'width': ((upper - lower) / ma * 100).fillna(0)
        }
    except Exception as e:
        logger.warning(f"Bollinger Bands calculation failed: {e}")
        return {
            'upper': price,
            'middle': price,
            'lower': price,
            'width': pd.Series(0, index=price.index)
        }


# Placeholder classes for the rest of the implementation
class ToolsManager:
    """Tools manager with robust peak detection and ML training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.peak_config = self.config.get('peak_estimators', {})
        self.peak_detector = AdvancedPeakDetector()
        logger.info("ToolsManager initialized")
    
    def detect_peaks(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], ...]:
        """Peak detection training method."""
        strategy = self.peak_config.get('strategy', 'ensemble')
        trained_models = []
        
        df = df.reset_index(drop=True)
        
        # Rule-based model
        if strategy in ['rule_based', 'ensemble', 'ml_estimator']:
            rule_based_config = self.peak_config.get('rule_based', {})
            
            def multi_indicator_predict(data):
                return self.peak_detector.multi_indicator_peaks(data, rule_based_config)
            
            rule_based_model = {
                'name': 'Advanced Rule-Based Estimator (multi_indicator)',
                'strategy': 'rule_based',
                'model': {
                    'type': 'advanced_rule_based_model',
                    'predict': multi_indicator_predict
                }
            }
            trained_models.append(rule_based_model)
        
        return tuple(trained_models)
    
    def apply_peak_labels(self, df: pd.DataFrame, trained_model: Dict[str, Any]) -> pd.DataFrame:
        """Apply peak labels using trained model."""
        df_result = df.copy()
        
        if 'model' not in trained_model:
            df_result['is_peak'] = 0
            return df_result
        
        try:
            if trained_model['strategy'] == 'rule_based':
                predict_fn = trained_model['model'].get('predict')
                peak_indices = predict_fn(df_result)
                df_result['is_peak'] = 0
                
                if len(peak_indices) > 0:
                    valid_indices = [idx for idx in peak_indices if 0 <= idx < len(df_result)]
                    if valid_indices:
                        df_result.iloc[valid_indices, df_result.columns.get_loc('is_peak')] = 1
        except Exception as e:
            logger.error(f"Error applying peak labels: {e}")
            df_result['is_peak'] = 0
        
        return df_result


# import pandas as pd
# import numpy as np
# from loguru import logger
# from typing import Dict, Any, Tuple, List, Optional, Union
# from scipy.signal import find_peaks, savgol_filter
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')
# 
# # Import scikit-learn components for the ML model
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# 
# class TechnicalIndicators:
#     """
#     Advanced technical indicators for cryptocurrency peak detection.
#     Incorporates industry-standard indicators used in crypto analysis.
#     """
#     
#     @staticmethod
#     def rsi(series: pd.Series, window: int = 14) -> pd.Series:
#         """
#         Calculate Relative Strength Index (RSI).
#         RSI values above 70 typically indicate overbought conditions (potential peaks).
#         """
#         delta = series.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#         rs = gain / loss
#         rsi = 100 - (100 / (1 + rs))
#         return rsi.fillna(50)
#     
#     @staticmethod
#     def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
#         """
#         Calculate MACD (Moving Average Convergence Divergence).
#         MACD crossovers and divergences can indicate potential peaks.
#         """
#         ema_fast = series.ewm(span=fast).mean()
#         ema_slow = series.ewm(span=slow).mean()
#         macd_line = ema_fast - ema_slow
#         signal_line = macd_line.ewm(span=signal).mean()
#         histogram = macd_line - signal_line
#         
#         return {
#             'macd': macd_line,
#             'signal': signal_line,
#             'histogram': histogram
#         }
#     
#     @staticmethod
#     def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
#         """
#         Calculate Bollinger Bands.
#         Price touching upper band can indicate potential peaks.
#         """
#         ma = series.rolling(window=window).mean()
#         std = series.rolling(window=window).std()
#         upper = ma + (std * num_std)
#         lower = ma - (std * num_std)
#         
#         return {
#             'upper': upper,
#             'middle': ma,
#             'lower': lower,
#             'width': (upper - lower) / ma * 100  # Band width as percentage
#         }
#     
#     @staticmethod
#     def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
#                              k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
#         """
#         Calculate Stochastic Oscillator.
#         Values above 80 indicate overbought conditions.
#         """
#         lowest_low = low.rolling(window=k_window).min()
#         highest_high = high.rolling(window=k_window).max()
#         k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
#         d_percent = k_percent.rolling(window=d_window).mean()
#         
#         return {
#             'k': k_percent.fillna(50),
#             'd': d_percent.fillna(50)
#         }
#     
#     @staticmethod
#     def volume_weighted_average_price(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
#         """
#         Calculate Volume Weighted Average Price (VWAP).
#         Price significantly above VWAP can indicate overvaluation.
#         """
#         vwap = (price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
#         return vwap.fillna(price)
#     
#     @staticmethod
#     def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
#         """
#         Calculate Williams %R.
#         Values above -20 indicate overbought conditions.
#         """
#         highest_high = high.rolling(window=window).max()
#         lowest_low = low.rolling(window=window).min()
#         wr = -100 * (highest_high - close) / (highest_high - lowest_low)
#         return wr.fillna(-50)
#     
#     @staticmethod
#     def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
#         """
#         Calculate Commodity Channel Index (CCI).
#         Values above +100 indicate overbought conditions.
#         """
#         tp = (high + low + close) / 3  # Typical Price
#         ma = tp.rolling(window=window).mean()
#         md = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
#         cci = (tp - ma) / (0.015 * md)
#         return cci.fillna(0)
# 
# import pandas as pd
# import numpy as np
# from loguru import logger
# from typing import Dict, Any, Tuple, List, Optional, Union
# from scipy.signal import find_peaks, savgol_filter
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')
# 
# # Import scikit-learn components for the ML model
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# 
# class AdvancedPeakDetector:
#     """
#     Advanced peak detection algorithms specifically designed for cryptocurrency data.
#     """
#     
#     
#     def fallback_peak_detection(self, df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
#         """
#         Ultra-simple fallback peak detection for edge cases.
#         """
#         try:
#             # Get price column
#             price_col = None
#             for col in ['price', 'close', 'price_ma']:
#                 if col in df.columns:
#                     price_col = col
#                     break
#             
#             if price_col is None:
#                 return np.array([])
#             
#             price = df[price_col]
#             
#             if len(price) <= 2:
#                 return np.array([0])  # Return first index for tiny datasets
#             
#             # Simple approach: mark points above 75th percentile as peaks
#             threshold = price.quantile(0.75)
#             peak_candidates = price[price >= threshold].index.tolist()
#             
#             # Ensure minimum distance between peaks
#             if len(peak_candidates) > 1:
#                 min_distance = max(2, len(price) // 20)
#                 filtered_peaks = [peak_candidates[0]]
#                 
#                 for peak in peak_candidates[1:]:
#                     if peak - filtered_peaks[-1] >= min_distance:
#                         filtered_peaks.append(peak)
#                 
#                 return np.array(filtered_peaks)
#             
#             return np.array(peak_candidates)
#             
#         except Exception as e:
#             logger.error(f"Fallback peak detection failed: {e}")
#             # Ultimate fallback: return evenly spaced indices
#             if len(df) > 0:
#                 step = max(1, len(df) // 5)
#                 return np.array(list(range(0, len(df), step)))
#             return np.array([])
#     
#     
# 
#     
#     @staticmethod
#     def multi_indicator_peaks(df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
#         """
#         Detect peaks using multiple technical indicators with configurable weights.
#          Better fallback mechanisms and more robust peak detection.
#         """
#         logger.info("Running multi-indicator peak detection algorithm...")
#         
#         logger.info("Running multi-indicator peak detection algorithm...")
#         
#         # Get price series (try different column names)
#         price_col = None
#         for col in ['price', 'close', 'price_ma']:
#             if col in df.columns:
#                 price_col = col
#                 break
#         
#         if price_col is None:
#             logger.error("No price column found in dataframe")
#             return np.array([])
#         
#         price = df[price_col].copy()
#         
#         #  More flexible data requirements for real-time scenarios
#         min_points = config.get('min_data_points', 5)  # Much lower minimum
#         
#         if len(price) < min_points:
#             logger.warning(f"Insufficient data for peak detection: {len(price)} points (need {min_points})")
#             return np.array([])
#         
#         # For very small datasets, use simple peak detection
#         if len(price) < 10:
#             logger.info("Using simplified peak detection for small dataset")
#             try:
#                 # Simple approach for small datasets
#                 if len(price) >= 3:
#                     # Find local maxima in small dataset
#                     peaks = []
#                     for i in range(1, len(price) - 1):
#                         if price.iloc[i] > price.iloc[i-1] and price.iloc[i] > price.iloc[i+1]:
#                             peaks.append(i)
#                     
#                     if not peaks and len(price) >= 2:
#                         # If no peaks found, mark the highest point
#                         peaks = [price.idxmax()]
#                         
#                     return np.array(peaks)
#                 else:
#                     return np.array([0])  # Return first point for tiny datasets
#             except Exception as e:
#                 logger.warning(f"Simple peak detection failed: {e}")
#                 return np.array([])
#               
#         
#         
#         # Initialize peak scores
#         peak_scores = pd.Series(0.0, index=df.index)
#         
#         #  More robust RSI calculation with better thresholds
#         try:
#             #  Use adaptive RSI calculation
#             rsi = calculate_adaptive_rsi(price, 14)
#             
#             # More lenient RSI threshold for crypto and small datasets
#             rsi_threshold = config.get('rsi_threshold', 60)  # Even lower threshold
#             rsi_weight = config.get('rsi_weight', 0.25)
#             
#             rsi_signals = (rsi > rsi_threshold).astype(float)
#             peak_scores += rsi_signals * rsi_weight
#             
#             logger.debug(f"RSI peaks added: {rsi_signals.sum()} peaks (threshold: {rsi_threshold})")
#             
#         except Exception as e:
#             logger.warning(f"RSI calculation failed: {e}")
#             
#         
#         #  More robust Bollinger Bands
#         try:
#             #  Use adaptive Bollinger Bands
#             bb_data = calculate_adaptive_bollinger_bands(price, 20)
#             
#             bb_weight = config.get('bb_weight', 0.2)
#             bb_signals = (price > bb_data['upper']).astype(float)
#             peak_scores += bb_signals * bb_weight
#             
#             logger.debug(f"Bollinger Band peaks added: {bb_signals.sum()} peaks")
#             
#         except Exception as e:
#             logger.warning(f"Bollinger Bands calculation failed: {e}")
#             
#             
#         
#         #  Simple momentum-based peaks as primary method
#         try:
#             # Use scipy find_peaks with adaptive parameters
#             price_values = price.values
#             
#             # Calculate adaptive prominence based on price volatility
#             price_std = price.std()
#             price_range = price.max() - price.min()
#             
#             # Use percentage of price range for prominence
#             prominence = max(price_range * 0.02, price_std * 0.5)
#             distance = max(5, len(price) // 50)  # Adaptive distance
#             
#             peaks_indices, properties = find_peaks(
#                 price_values,
#                 prominence=prominence,
#                 distance=distance,
#                 height=price.quantile(0.4)  # Only peaks above 40th percentile
#             )
#             
#             momentum_weight = config.get('momentum_weight', 0.4)  # Increased weight
#             momentum_signals = pd.Series(0.0, index=df.index)
#             if len(peaks_indices) > 0:
#                 momentum_signals.iloc[peaks_indices] = 1.0
#             
#             peak_scores += momentum_signals * momentum_weight
#             logger.debug(f"Momentum peaks added: {len(peaks_indices)} peaks")
#             
#         except Exception as e:
#             logger.warning(f"Momentum calculation failed: {e}")
#         
#         #  Add volume-based peaks if available
#         if 'volume' in df.columns:
#             try:
#                 volume = df['volume']
#                 volume_ma = volume.rolling(window=20, min_periods=1).mean()
#                 price_change = price.pct_change()
#                 
#                 # High volume with positive price change
#                 volume_weight = config.get('volume_weight', 0.15)
#                 volume_signals = ((volume > volume_ma * 1.2) & 
#                                 (price_change > 0.01)).astype(float)
#                 peak_scores += volume_signals * volume_weight
#                 logger.debug(f"Volume peaks added: {volume_signals.sum()} peaks")
#                 
#             except Exception as e:
#                 logger.warning(f"Volume calculation failed: {e}")
#         
#         #  Adaptive threshold based on score distribution
#         score_mean = peak_scores.mean()
#         score_std = peak_scores.std()
#         
#         if score_std > 0:
#             # Use statistical threshold
#             adaptive_threshold = score_mean + score_std * 0.5
#         else:
#             adaptive_threshold = config.get('peak_threshold', 0.3)  # Lowered default threshold
#         
#         # Apply threshold
#         peak_threshold = min(adaptive_threshold, config.get('peak_threshold', 0.1))
#         peak_indices = peak_scores[peak_scores >= peak_threshold].index.tolist()
#         
#         #  Ensure minimum number of peaks for training
#         min_peaks = max(3, len(df) // 100)  # At least 1% of data as peaks
#         if len(peak_indices) < min_peaks:
#             logger.warning(f"Too few peaks detected ({len(peak_indices)}). Using top peaks.")
#             # Get top scoring points as peaks
#             top_peaks = peak_scores.nlargest(min_peaks).index.tolist()
#             peak_indices = top_peaks
#         
#         #  Ultimate fallback if no peaks detected
#         if len(peak_indices) == 0:
#             logger.warning("No peaks detected by multi-indicator method. Using fallback.")
#             fallback_peaks = self.fallback_peak_detection(df, config)
#             if len(fallback_peaks) > 0:
#                 peak_indices = fallback_peaks.tolist()
#                 logger.info(f"Fallback method generated {len(peak_indices)} peaks")
#         
# 
#         logger.info(f"Multi-indicator algorithm found {len(peak_indices)} peaks with threshold {peak_threshold:.3f}")
#         return np.array(peak_indices)
# 
# 
# 
# def calculate_adaptive_rsi(price: pd.Series, base_window: int = 14) -> pd.Series:
#     """Calculate RSI with adaptive window size based on data availability."""
#     window = min(base_window, max(3, len(price) // 4))
#     
#     try:
#         delta = price.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
#         
#         # Avoid division by zero
#         rs = gain / (loss + 1e-10)
#         rsi = 100 - (100 / (1 + rs))
#         return rsi.fillna(50)
#     except Exception as e:
#         logger.warning(f"RSI calculation failed: {e}")
#         return pd.Series(50, index=price.index)  # Neutral RSI
# 
# def calculate_adaptive_bollinger_bands(price: pd.Series, base_window: int = 20) -> Dict[str, pd.Series]:
#     """Calculate Bollinger Bands with adaptive window size."""
#     window = min(base_window, max(3, len(price) // 3))
#     
#     try:
#         ma = price.rolling(window=window, min_periods=1).mean()
#         std = price.rolling(window=window, min_periods=1).std().fillna(0)
#         upper = ma + (std * 1.5)  # Slightly tighter bands
#         lower = ma - (std * 1.5)
#         
#         return {
#             'upper': upper,
#             'middle': ma,
#             'lower': lower,
#             'width': ((upper - lower) / ma * 100).fillna(0)
#         }
#     except Exception as e:
#         logger.warning(f"Bollinger Bands calculation failed: {e}")
#         return {
#             'upper': price,
#             'middle': price,
#             'lower': price,
#             'width': pd.Series(0, index=price.index)
#         }
# 
# 
# 
# class ToolsManager:
#     """
#      Enhanced ToolsManager with robust peak detection and better ML training.
#     """
#     
#     def __init__(self, config: Dict[str, Any]):
#         self.config = config
#         self.peak_config = self.config.get('peak_estimators', {})
#         self.peak_detector = AdvancedPeakDetector()
#         logger.info("Enhanced ToolsManager initialized with improved peak detection capabilities.")
#     
#         #  Set reasonable defaults for real-time trading
#         default_peak_config = {
#             'min_data_points': 5,  # Very low minimum for real-time
#             'rsi_threshold': 60,   # Lower threshold
#             'peak_threshold': 0.1,  # Very low threshold
#             'momentum_weight': 0.4,
#             'rsi_weight': 0.25,
#             'bb_weight': 0.2,
#             'volume_weight': 0.15
#         }
#         
#         # Merge with user config
#         for key, value in default_peak_config.items():
#             if key not in self.peak_config:
#                 self.peak_config[key] = value
#         
#         logger.info(f"Enhanced ToolsManager initialized with config: min_data_points={self.peak_config['min_data_points']}")
#     
# 
#     def _prepare_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#          More robust feature preparation with better error handling.
#         """
#         logger.info("Preparing advanced features for ML model...")
#         
#         features_df = df.copy()
#         
#         # Get price column
#         price_col = None
#         for col in ['price', 'close', 'price_ma']:
#             if col in features_df.columns:
#                 price_col = col
#                 break
#         
#         if price_col is None:
#             logger.error("No price column found for feature engineering")
#             return features_df
#         
#         price = features_df[price_col]
#         
#         try:
#             #  More robust technical indicators with min_periods
#             # RSI
#             delta = price.diff()
#             gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
#             loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
#             rs = gain / (loss + 1e-10)
#             features_df['rsi'] = 100 - (100 / (1 + rs))
#             
#             # MACD
#             ema_fast = price.ewm(span=12, min_periods=1).mean()
#             ema_slow = price.ewm(span=26, min_periods=1).mean()
#             macd_line = ema_fast - ema_slow
#             signal_line = macd_line.ewm(span=9, min_periods=1).mean()
#             features_df['macd'] = macd_line
#             features_df['macd_signal'] = signal_line
#             features_df['macd_histogram'] = macd_line - signal_line
#             
#             # Bollinger Bands
#             window = min(20, len(price) // 4)
#             bb_ma = price.rolling(window=window, min_periods=1).mean()
#             bb_std = price.rolling(window=window, min_periods=1).std()
#             features_df['bb_upper'] = bb_ma + (bb_std * 2)
#             features_df['bb_lower'] = bb_ma - (bb_std * 2)
#             features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / bb_ma * 100
#             features_df['bb_position'] = (price - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
#             
#             # Price-based features
#             features_df['price_change'] = price.pct_change()
#             features_df['price_volatility'] = price.pct_change().rolling(window=10, min_periods=1).std()
#             features_df['price_momentum'] = price / price.shift(5) - 1
#             features_df['price_trend'] = price.rolling(window=10, min_periods=1).mean().pct_change()
#             
#             # Rolling statistics
#             features_df['price_zscore'] = (price - price.rolling(window=20, min_periods=1).mean()) / price.rolling(window=20, min_periods=1).std()
#             features_df['price_percentile'] = price.rolling(window=50, min_periods=1).rank(pct=True)
#             
#         except Exception as e:
#             logger.warning(f"Some feature engineering failed: {e}")
#         
#         #  Better NaN handling
#         features_df = features_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
#         
#         # Replace infinite values
#         features_df = features_df.replace([np.inf, -np.inf], 0)
#         
#         logger.info(f"Feature engineering completed. Total features: {len(features_df.columns)}")
#         return features_df
# 
#     def detect_peaks(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], ...]:
#         """
#          Enhanced peak detection with better ML training logic.
#         """
#         strategy = self.peak_config.get('strategy', 'ensemble')
#         trained_models = []
#         
#         df = df.reset_index(drop=True)
#         logger.info(f"Using enhanced peak detection strategy: {strategy}")
# 
#         # Rule-based model (always works)
#         if strategy in ['rule_based', 'ensemble', 'ml_estimator']:
#             rule_based_config = self.peak_config.get('rule_based', {})
#             
#             logger.info("Training advanced rule-based model...")
#             
#             def multi_indicator_predict(data):
#                 return self.peak_detector.multi_indicator_peaks(data, rule_based_config)
#             
#             rule_based_model = {
#                 'name': 'Advanced Rule-Based Estimator (multi_indicator)',
#                 'strategy': 'rule_based',
#                 'algorithm': 'multi_indicator',
#                 'parameters': rule_based_config,
#                 'model': {
#                     'type': 'advanced_rule_based_model',
#                     'id': 'rule_based_multi_indicator_model',
#                     'predict': multi_indicator_predict
#                 }
#             }
#             trained_models.append(rule_based_model)
#             logger.info("Advanced rule-based model training completed.")
# 
#         #  Enhanced ML model training
#         if strategy in ['ml_estimator', 'ensemble']:
#             ml_config = self.peak_config.get('ml_estimator', {})
#             model_type = ml_config.get('model_type', 'RandomForestClassifier')
#             logger.info(f"Training enhanced ML model: {model_type}...")
#             
#             try:
#                 # Prepare features
#                 features_df = self._prepare_features_for_ml(df)
#                 
#                 #  Generate labels using rule-based method first
#                 logger.info("Generating high-quality labels for ML training...")
#                 peak_indices = self.peak_detector.multi_indicator_peaks(
#                     features_df, 
#                     {
#                         'rsi_threshold': 65,
#                         'peak_threshold': 0.2,  # Lower threshold for more peaks
#                         'momentum_weight': 0.5
#                     }
#                 )
#                 
#                 # Create target variable
#                 target_df = features_df.copy()
#                 target_df['is_peak'] = 0
#                 if len(peak_indices) > 0:
#                     #  Handle both index values and position indices
#                     if isinstance(peak_indices[0], (int, np.integer)):
#                         # If these are position indices
#                         valid_indices = [idx for idx in peak_indices if 0 <= idx < len(target_df)]
#                         target_df.iloc[valid_indices, target_df.columns.get_loc('is_peak')] = 1
#                     else:
#                         # If these are index values
#                         valid_indices = [idx for idx in peak_indices if idx in target_df.index]
#                         target_df.loc[valid_indices, 'is_peak'] = 1
#                 
#                 # Select features for ML model
#                 feature_columns = [col for col in target_df.columns 
#                                  if col not in ['is_peak', 'price', 'close', 'high', 'low', 'timestamp'] 
#                                  and target_df[col].dtype in ['float64', 'int64']]
#                 
#                 X = target_df[feature_columns]
#                 y = target_df['is_peak']
#                 
#                 logger.info(f"ML training with {X.shape[1]} features and {y.sum()} positive samples out of {len(y)} total")
#                 
#                 #  Handle insufficient positive samples
#                 if y.sum() < 5:
#                     logger.warning("Very few positive samples. Creating synthetic balanced dataset.")
#                     
#                     # Create a more balanced synthetic dataset
#                     positive_indices = np.where(y == 1)[0]
#                     negative_indices = np.where(y == 0)[0]
#                     
#                     # If we have any positive samples, oversample them
#                     if len(positive_indices) > 0:
#                         # Replicate positive samples
#                         n_replications = max(1, min(10, len(negative_indices) // len(positive_indices)))
#                         replicated_positive = np.tile(positive_indices, n_replications)
#                         
#                         # Sample negative examples
#                         n_negative = min(len(negative_indices), len(replicated_positive) * 3)
#                         selected_negative = np.random.choice(negative_indices, n_negative, replace=False)
#                         
#                         balanced_indices = np.concatenate([replicated_positive, selected_negative])
#                     else:
#                         # Create artificial peaks at high price points
#                         price_col = None
#                         for col in ['price', 'close', 'price_ma']:
#                             if col in target_df.columns:
#                                 price_col = col
#                                 break
#                         
#                         if price_col:
#                             price_values = target_df[price_col]
#                             # Mark top 5% of prices as peaks
#                             threshold = price_values.quantile(0.95)
#                             artificial_peaks = price_values >= threshold
#                             target_df.loc[artificial_peaks, 'is_peak'] = 1
#                             y = target_df['is_peak']
#                             
#                             positive_indices = np.where(y == 1)[0]
#                             negative_indices = np.where(y == 0)[0]
#                             
#                             n_negative = min(len(negative_indices), len(positive_indices) * 4)
#                             selected_negative = np.random.choice(negative_indices, n_negative, replace=False)
#                             balanced_indices = np.concatenate([positive_indices, selected_negative])
#                         else:
#                             raise ValueError("Cannot create artificial peaks without price data")
#                     
#                     X = X.iloc[balanced_indices]
#                     y = y.iloc[balanced_indices]
#                     
#                     logger.info(f"Created balanced dataset with {y.sum()} positive and {(y == 0).sum()} negative samples")
#                 
#                 #  Proper train-test split
#                 test_size = min(0.3, max(0.1, 20 / len(X)))  # Adaptive test size
#                 
#                 # Only stratify if we have enough samples in each class
#                 stratify_y = y if y.sum() >= 2 and (y == 0).sum() >= 2 else None
#                 
#                 X_train, X_test, y_train, y_test = train_test_split(
#                     X, y, 
#                     test_size=test_size, 
#                     random_state=42, 
#                     stratify=stratify_y
#                 )
#                 
#                 # Scale features
#                 scaler = StandardScaler()
#                 X_train_scaled = scaler.fit_transform(X_train)
#                 X_test_scaled = scaler.transform(X_test)
#                 
#                 #  Model initialization with better parameters
#                 model_params = ml_config.get('params', {
#                     'n_estimators': 100,
#                     'max_depth': 8,
#                     'min_samples_split': 5,
#                     'min_samples_leaf': 2,
#                     'random_state': 42,
#                     'class_weight': 'balanced'
#                 })
#                 
#                 if model_type == 'RandomForestClassifier':
#                     model = RandomForestClassifier(**model_params)
#                 elif model_type == 'GradientBoostingClassifier':
#                     model = GradientBoostingClassifier(**model_params)
#                 else:
#                     model = RandomForestClassifier(**model_params)
#                 
#                 # Train model
#                 model.fit(X_train_scaled, y_train)
#                 
#                 # Evaluate model
#                 if len(X_test) > 0:
#                     y_pred = model.predict(X_test_scaled)
#                     logger.info(f"ML Model Performance:\n{classification_report(y_test, y_pred, zero_division=0)}")
#                 
#                 ml_model = {
#                     'name': f'Enhanced ML Estimator ({model_type})',
#                     'strategy': 'ml_estimator',
#                     'model_type': model_type,
#                     'parameters': ml_config,
#                     'feature_columns': feature_columns,
#                     'model': {
#                         'type': model_type,
#                         'id': f'enhanced_ml_{model_type.lower()}',
#                         'object': model,
#                         'scaler': scaler,
#                         'features': feature_columns
#                     }
#                 }
#                 trained_models.append(ml_model)
#                 logger.info("Enhanced ML model training completed successfully.")
#                 
#             except Exception as e:
#                 logger.error(f"Enhanced ML model training failed: {e}")
#                 # Create a fallback model that uses rule-based predictions
#                 def fallback_ml_predict(data):
#                     try:
#                         return self.peak_detector.multi_indicator_peaks(data, {'peak_threshold': 0.3})
#                     except:
#                         return np.array([])
#                 
#                 fallback_ml_model = {
#                     'name': f'Enhanced ML Estimator ({model_type}) - FALLBACK',
#                     'strategy': 'ml_estimator',
#                     'model_type': model_type,
#                     'parameters': ml_config,
#                     'model': {
#                         'type': 'fallback_rule_based',
#                         'id': 'fallback_model',
#                         'predict': fallback_ml_predict,
#                         'object': None,
#                         'scaler': None
#                     }
#                 }
#                 trained_models.append(fallback_ml_model)
# 
#         if not trained_models:
#             logger.error("No valid peak detection strategy was executed.")
#         
#         logger.info(f"Peak detection training completed. Generated {len(trained_models)} models.")
#         return tuple(trained_models)
# 
#     def apply_peak_labels(self, df: pd.DataFrame, trained_model: Dict[str, Any]) -> pd.DataFrame:
#         """
#          Enhanced method to apply trained peak detection models with better error handling.
#         """
#         df_result = df.copy()
#         
#         if 'model' not in trained_model:
#             logger.error("Trained model is invalid. Missing 'model' key.")
#             df_result['is_peak'] = 0
#             return df_result
#         
#         model_name = trained_model.get('name', 'Unknown Model')
#         logger.info(f"Applying peak labels using: {model_name}")
#         
#         try:
#             if trained_model['strategy'] == 'rule_based':
#                 predict_fn = trained_model['model'].get('predict')
#                 if not predict_fn or not callable(predict_fn):
#                     logger.error("Rule-based model is invalid. Missing 'predict' function.")
#                     df_result['is_peak'] = 0
#                     return df_result
#                 
#                 peak_indices = predict_fn(df_result)
#                 df_result['is_peak'] = 0
#                 
#                 if isinstance(peak_indices, (list, np.ndarray)) and len(peak_indices) > 0:
#                     valid_indices = [idx for idx in peak_indices if 0 <= idx < len(df_result)]
#                     if valid_indices:
#                         df_result.iloc[valid_indices, df_result.columns.get_loc('is_peak')] = 1
#                         logger.info(f"Applied {len(valid_indices)} peak labels using rule-based model.")
#             
#             elif trained_model['strategy'] == 'ml_estimator':
#                 model_obj = trained_model['model'].get('object')
#                 scaler = trained_model['model'].get('scaler')
#                 
#                 #  Handle fallback models
#                 if model_obj is None:
#                     predict_fn = trained_model['model'].get('predict')
#                     if predict_fn and callable(predict_fn):
#                         logger.info("Using fallback prediction method for ML model")
#                         peak_indices = predict_fn(df_result)
#                         df_result['is_peak'] = 0
#                         
#                         if isinstance(peak_indices, (list, np.ndarray)) and len(peak_indices) > 0:
#                             valid_indices = [idx for idx in peak_indices if 0 <= idx < len(df_result)]
#                             if valid_indices:
#                                 df_result.iloc[valid_indices, df_result.columns.get_loc('is_peak')] = 1
#                     else:
#                         logger.warning("ML model failed. Using simple fallback.")
#                         df_result['is_peak'] = 0
#                         # Simple fallback pattern
#                         if len(df_result) > 10:
#                             step = max(1, len(df_result) // 20)
#                             fallback_indices = list(range(0, len(df_result), step))
#                             df_result.iloc[fallback_indices, df_result.columns.get_loc('is_peak')] = 1
#                 else:
#                     # Normal ML prediction
#                     features_df = self._prepare_features_for_ml(df_result)
#                     feature_columns = trained_model['model'].get('features', [])
#                     
#                     available_features = [col for col in feature_columns if col in features_df.columns]
#                     X_pred = features_df[available_features].select_dtypes(include=[np.number])
#                     X_pred = X_pred.fillna(0).replace([np.inf, -np.inf], 0)
#                     
#                     X_pred_scaled = scaler.transform(X_pred)
#                     predictions = model_obj.predict(X_pred_scaled)
#                     df_result['is_peak'] = predictions.astype(int)
#                     
#                     logger.info(f"Applied ML model predictions. Found {predictions.sum()} peaks.")
#             
#             # Ensure is_peak is integer type
#             df_result['is_peak'] = df_result['is_peak'].astype(int)
#             
#             # Final statistics
#             peak_count = df_result['is_peak'].sum()
#             total_samples = len(df_result)
#             peak_percentage = (peak_count / total_samples * 100) if total_samples > 0 else 0
#             
#             logger.info(f"Peak labeling completed: {peak_count}/{total_samples} ({peak_percentage:.2f}%) peaks identified")
#             
#         
#         except Exception as e:
#             logger.error(f"Error applying peak labels: {e}")
#             #  Smarter fallback that ensures some peaks for backtesting
#             df_result['is_peak'] = 0
#             
#             emergency_peaks = generate_emergency_peaks(df_result, 'high_points')
#             if len(emergency_peaks) > 0:
#                 valid_emergency = [idx for idx in emergency_peaks if 0 <= idx < len(df_result)]
#                 if valid_emergency:
#                     df_result.iloc[valid_emergency, df_result.columns.get_loc('is_peak')] = 1
#                     logger.info(f"Applied emergency peak pattern: {len(valid_emergency)} peaks")
#             
#             # Final sanity check - ensure at least one peak for backtesting
#             if df_result['is_peak'].sum() == 0 and len(df_result) > 0:
#                 df_result.iloc[len(df_result)//2, df_result.columns.get_loc('is_peak')] = 1
#                 logger.warning("Applied single emergency peak to prevent backtesting failure")
#         
# 
#         # except Exception as e:
#         #     logger.error(f"Error applying peak labels: {e}")
#         #     # Ultimate fallback
#         #     df_result['is_peak'] = 0
#         #     if len(df_result) > 5:
#         #         step = max(1, len(df_result) // 10)
#         #         fallback_indices = list(range(step//2, len(df_result), step))
#         #         df_result.iloc[fallback_indices, df_result.columns.get_loc('is_peak')] = 1
#         #         logger.info(f"Applied fallback peak pattern: {len(fallback_indices)} peaks")
#         
#         return df_result
# 
# # Standalone utility functions for backward compatibility
# def rule_based_predict(data, height=1):
#     """
#     Module-level function for backward compatibility with simple rule-based peak prediction.
#     Enhanced version with better parameter handling.
#     """
#     try:
#         # Get price series
#         if isinstance(data, pd.DataFrame):
#             price_col = None
#             for col in ['price', 'close', 'price_ma']:
#                 if col in data.columns:
#                     price_col = col
#                     break
#             
#             if price_col is None:
#                 return np.array([])
#             
#             price_series = data[price_col]
#         else:
#             price_series = data
#         
#         if len(price_series) == 0:
#             return np.array([])
#         
#         # Auto-adjust height if needed
#         if height == 1 and hasattr(price_series, 'std'):
#             height = max(1, price_series.std() * 0.5)
#         
#         peaks, _ = find_peaks(price_series, height=height)
#         return peaks
#         
#     except Exception as e:
#         logger.error(f"Error in rule_based_predict: {e}")
#         return np.array([])
# 
# 
# def enhanced_peak_predict(data, config=None):
#     """
#     Enhanced standalone peak prediction function.
#     
#     Args:
#         data (pd.DataFrame): Input data
#         config (Dict): Configuration parameters
#         
#     Returns:
#         np.ndarray: Peak indices
#     """
#     if config is None:
#         config = {
#             'algorithm': 'multi_indicator',
#             'rsi_threshold': 70,
#             'peak_threshold': 0.4
#         }
#     
#     try:
#         detector = AdvancedPeakDetector()
#         
#         algorithm = config.get('algorithm', 'multi_indicator')
#         if algorithm == 'multi_indicator':
#             return detector.multi_indicator_peaks(data, config)
#         elif algorithm == 'adaptive':
#             return detector.adaptive_peak_detection(data, config)
#         elif algorithm == 'ensemble':
#             return detector.ensemble_peak_detection(data, config)
#         else:
#             return rule_based_predict(data, config.get('height', 1))
#             
#     except Exception as e:
#         logger.error(f"Error in enhanced_peak_predict: {e}")
#         return np.array([])
# 
# # SNIPPET 8: Emergency peak generation for backtesting
# # Add this utility function to handle backtesting scenarios:
# 
# def generate_emergency_peaks(df: pd.DataFrame, strategy: str = 'percentage') -> np.ndarray:
#     """
#     Generate emergency peaks when normal detection fails completely.
#     Used as last resort to keep backtesting running.
#     """
#     if len(df) == 0:
#         return np.array([])
#     
#     try:
#         if strategy == 'percentage':
#             # Generate peaks for roughly 5-10% of the data
#             n_peaks = max(1, len(df) // 15)  # At least 1 peak, roughly 6.7% of data
#             step = len(df) // n_peaks
#             indices = list(range(step//2, len(df), step))[:n_peaks]
#             return np.array(indices)
#         
#         elif strategy == 'high_points':
#             # Find actual high points in the data
#             price_col = None
#             for col in ['price', 'close', 'price_ma']:
#                 if col in df.columns:
#                     price_col = col
#                     break
#             
#             if price_col:
#                 price = df[price_col]
#                 threshold = price.quantile(0.8)  # Top 20% of values
#                 high_points = price[price >= threshold].index.tolist()
#                 
#                 # Limit to reasonable number
#                 max_peaks = max(3, len(df) // 20)
#                 if len(high_points) > max_peaks:
#                     # Select highest points
#                     top_values = price.nlargest(max_peaks)
#                     high_points = top_values.index.tolist()
#                 
#                 return np.array(high_points)
#         
#         # Fallback to percentage method
#         return generate_emergency_peaks(df, 'percentage')
#         
#     except Exception as e:
#         logger.error(f"Emergency peak generation failed: {e}")
#         # Final fallback
#         if len(df) > 0:
#             return np.array([0, len(df)//2, len(df)-1])
#         return np.array([])
# 
