import pandas as pd
from typing import Dict, Any, Union # Added Union
from loguru import logger
import joblib # For saving simple state
from pathlib import Path

from .base_estimator import BasePeakEstimator
# from peak_estimators.strategies.base_estimator import BasePeakEstimator


class RuleBasedPeakEstimator(BasePeakEstimator):
    """
    A simple rule-based estimator for peak detection.
    Example rules: A peak occurs if price rises by X% in Y timesteps and then drops by Z%.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.price_col = self.config.get('price_column', 'price')
        self.amount_col = self.config.get('amount_column', 'amount')
        
        # Rule parameters
        self.price_rise_threshold = self.config.get('price_rise_threshold', 0.05) # 5% rise
        self.lookback_window = self.config.get('lookback_window', 5) # 5 timesteps
        self.price_drop_threshold = self.config.get('price_drop_threshold', 0.02) # 2% drop after rise
        self.min_amount_increase = self.config.get('min_amount_increase', 0.1) # 10% amount increase
        self.check_next_timesteps = self.config.get('check_next_timesteps', 3) # How many future timesteps to check for drop

        # is_fitted should remain False until a dummy train is called
        self.is_fitted = False
        logger.info(f"RuleBasedPeakEstimator initialized with config: {self.config}")

    def train(self, data: pd.DataFrame, labels: pd.Series):
        """
        Rule-based estimators typically don't "train" in the traditional sense,
        but this method could be used to optimize rule parameters or learn thresholds
        based on training data. For now, we'll just acknowledge it.
        
        Note: Implementing classes must set `self.is_trained = True` upon successful training.
        """
        self.is_trained = True
        self.is_fitted = True
        logger.info("Rule-based estimator: No traditional training performed. Rules are pre-defined.")
        # In a more advanced rule-based system, you might use optimization to find best thresholds
        # based on training data and F1-score, etc.
        

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Applies the defined rules to identify peaks.
        Assumes 'data' is sorted by index (e.g., timestamp).
        """
        # if not self.is_trained:
        #     logger.warning("Estimator not trained. Calling `train` with dummy data before prediction.")
        #     # This is a safety net; typically train should be called explicitly.
        #     # Create dummy labels for the safety net train call.
        #     dummy_labels = pd.Series([0] * len(data), index=data.index)
        #     self.train(data.copy(), dummy_labels)
        # 
        # 
        # predictions = pd.Series(0, index=data.index, dtype=int)
        # 
        # # Ensure data has required columns
        # if self.price_col not in data.columns or self.amount_col not in data.columns:
        #     logger.error(f"Missing required columns: '{self.price_col}' or '{self.amount_col}' for rule-based prediction.")
        #     raise ValueError(f"Input data must contain '{self.price_col}' and '{self.amount_col}' columns.")
        # 
        # # Group by 'sequence_id' if it exists (for batch processing of multiple sequences)
        # if 'sequence_id' in data.index.names:
        #     for seq_id, group in data.groupby(level='sequence_id'):
        #         seq_predictions = self._apply_rules_to_sequence(group)
        #         predictions.loc[group.index] = seq_predictions
        # else:
        #     predictions = self._apply_rules_to_sequence(data)
        # 
        # logger.info(f"Rule-based prediction completed for {len(data)} samples. Found {predictions.sum()} peaks.")
        # return predictions
        if not all(col in data.columns for col in [self.price_col, self.amount_col]):
            raise ValueError(
                f"Input data must contain '{self.price_col}' and '{self.amount_col}' columns."
            )

        predictions = pd.Series(0, index=data.index, dtype=int)
        
        if isinstance(data.index, pd.MultiIndex):
            # Apply prediction logic to each sequence separately
            for sequence_id, group in data.groupby(level=0):
                predictions.loc[sequence_id] = self._apply_rules_to_sequence(group)
        else:
            # Apply prediction logic to the single sequence
            predictions = self._apply_rules_to_sequence(data)
            
        peak_count = predictions.sum()
        logger.info(f"Rule-based prediction completed for {len(data)} samples. Found {peak_count} peaks.")
        
        return predictions
      

    def _apply_rules_to_sequence(self, sequence_data: pd.DataFrame) -> pd.Series:
        #
        predictions = pd.Series(0, index=sequence_data.index, dtype=int)
        
        for i in range(self.lookback_window, len(sequence_data) - self.check_next_timesteps):
            current_price = sequence_data[self.price_col].iloc[i]
            
            # Rule 1: Check for a significant price rise
            price_lookback = sequence_data[self.price_col].iloc[i - self.lookback_window : i]
            if not price_lookback.empty and current_price > price_lookback.max() * (1 + self.price_rise_threshold):
                
                # Rule 2: Check for a significant amount increase
                current_amount = sequence_data[self.amount_col].iloc[i]
                amount_lookback = sequence_data[self.amount_col].iloc[i - self.lookback_window : i]
                if not amount_lookback.empty and current_amount > amount_lookback.max() * (1 + self.min_amount_increase):
                    
                    # Rule 3: Check for a price drop in subsequent time steps
                    subsequent_prices = sequence_data[self.price_col].iloc[i + 1 : i + 1 + self.check_next_timesteps]
                    if not subsequent_prices.empty and subsequent_prices.min() < current_price * (1 - self.price_drop_threshold):
                        predictions.iloc[i] = 1
        
        return predictions

    def save_model(self, path: Union[str, Path]): # Updated type hint
        """Saves the configuration, as there's no traditional model."""
        save_path = Path(path) / "config.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        joblib.dump(self.config, save_path)
        logger.info(f"Rule-based estimator configuration saved to {save_path}")

    def load_model(self, path: Union[str, Path]): # Updated type hint
        """Loads the configuration."""
        load_path = Path(path) / "config.pkl"
        if not load_path.exists():
            raise FileNotFoundError(f"Rule-based estimator config not found at {load_path}")
            
        self.config = joblib.load(load_path)
        self.price_col = self.config.get('price_column', 'price')
        self.amount_col = self.config.get('amount_column', 'amount')
        self.price_rise_threshold = self.config.get('price_rise_threshold', 0.05)
        self.lookback_window = self.config.get('lookback_window', 5)
        self.price_drop_threshold = self.config.get('price_drop_threshold', 0.02)
        self.min_amount_increase = self.config.get('min_amount_increase', 0.1)
        self.check_next_timesteps = self.config.get('check_next_timesteps', 3)
        self.is_trained = True # Consider it loaded
        logger.info(f"Rule-based estimator configuration loaded from {load_path}")
