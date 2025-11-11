from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List, Tuple, Union # Added Union for Path flexibility
from pathlib import Path # Added Path for type hinting consistency
from loguru import logger
import pickle

class BasePeakEstimator(ABC):
    """
    Abstract Base Class for all peak estimation strategies.
    Defines the common interface for training, prediction, and configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_trained = False
        logger.info(f"BasePeakEstimator initialized.")

    @abstractmethod
    def train(self, data: pd.DataFrame, labels: pd.Series):
        """
        Trains the peak estimation model/strategy.
        
        Args:
            data (pd.DataFrame): Input features for training.
            labels (pd.Series): Binary labels indicating peak (1) or not (0).
        
        Note: Implementing classes must set `self.is_trained = True` upon successful training.
        """
        # pass # Removed pass from docstring
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predicts peak occurrences based on new data.
        
        Args:
            data (pd.DataFrame): Input features for prediction.
            
        Returns:
            pd.Series: Binary predictions (1 for peak, 0 for not a peak).
        """
        # pass # Removed pass from docstring
        pass

    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """
        Predicts probabilities of peak occurrences based on new data.
        This method is optional and may not be implemented by all estimators (e.g., rule-based).
        
        Args:
            data (pd.DataFrame): Input features for prediction.
            
        Returns:
            pd.Series: Probabilities of the positive class (peak).
        
        Raises:
            NotImplementedError: If the estimator does not support probability prediction.
        """
        raise NotImplementedError("This estimator does not support probability prediction.")

    @abstractmethod
    def save_model(self, path: Union[str, Path]): # Updated type hint
        """Saves the trained model/strategy state to a specified path."""
        pass

    @abstractmethod
    def load_model(self, path: Union[str, Path]): # Updated type hint
        """Loads a trained model/strategy state from a specified path."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration used by this estimator."""
        return self.config

    def is_fitted(self) -> bool:
        """Returns True if the estimator has been trained."""
        return self.is_trained
      
    def save(self, path: Path):
        """
        Saves the fitted model to a file.

        Parameters:
        - path (Path): The file path to save the model to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """
        Loads a model from a file.

        Parameters:
        - path (Path): The file path to load the model from.

        Returns:
        - BasePeakEstimator: The loaded model instance.
        """
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
