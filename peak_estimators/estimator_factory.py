import pandas as pd
from typing import Dict, Any, Type, List, Union # Ensure Union is imported
from loguru import logger
from pathlib import Path

# Import concrete estimator strategies
from .strategies.base_estimator import BasePeakEstimator
from .strategies.rule_based_estimator import RuleBasedPeakEstimator
from .strategies.ml_estimator import MLEstimator

class PeakEstimatorFactory:
    """
    Factory class to create and manage peak estimation strategy instances.
    It can load and save estimators based on configuration and their specific model paths.
    """
    _STRATEGY_MAP: Dict[str, Type[BasePeakEstimator]] = {
        'rule_based': RuleBasedPeakEstimator,
        'ml_estimator': MLEstimator,
        # Add more mappings for other strategies (e.g., 'statistical_estimator', 'ensemble_estimator')
    }

    def __init__(self, full_config: Dict[str, Any], base_peak_model_path: Path):
        # self.config = full_config.get('peak_detection', {}) # Assuming a 'peak_detection' section in full config
        self.config = full_config.get('peak_estimators', {}) # Corrected!

        # self.model_storage_path = Path(self.config.get('model_dir', 'models/peak_detection/'))
        # self.model_storage_path.mkdir(parents=True, exist_ok=True)
        # Use the path passed from the pipeline directly
        self.model_storage_path = base_peak_model_path
        self.model_storage_path.mkdir(parents=True, exist_ok=True) # Ensure it exists if not already
        logger.info(f"PeakEstimatorFactory initialized. Model storage base for loading: {self.model_storage_path}")
        logger.info(f"PeakEstimatorFactory initialized. Model storage base: {self.model_storage_path}")

    def create_estimator(self, estimator_name: str) -> BasePeakEstimator:
        """
        Creates a *new* instance of a specified peak estimator strategy based on configuration.
        
        Args:
            estimator_name (str): The name of the estimator as defined in the config.
                                  e.g., 'my_rule_estimator', 'my_lstm_model'.
                                  
        Returns:
            BasePeakEstimator: A new instance of the requested peak estimator.
            
        Raises:
            ValueError: If the estimator name or type is not found in the configuration.
        """
        estimator_config = self.config.get('estimators', {}).get(estimator_name)
        if not estimator_config:
            raise ValueError(f"Estimator '{estimator_name}' not found in peak_detection configuration.")

        strategy_type = estimator_config.get('type')
        if not strategy_type:
            raise ValueError(f"Estimator '{estimator_name}' has no 'type' specified in config.")

        StrategyClass = self._STRATEGY_MAP.get(strategy_type.lower())
        if not StrategyClass:
            raise ValueError(f"Unknown peak estimation strategy type: '{strategy_type}' for estimator '{estimator_name}'.")

        # Pass the specific config for this estimator
        estimator_instance = StrategyClass(estimator_config)
        
        # Set the model_save_path_root for the estimator if it needs it
        # This allows the estimator itself to manage its sub-paths within its dedicated model directory
        if hasattr(estimator_instance, 'model_save_path_root'):
            estimator_instance.model_save_path_root = self.model_storage_path / estimator_name
            estimator_instance.model_save_path_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created new instance of '{strategy_type}' for estimator '{estimator_name}'.")
        return estimator_instance

    def load_estimator(self, estimator_name: str) -> BasePeakEstimator:
        """
        Loads a previously trained peak estimator.
        
        Args:
            estimator_name (str): The name of the estimator to load.
            
        Returns:
            BasePeakEstimator: A loaded and initialized peak estimator instance.
            
        Raises:
            ValueError: If the estimator cannot be loaded.
            FileNotFoundError: If the model's directory or config is missing.
        """
        # Create an instance first using the factory, which sets up its internal config
        estimator_instance = self.create_estimator(estimator_name) 
        
        # The path where this specific estimator's model data is stored
        model_path_for_estimator = self.model_storage_path / estimator_name
        
        if not model_path_for_estimator.exists():
            raise FileNotFoundError(f"Model path for estimator '{estimator_name}' not found at {model_path_for_estimator}. Has it been trained and saved?")
        
        try:
            # Call the load_model method on the created instance, passing Path object directly
            estimator_instance.load_model(model_path_for_estimator)
            logger.info(f"Estimator '{estimator_name}' loaded successfully from {model_path_for_estimator}.")
            return estimator_instance
        except Exception as e:
            logger.error(f"Failed to load estimator '{estimator_name}' from {model_path_for_estimator}: {e}", exc_info=True)
            raise ValueError(f"Failed to load estimator '{estimator_name}': {e}")

    def get_available_estimators(self) -> List[str]:
        """Returns a list of all estimator names defined in the configuration."""
        return list(self.config.get('estimators', {}).keys())