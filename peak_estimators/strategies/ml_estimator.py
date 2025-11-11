import pandas as pd
import numpy as np # Added numpy import
from typing import Dict, Any, Tuple, Union # Added Union for Path flexibility
from loguru import logger
import joblib # For saving simple ML models or metadata
from pathlib import Path
import yaml # For saving metadata

from .base_estimator import BasePeakEstimator

# Import scikit-learn models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Add any other ML models you might support

# Placeholder for actual ML framework (e.g., PyTorch, TensorFlow, scikit-learn)
# For simplicity, we'll use a dummy scikit-learn like classifier.
# In a real scenario, you'd import specific model classes and implement their
# training/prediction logic here (e.g., LSTM, Transformer, RandomForest).
class DummyMLClassifier:
    """
    A simple dummy classifier for demonstration purposes.
    Simulates training and predicts based on a very basic rule on the first feature.
    """
    def __init__(self, n_features: int, epochs: int = 10):
        self.n_features = n_features
        self.epochs = epochs
        self.is_fitted = False
        self.weights = None # Dummy weights
        self.bias = 0.0 # Dummy bias
        logger.info(f"DummyMLClassifier initialized with {n_features} features, {epochs} epochs.")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info(f"DummyMLClassifier: Simulating training for {self.epochs} epochs on {len(X)} samples...")
        # In a real model, this would be where your neural network trains
        # or where a scikit-learn model's fit method is called.
        # Dummy "training": set weights to something simple based on input data
        self.weights = [1.0 / self.n_features] * self.n_features # Equal weight to all features
        self.bias = y.mean() # Dummy bias based on label distribution
        self.is_fitted = True
        logger.info("DummyMLClassifier training simulated.")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        logger.info("DummyMLClassifier: Simulating prediction probabilities...")
        if not self.is_fitted:
            raise RuntimeError("DummyMLClassifier not fitted. Call fit() first.")
        
        # Simple linear combination for probabilities
        # This is highly simplistic, a real model would have proper activation
        raw_scores = (X * self.weights).sum(axis=1) + self.bias
        # Sigmoid-like transformation to get probabilities between 0 and 1
        probabilities = 1 / (1 + np.exp(-raw_scores)) # Changed pd.np.exp to np.exp
        
        return probabilities

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        logger.info(f"DummyMLClassifier: Simulating binary prediction with threshold {threshold}...")
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path: Union[str, Path]): # Updated type hint
        """Saves dummy model parameters."""
        model_state = {'weights': self.weights, 'bias': self.bias, 'n_features': self.n_features, 'epochs': self.epochs}
        joblib.dump(model_state, path)
        logger.info(f"DummyMLClassifier model state saved to {path}")

    def load(self, path: Union[str, Path]): # Updated type hint
        """Loads dummy model parameters."""
        model_state = joblib.load(path)
        self.weights = model_state['weights']
        self.bias = model_state['bias']
        self.n_features = model_state['n_features']
        self.epochs = model_state['epochs']
        self.is_fitted = True
        logger.info(f"DummyMLClassifier model state loaded from {path}")


class MLEstimator(BasePeakEstimator):
    """
    An ML-based estimator for peak detection.
    This could use LSTM, Transformer, or simpler models like RandomForest, GradientBoosting.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize attributes directly from the 'config' dictionary passed to __init__
        self.model_type = self.config.get('model_type') # Removed default, as it's required for model instantiation
        self.features = self.config.get('features', ['price', 'amount']) # Provide a sensible default
        self.epochs = self.config.get('epochs', 10)
        self.prediction_threshold = self.config.get('prediction_threshold', 0.5)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)

        self.model = None # Initialize model as None, it will be set based on model_type
        self.model_params = config.get('params', {}) # Model-specific parameters from config

        # logger.info(f"MLEstimator __init__: Initializing with config: '{self.config}' ")
        # logger.info(f"MLEstimator __init__: Determined model_type: '{self.model_type}' (type: {type(self.model_type)}, repr: {repr(self.model_type)})")
        logger.info(f"MLEstimator __init__: Comparison check - is '{self.model_type}' == 'RandomForestClassifier'? {self.model_type == 'RandomForestClassifier'}")
        logger.info(f"MLEstimator __init__: Comparison check - is '{self.model_type}' == 'GradientBoostingClassifier'? {self.model_type == 'GradientBoostingClassifier'}")

        # Dynamically create the scikit-learn model based on model_type and self.model_params
        if self.model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier(**self.model_params)
            logger.info("MLEstimator __init__: Instantiated RandomForestClassifier.")
        elif self.model_type == 'GradientBoostingClassifier':
            self.model = GradientBoostingClassifier(**self.model_params)
            logger.info("MLEstimator __init__: Instantiated GradientBoostingClassifier.")
        elif self.model_type == 'dummy_classifier':
            # For dummy_classifier, ensure n_features and epochs are available in config or derived
            dummy_n_features = len(self.features) if self.features else 1 # Ensure features is not empty
            dummy_epochs = self.config.get('epochs', 10)
            self.model = DummyMLClassifier(n_features=dummy_n_features, epochs=dummy_epochs)
            logger.info("MLEstimator __init__: Instantiated DummyMLClassifier.")
        else:
            raise ValueError(f"Unsupported 'model_type' for MLEstimator: '{self.model_type}'. "
                             f"Supported types: RandomForestClassifier, GradientBoostingClassifier, dummy_classifier.")

        self.model_save_path_root = None # This will be set by EstimatorFactory
        self.is_trained = False # Initialize training status

        logger.info(f"MLEstimator initialized with config: {self.config}")
        

    def train(self, data: pd.DataFrame, labels: pd.Series):
        """
        Trains the ML model.
        Assumes data contains the features specified in self.features.
        """
        
        if self.model is None:
            raise RuntimeError("ML Model not initialized. Check estimator_config for 'model_type'.")
        logger.info(f"Training {self.model_type} model...")

        if not all(f in data.columns for f in self.features):
            missing_features = [f for f in self.features if f not in data.columns]
            logger.info(f"{data}")
            logger.error(f"Missing required features for training: {missing_features}. Available: {list(data.columns)}")
            raise ValueError(f"Training data must contain all specified features: {self.features}")

        X = data[self.features]
        y = labels

        logger.info(f"Starting training for ML estimator ({self.model_type}) on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"ML estimator ({self.model_type}) trained successfully.")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predicts peak occurrences using the trained ML model.
        """
        if self.model is None:
            raise RuntimeError("ML Model not trained or loaded.")

        if not self.is_trained: # Removed 'or self.model is None' as it's redundant with first check
            raise RuntimeError("ML estimator not trained. Call train() first.")

        if not all(f in data.columns for f in self.features):
            missing_features = [f for f in self.features if f not in data.columns]
            logger.error(f"Missing required features for prediction: {missing_features}. Available: {list(data.columns)}")
            raise ValueError(f"Prediction data must contain all specified features: {self.features}")

        X = data[self.features]

        logger.info(f"Predicting with ML estimator ({self.model_type}) on {len(X)} samples...")

        # --- MODIFIED LOGIC HERE ---
        if hasattr(self.model, 'predict_proba') and self.prediction_threshold is not None:
            # Use predict_proba and apply the custom threshold
            probabilities = self.model.predict_proba(X)[:, 1] # Get probabilities for the positive class
            predictions = (probabilities >= self.prediction_threshold).astype(int)
            logger.debug(f"Applied custom prediction threshold ({self.prediction_threshold}) to probabilities for {self.model_type}.")
        else:
            # If model doesn't support predict_proba, or no custom threshold is set,
            # use the model's default predict method (which usually uses a 0.5 threshold internally)
            predictions = self.model.predict(X)
            if self.prediction_threshold is None:
                 logger.debug(f"Using default predict() method for {self.model_type} (no custom threshold specified).")
            else:
                 logger.warning(f"Model type '{self.model_type}' does not support 'predict_proba'. Cannot apply custom threshold {self.prediction_threshold}. Using direct predict().")


        predictions = pd.Series(predictions, index=data.index) # Ensure index is preserved
        predictions.name = 'predicted_peak' # Name the series for clarity

        logger.info(f"ML prediction completed for {len(data)} samples. Found {predictions.sum()} peaks.")
        return predictions


    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """
        Predicts probabilities of peak occurrences using the trained ML model.
        """
        
        if self.model is None:
            raise RuntimeError("No model to save.") # Should be "No model trained or loaded" perhaps

        if not self.is_trained: # Removed 'or self.model is None' for redundancy
            raise RuntimeError("ML estimator not trained. Call train() first.")

        if not all(f in data.columns for f in self.features):
            missing_features = [f for f in self.features if f not in data.columns]
            logger.error(f"Missing required features for probability prediction: {missing_features}. Available: {list(data.columns)}")
            raise ValueError(f"Prediction data must contain all specified features: {self.features}")

        X = data[self.features]

        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(f"Model type {self.model_type} does not support probability prediction.")

        logger.info(f"Predicting probabilities with ML estimator ({self.model_type}) on {len(X)} samples...")
        probabilities = self.model.predict_proba(X)
        # For binary classification, predict_proba usually returns an array of shape (n_samples, 2).
        # You often want the probability of the positive class (column 1).
        # Let's ensure this is a 1D Series for consistency
        probabilities_series = pd.Series(probabilities[:, 1], index=data.index)
        probabilities_series.name = 'predicted_proba'
        logger.info(f"ML probability prediction completed for {len(data)} samples.")
        return probabilities_series

    def save_model(self, path: Union[str, Path]): # Updated type hint
        """Saves the trained ML model and its metadata."""

        # Removed redundant check, the next one is sufficient
        if self.model is None or not self.is_trained:
            raise RuntimeError("Cannot save, ML estimator not trained.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        model_weights_filename = "model_weights.joblib" # Use .joblib for scikit-learn
        model_weights_path = save_path / model_weights_filename
        metadata_path = save_path / "metadata.yaml"

        model_weights_file_in_metadata = None # Initialize before try block

        try:
            logger.debug(f"Attempting to save model weights to {model_weights_path}...")
            joblib.dump(self.model, model_weights_path)
            logger.info(f"ML model weights saved to {model_weights_path}")
            model_weights_file_in_metadata = model_weights_filename
        except Exception as e:
            logger.error(f"Failed to save model weights using joblib for {self.model_type}: {e}", exc_info=True)
            # You might want to re-raise this if failure to save weights is critical
            # raise
            pass # Continue to try and save metadata

        logger.debug(f"Preparing metadata for estimator '{self.model_type}'.")
        metadata = { # This is where metadata should be assigned
            'model_type': self.model_type,
            'features': list(self.features),
            'epochs': getattr(self, 'epochs', None),
            'prediction_threshold': getattr(self, 'prediction_threshold', None),
            'batch_size': getattr(self, 'batch_size', None),
            'learning_rate': getattr(self, 'learning_rate', None),
            'model_weights_file': model_weights_file_in_metadata,
            'trained_on_features': list(self.features),
            'model_params': self.model_params # Crucial: Save the model initialization parameters
        }
        logger.debug(f"Metadata prepared: {metadata}")

        try:
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, indent=4, default_flow_style=False)
            logger.info(f"ML estimator metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {metadata_path}: {e}", exc_info=True)
            raise # Re-raise this, as metadata saving is critical

        logger.info(f"ML estimator model saved successfully to {save_path}.")
        
        
    def load_model(self, path: Union[str, Path]): # Updated type hint
        """Loads a trained ML model and its metadata."""
        load_path = Path(path)
        metadata_path = load_path / "metadata.yaml"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"ML estimator metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Load configuration from metadata to re-initialize the model shell
        self.model_type = metadata.get('model_type')
        self.features = metadata.get('features', [])
        self.epochs = metadata.get('epochs')
        self.prediction_threshold = metadata.get('prediction_threshold')
        self.batch_size = metadata.get('batch_size')
        self.learning_rate = metadata.get('learning_rate')
        # Retrieve the original model parameters from metadata
        loaded_model_params = metadata.get('model_params', {})

        # Re-initialize the model shell using the *loaded* parameters from metadata
        if self.model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier(**loaded_model_params)
        elif self.model_type == 'GradientBoostingClassifier':
            self.model = GradientBoostingClassifier(**loaded_model_params)
        elif self.model_type == 'dummy_classifier': # If you want to load dummy models too
            # Need to ensure n_features and epochs are properly derived or stored in metadata
            n_features_dummy = len(self.features) # Assuming features are defined by this point
            epochs_dummy = self.epochs # Assuming epochs is loaded from metadata or default
            self.model = DummyMLClassifier(n_features=n_features_dummy, epochs=epochs_dummy)
        else:
            raise ValueError(f"Unsupported ML model type for loading: {self.model_type}")

        # Load model weights from the specified file
        if 'model_weights_file' in metadata and metadata['model_weights_file']:
            model_weights_path = load_path / metadata['model_weights_file']
            if model_weights_path.exists():
                try:
                    self.model = joblib.load(model_weights_path) # Overwrite with loaded model
                    logger.info(f"ML model weights loaded from {model_weights_path}")
                except Exception as e:
                    logger.error(f"Failed to load model weights from {model_weights_path}: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to load model weights: {e}")
            else:
                logger.warning(f"Model weights file not found at {model_weights_path}. Model will use initialized state from metadata.")
        else:
            logger.info("No specific model weights file found in metadata. Using default initialized model.")

        self.is_trained = True
        logger.info(f"ML estimator model and configuration loaded from {load_path}")
