import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
# Assuming MLEstimator is correctly imported from its location
from peak_estimators.strategies.ml_estimator import MLEstimator
from sklearn.ensemble import RandomForestClassifier 

class TestMLEstimator(unittest.TestCase):

    def setUp(self):
        """Set up mock data and configuration for the ML Estimator."""
        self.features = ['feature_1', 'feature_2'] # Features used in the mock data
        self.target = 'is_peak'

        self.mock_config = {
            'model_type': 'RandomForestClassifier', 
            'params': {'n_estimators': 100}, 
            # FIX 1: Explicitly set the features to match the mock data (for feature validation in predict/train)
            'features': self.features, 
            'target_col': self.target,
            'train_test_split': 0.8
        }
        
        # MLEstimator init will now succeed and use the correct feature names
        self.estimator = MLEstimator(self.mock_config)
        
        # Mock dataset
        data = {
            'date': pd.date_range('2023-01-01', periods=100),
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100),
            'is_peak': np.random.randint(0, 2, 100) # Binary target
        }
        self.data_df = pd.DataFrame(data)

    
    # Patch the model class that MLEstimator actually instantiates in __init__
    @patch('peak_estimators.strategies.ml_estimator.RandomForestClassifier')
    def test_train_model_initialization_and_fit(self, MockRFC):
        """Test if the correct model is initialized and the fit method is called."""

        # Create a mock instance of the RandomForestClassifier model
        mock_model_instance = MagicMock()
        MockRFC.return_value = mock_model_instance
        
        # Re-initialize the estimator to ensure it uses the MockRFC
        self.estimator = MLEstimator(self.mock_config)

        # Train the model. Argument count is correct (data, labels)
        self.estimator.train(self.data_df.copy(), pd.Series(self.data_df[self.target])) 
        
        # Assert model instance was created with correct params
        MockRFC.assert_called_once_with(n_estimators=100)
        
        # Assert fit was called
        mock_model_instance.fit.assert_called_once()
        
        # Assert the estimator stored the trained model
        self.assertEqual(self.estimator.model, mock_model_instance)
        
        # Clean up is_trained state
        self.estimator.is_trained = False


    def test_train_invalid_model_type(self):
        """Test handling of an unsupported model type."""
        # Create a new config for the test to ensure init is called with the bad type
        bad_config = self.mock_config.copy()
        bad_config['model_type'] = 'UnsupportedModel'
        
        with self.assertRaises(ValueError):
            MLEstimator(bad_config)

    
    def test_estimate_peaks_prediction(self):
        """
        Test peak estimation behavior by calling predict and predict_proba 
        and verifying the peak indices and confidence scores.
        """
        
        # Mock the underlying model instance
        mock_model = MagicMock()
        
        mock_model.predict.return_value = np.array([1, 0, 1, 0])
        # Mock predict_proba: [prob_class_0, prob_class_1]. We care about class 1 (index 1)
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        
        # Inject the mock model and set the training flag
        self.estimator.model = mock_model 
        self.estimator.is_trained = True 
        
        # Use a small test slice (which uses DatetimeIndex)
        test_df = self.data_df.head(4).copy()
        
        # 1. Get predictions (binary) -> calls mock_model.predict_proba (Count: 1)
        predictions = self.estimator.predict(test_df)
        
        # 2. Get probabilities (confidence) -> calls mock_model.predict_proba (Count: 2)
        probabilities = self.estimator.predict_proba(test_df)
        
        # 3. Derive the "peaks" (indices where prediction is 1)
        peak_mask = (predictions == 1)
        peak_indices = predictions[peak_mask].index.to_numpy()
        
        # 4. Derive confidence for those peaks
        peak_confidence = probabilities.loc[peak_mask].to_numpy()

        # Assertions
        # FIX 2: Correct assertion for being called twice.
        self.assertEqual(mock_model.predict_proba.call_count, 2)
        
        # Check data integrity
        np.testing.assert_array_equal(peak_indices, test_df.index[[0, 2]])
        np.testing.assert_array_almost_equal(peak_confidence, np.array([0.9, 0.7]))
