import unittest
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
# Assuming the file is located at peak_estimators/strategies/base_estimator.py
from peak_estimators.strategies.base_estimator import BasePeakEstimator 
from typing import Dict, Any, Tuple, Optional, List, Union
from pathlib import Path


# Helper class definitions mirroring the required BasePeakEstimator imports
# You may need to adjust the import path for BasePeakEstimator if it's incorrect.
# If BasePeakEstimator is abstract, its methods must be implemented.

class TestBasePeakEstimator(unittest.TestCase):

    def test_abstract_methods_enforcement(self):
        """
        Verify that BasePeakEstimator cannot be instantiated directly
        and that abstract methods must be implemented by subclasses.
        """
        # Test direct instantiation raises TypeError
        with self.assertRaises(TypeError) as context:
            BasePeakEstimator(config={})
        
        # Check for expected error message part
        self.assertIn("Can't instantiate abstract class BasePeakEstimator with abstract method", str(context.exception))

    def test_concrete_implementation(self):
        """
        Test that a subclass implementing all abstract methods can be instantiated
         and its methods can be called without errors (even with mock logic).
        """
        # Define the concrete class, implementing ALL abstract methods
        class ConcreteEstimator(BasePeakEstimator):
            def __init__(self, config: Dict[str, Any]):
                super().__init__(config)
                # self.trained is now managed by the base class self.is_trained

            # 1. Implementation of abstract method 'train'
            def train(self, data: pd.DataFrame, labels: pd.Series) -> None:
                # Mock training logic
                self.is_trained = True

            # 2. Implementation of abstract method 'predict'
            def predict(self, data: pd.DataFrame) -> pd.Series:
                """Mock implementation of predict. Returns all non-peaks (0)"""
                # Should return a Series of predictions (0s or 1s)
                return pd.Series(0, index=data.index)

            # 3. Implementation of abstract method 'save_model'
            def save_model(self, path: Union[str, Path]) -> None:
                """Mock implementation of save_model."""
                pass # Just needs to exist

            # 4. Implementation of abstract method 'load_model'
            def load_model(self, path: Union[str, Path]):
                """Mock implementation of load_model. Must exist."""
                pass # Just needs to exist
            
        mock_config = {'estimator': {'name': 'Test'}}
        
        # Instantiation now succeeds because all abstract methods are implemented.
        estimator = ConcreteEstimator(mock_config)
        
        # Test instantiation success
        self.assertIsInstance(estimator, BasePeakEstimator)
        
        # Test train method
        mock_df = pd.DataFrame(np.random.rand(50, 2), columns=['price', 'feature1'])
        mock_labels = pd.Series(np.zeros(50), name='is_peak')
        
        estimator.train(mock_df, mock_labels)
        self.assertTrue(estimator.is_fitted()) # Use the base class method for checking
        
        # Test predict method
        predictions = estimator.predict(mock_df)
        self.assertIsInstance(predictions, pd.Series)
        self.assertEqual(len(predictions), 50)
        
        # Test predict_proba raises NotImplementedError (default base class behavior)
        with self.assertRaises(NotImplementedError):
            estimator.predict_proba(mock_df)

        # Test save/load mock methods don't raise errors
        estimator.save_model("dummy/path.pkl")
        estimator.load_model("dummy/path.pkl")


# If running this file directly
if __name__ == '__main__':
    unittest.main()
# import unittest
# import pandas as pd
# import numpy as np
# from abc import ABC, abstractmethod
# from peak_estimators.strategies.base_estimator import BasePeakEstimator
# from typing import Dict, Any, Tuple, Optional, List
# 
# class TestBasePeakEstimator(unittest.TestCase):
# 
#     def test_abstract_methods_enforcement(self):
#         """
#         Verify that BasePeakEstimator cannot be instantiated directly
#         and that abstract methods must be implemented by subclasses.
#         """
#         # Test direct instantiation raises TypeError
#         with self.assertRaises(TypeError) as context:
#             BasePeakEstimator(config={})
#         
#         self.assertIn("Can't instantiate abstract class BasePeakEstimator with abstract method", str(context.exception))
# 
#     def test_concrete_implementation(self):
#         """
#         Test that a subclass implementing all abstract methods can be instantiated
#         and its methods can be called without errors (even with mock logic).
#         """
#         class ConcreteEstimator(BasePeakEstimator):
#             def __init__(self, config: Dict[str, Any]):
#                 super().__init__(config)
#                 self.trained = False
# 
#             def train(self, data: pd.DataFrame, features: List[str], target: str) -> None:
#                 self.trained = True
#             
#             def estimate_peaks(self, data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
#                 # Returns dummy indices and confidence scores
#                 return np.array([10, 20, 30]), np.array([0.9, 0.8, 0.7])
# 
#         mock_config = {'estimator': {'name': 'Test'}}
#         estimator = ConcreteEstimator(mock_config)
#         
#         # Test instantiation success
#         self.assertIsInstance(estimator, BasePeakEstimator)
#         
#         # Test train method
#         mock_df = pd.DataFrame(np.random.rand(50, 2), columns=['price', 'is_peak'])
#         estimator.train(mock_df, ['price'], 'is_peak')
#         self.assertTrue(estimator.trained)
#         
#         # Test peak estimation method
#         peaks, confidence = estimator.estimate_peaks(mock_df, ['price'])
#         self.assertEqual(len(peaks), 3)
#         self.assertEqual(len(confidence), 3)
