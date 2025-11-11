# tradebook_pipeline/synthetic_data/synthesizer.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Import augmenters
from synthetic_data.augmenters.noise_augmenter import NoiseAugmenter


class SyntheticDataValidator:
    """Validates synthetic data quality against real data"""
    
    def __init__(self, real_data: pd.DataFrame):
        self.real_data = real_data
        self.numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
    def calculate_distribution_similarity(self, synthetic_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate KL divergence and statistical similarity metrics"""
        similarities = {}
        
        for col in self.numeric_cols:
            if col in synthetic_data.columns:
                try:
                    # KS test for distribution similarity
                    ks_stat, ks_pval = stats.ks_2samp(
                        self.real_data[col].dropna(),
                        synthetic_data[col].dropna()
                    )
                    
                    # Mean and std comparison
                    real_mean = self.real_data[col].mean()
                    synth_mean = synthetic_data[col].mean()
                    real_std = self.real_data[col].std()
                    synth_std = synthetic_data[col].std()
                    
                    similarities[col] = {
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pval,
                        'mean_diff_pct': abs(synth_mean - real_mean) / abs(real_mean) * 100 if real_mean != 0 else 0,
                        'std_diff_pct': abs(synth_std - real_std) / abs(real_std) * 100 if real_std != 0 else 0
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate similarity for column {col}: {e}")
                    
        return similarities
    
    def validate_synthetic_data(self, synthetic_data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive validation of synthetic data"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check column consistency
        missing_cols = set(self.real_data.columns) - set(synthetic_data.columns)
        if missing_cols:
            validation_results['issues'].append(f"Missing columns: {missing_cols}")
            validation_results['is_valid'] = False
        
        # Check for NaN values
        nan_cols = synthetic_data.columns[synthetic_data.isna().any()].tolist()
        if nan_cols:
            validation_results['warnings'].append(f"Columns with NaN values: {nan_cols}")
        
        # Check distribution similarity
        similarities = self.calculate_distribution_similarity(synthetic_data)
        validation_results['metrics']['distribution_similarity'] = similarities
        
        # Flag distributions that are too different
        for col, metrics in similarities.items():
            if metrics['ks_pvalue'] < 0.01:  # Significantly different distribution
                validation_results['warnings'].append(
                    f"Column '{col}' has significantly different distribution (p={metrics['ks_pvalue']:.4f})"
                )
            if metrics['mean_diff_pct'] > 20:  # Mean differs by >20%
                validation_results['warnings'].append(
                    f"Column '{col}' mean differs by {metrics['mean_diff_pct']:.1f}%"
                )
        
        # Check data ranges
        for col in self.numeric_cols:
            if col in synthetic_data.columns:
                real_min, real_max = self.real_data[col].min(), self.real_data[col].max()
                synth_min, synth_max = synthetic_data[col].min(), synthetic_data[col].max()
                
                if synth_min < real_min * 0.8 or synth_max > real_max * 1.2:
                    validation_results['warnings'].append(
                        f"Column '{col}' has values outside expected range"
                    )
        
        return validation_results['is_valid'], validation_results


class SyntheticDataGenerator:
    """
    Generates synthetic time-series data locally with enhanced validation and benchmarking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the generator with configuration and validation"""
        self.config = config
        
        self.generator_model_type = self.config.get('generator_model_type', 'Gaussian')
        # Don't hardcode feature_cols - will be determined from data
        self.feature_cols = None
        self.seq_len = self.config.get('seq_len', 24)
        self.training_params = self.config.get('training_params', {})
        
        # Model save path
        self.model_save_path = Path(self.training_params.get('model_save_path', 'models/synthetic_data/'))
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Augmentation
        self.augmenter = NoiseAugmenter(self.config.get('augmentation', {})) \
            if self.config.get('augmentation', {}).get('enabled', False) else None
        
        # Model state
        self.generator_model = None
        self.is_generator_trained = False
        self.scaler = None
        self.validator = None
        self.training_metadata = {}

        logger.info(f"SyntheticDataGenerator initialized. Generator type: {self.generator_model_type}")
        if self.augmenter:
            logger.info(f"Data augmentation enabled: {self.augmenter.noise_type}")

    # def _detect_feature_columns(self, data: pd.DataFrame) -> List[str]:
    #     """Automatically detect feature columns from data"""
    #     # Exclude known non-feature columns
    #     exclude_cols = {'timestamp', 'date', 'datetime', 'time', 'is_synthetic', 
    #                    'sequence_id', 'is_peak'}
    #     
    #     # Get numeric and categorical columns
    #     numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    #     categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    #     
    #     # Combine and filter
    #     feature_cols = [col for col in numeric_cols + categorical_cols 
    #                    if col.lower() not in exclude_cols]
    #     
    #     logger.info(f"Detected feature columns: {feature_cols}")
    #     return feature_cols
    
    def _detect_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Automatically detect feature columns from data"""
        # Apply side column transformation if it exists
        if 'side' in data.columns:
            logger.info("Transforming 'side' column...")
            side_counts_before = data['side'].value_counts().to_dict()
            logger.info(f"  Before mapping: {side_counts_before}")
            
            data['side'] = data['side'].map({'ask': 'a', 'bid': 'b'})
            
            side_counts_after = data['side'].value_counts().to_dict()
            logger.info(f"  After mapping: {side_counts_after}")
            
            # Check for unmapped values
            unmapped = data['side'].isnull().sum()
            if unmapped > 0:
                logger.warning(f"  WARNING: {unmapped} values could not be mapped!")
            
            # Rename column
            logger.info("Renaming column: 'side' -> 'type'")
            data.rename(columns={'side': 'type'}, inplace=True)
        
        # Exclude known non-feature columns
        exclude_cols = {'timestamp', 'date', 'datetime', 'time', 'is_synthetic', 
                       'sequence_id', 'is_peak', 'symbol', 'source'}
        
        # Get numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Combine and filter
        feature_cols = [col for col in numeric_cols + categorical_cols 
                       if col.lower() not in exclude_cols]
        
        logger.info(f"Detected feature columns: {feature_cols}")
        return feature_cols
      

    def _validate_training_data(self, data: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """Validate and prepare training data"""
        if data.empty:
            logger.error("Training data is empty")
            return False, data
        
        # Detect feature columns if not set
        if self.feature_cols is None:
            self.feature_cols = self._detect_feature_columns(data)
        
        # Validate feature columns exist
        missing_cols = [col for col in self.feature_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns in training data: {missing_cols}")
            # Use only available columns
            self.feature_cols = [col for col in self.feature_cols if col in data.columns]
            
            if not self.feature_cols:
                logger.error("No valid feature columns found in training data")
                return False, data
        
        # Prepare training data
        training_data = data[self.feature_cols].copy()
        
        # Handle missing values
        if training_data.isna().any().any():
            logger.warning("Training data contains NaN values. Filling with forward fill then mean.")
            training_data = training_data.fillna(method='ffill').fillna(training_data.mean())
        
        # Store metadata
        self.training_metadata = {
            'n_samples': len(training_data),
            'n_features': len(self.feature_cols),
            'feature_columns': self.feature_cols,
            'data_types': training_data.dtypes.astype(str).to_dict(),
            'numeric_columns': training_data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': training_data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        logger.info(f"Training data validated: {self.training_metadata['n_samples']} samples, "
                   f"{self.training_metadata['n_features']} features")
        
        return True, training_data

    def train_generator(self, raw_data: pd.DataFrame, model_type: str = None):
        """
        Train the synthetic data generator with enhanced validation.
        """
        train_model_type = model_type if model_type else self.generator_model_type
        logger.info(f"Starting training for {train_model_type} generator...")

        # Validate and prepare data
        is_valid, training_data = self._validate_training_data(raw_data)
        if not is_valid:
            raise ValueError("Training data validation failed")

        # Apply augmentation if enabled
        if self.augmenter:
            logger.info("Applying augmentation to training data")
            training_data = self.augmenter.apply_augmentation(training_data)

        # Initialize validator for quality checks
        self.validator = SyntheticDataValidator(training_data)

        # Train based on model type
        if train_model_type == 'Gaussian':
            self._train_gaussian_model(training_data)
        elif train_model_type == 'CTGAN':
            self._train_ctgan_model(training_data)
        elif train_model_type == 'TimeGAN':
            self._train_timegan_model(training_data)
        else:
            raise ValueError(f"Unsupported generator model type: {train_model_type}")

        self.is_generator_trained = True
        self.generator_model_type = train_model_type
        self._save_generator_model()
        
        logger.info(f"Generator '{self.generator_model_type}' training completed successfully")

    def _train_gaussian_model(self, training_data: pd.DataFrame):
        """Train Gaussian statistical model with normalization"""
        logger.info("Training Gaussian statistical model...")
        
        numeric_cols = self.training_metadata['numeric_columns']
        categorical_cols = self.training_metadata['categorical_columns']
        
        # Fit scaler on numeric data
        if numeric_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(training_data[numeric_cols])
        
        # Store statistics
        self.generator_model = {
            'model_type': 'Gaussian',
            'numeric_stats': {},
            'categorical_stats': {},
            'correlations': None
        }
        
        # Numeric columns statistics
        for col in numeric_cols:
            self.generator_model['numeric_stats'][col] = {
                'mean': float(training_data[col].mean()),
                'std': float(training_data[col].std()),
                'min': float(training_data[col].min()),
                'max': float(training_data[col].max()),
                'median': float(training_data[col].median()),
                'q25': float(training_data[col].quantile(0.25)),
                'q75': float(training_data[col].quantile(0.75))
            }
        
        # Categorical columns statistics
        for col in categorical_cols:
            value_counts = training_data[col].value_counts(normalize=True)
            self.generator_model['categorical_stats'][col] = {
                'values': value_counts.index.tolist(),
                'probabilities': value_counts.values.tolist()
            }
        
        # Store correlations for numeric features
        if len(numeric_cols) > 1:
            self.generator_model['correlations'] = training_data[numeric_cols].corr().to_dict()
        
        logger.info("Gaussian model trained with feature statistics and correlations")

    def _train_ctgan_model(self, training_data: pd.DataFrame):
        """Placeholder for CTGAN training"""
        logger.info("Simulating CTGAN training...")
        # This would integrate with actual CTGAN library
        self._train_gaussian_model(training_data)  # Fallback to Gaussian for now
        self.generator_model['model_type'] = 'CTGAN'

    def _train_timegan_model(self, training_data: pd.DataFrame):
        """Placeholder for TimeGAN training"""
        logger.info("Simulating TimeGAN training...")
        # This would integrate with actual TimeGAN implementation
        self._train_gaussian_model(training_data)  # Fallback to Gaussian for now
        self.generator_model['model_type'] = 'TimeGAN'

    def generate_data(self, num_sequences: int, raw_data_base: Optional[pd.DataFrame] = None, 
                     validate: bool = True) -> pd.DataFrame:
        """
        Generate synthetic data with validation.
        """
        if not self.is_generator_trained:
            if raw_data_base is not None and not raw_data_base.empty:
                logger.warning("Generator not trained. Training now with provided data...")
                self.train_generator(raw_data_base)
            else:
                raise RuntimeError("Generator not trained and no base data provided")

        logger.info(f"Generating {num_sequences} synthetic sequences using {self.generator_model_type}...")

        # Generate data
        if self.generator_model_type in ['Gaussian', 'CTGAN', 'TimeGAN']:
            synthetic_data = self._generate_gaussian_data(num_sequences)
        else:
            raise ValueError(f"Generation not implemented for: {self.generator_model_type}")

        # Validate generated data
        if validate and self.validator:
            is_valid, validation_results = self.validator.validate_synthetic_data(synthetic_data)
            
            # Log validation results
            logger.info("Synthetic data validation results:")
            logger.info(f"  Valid: {is_valid}")
            
            if validation_results['warnings']:
                logger.warning(f"  Warnings: {len(validation_results['warnings'])}")
                for warning in validation_results['warnings'][:5]:  # Show first 5
                    logger.warning(f"    - {warning}")
            
            if validation_results['issues']:
                logger.error(f"  Issues: {len(validation_results['issues'])}")
                for issue in validation_results['issues']:
                    logger.error(f"    - {issue}")
            
            # Save validation report
            self._save_validation_report(validation_results)
        
        logger.info(f"Synthetic data generation completed: {len(synthetic_data)} samples")
        return synthetic_data

    def _generate_gaussian_data(self, num_sequences: int) -> pd.DataFrame:
        """Generate data from Gaussian model"""
        synthetic_data_list = []
        
        numeric_stats = self.generator_model['numeric_stats']
        categorical_stats = self.generator_model['categorical_stats']
        
        total_samples = num_sequences * self.seq_len
        
        # Generate numeric features
        synthetic_dict = {}
        
        for col, stats in numeric_stats.items():
            # Generate from truncated normal distribution
            values = np.random.normal(stats['mean'], stats['std'], total_samples)
            # Clip to observed range
            values = np.clip(values, stats['min'], stats['max'])
            synthetic_dict[col] = values
        
        # Generate categorical features
        for col, stats in categorical_stats.items():
            values = np.random.choice(
                stats['values'],
                size=total_samples,
                p=stats['probabilities']
            )
            synthetic_dict[col] = values
        
        # Create DataFrame
        synthetic_data = pd.DataFrame(synthetic_dict)
        
        # Add metadata column
        synthetic_data['is_synthetic'] = True
        
        # Add sequence IDs
        sequence_ids = np.repeat(range(num_sequences), self.seq_len)
        synthetic_data['sequence_id'] = sequence_ids
        
        return synthetic_data

    def _save_generator_model(self):
        """Save generator model and metadata"""
        if not self.generator_model:
            logger.warning("No generator model to save")
            return
        
        model_file = self.model_save_path / f"{self.generator_model_type.lower()}_generator.json"
        
        try:
            save_data = {
                'model_type': self.generator_model_type,
                'generator_model': self.generator_model,
                'training_metadata': self.training_metadata,
                'feature_columns': self.feature_cols,
                'seq_len': self.seq_len
            }
            
            with open(model_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.info(f"Generator model saved to {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save generator model: {e}")

    def _save_validation_report(self, validation_results: Dict[str, Any]):
        """Save validation report"""
        report_file = self.model_save_path / f"validation_report_{self.generator_model_type.lower()}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            logger.info(f"Validation report saved to {report_file}")
        except Exception as e:
            logger.warning(f"Failed to save validation report: {e}")

    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of generator capabilities and training"""
        return {
            'model_type': self.generator_model_type,
            'is_trained': self.is_generator_trained,
            'feature_columns': self.feature_cols,
            'sequence_length': self.seq_len,
            'training_metadata': self.training_metadata,
            'augmentation_enabled': self.augmenter is not None
        }


# # tradebook_pipeline/synthetic_data/synthesizer.py
# 
# import pandas as pd
# import numpy as np
# from typing import Dict, Any, List, Optional
# from loguru import logger
# from pathlib import Path
# import random # For dummy generator
# import time # For simulating training time
# 
# # Import augmenters
# from synthetic_data.augmenters.noise_augmenter import NoiseAugmenter
# 
# 
# class SyntheticDataGenerator:
#     """
#     Generates synthetic time-series data locally.
#     Supports different generator models and data augmentation.
#     """
#     def __init__(self, config: Dict[str, Any]):
#         """
#         Initializes the generator with a given configuration.
#         """
#         self.config = config
#         
#         self.generator_model_type = self.config.get('generator_model_type', 'Gaussian')
#         self.feature_cols = self.config.get('feature_cols', ['price', 'volume', 'type'])
#         self.seq_len = self.config.get('seq_len', 24)
#         self.training_params = self.config.get('training_params', {})
#         
#         # Define the path where trained generator models will be saved/loaded
#         self.model_save_path = Path(self.training_params.get('model_save_path', 'models/synthetic_data/'))
#         self.model_save_path.mkdir(parents=True, exist_ok=True)
# 
#         self.augmenter = NoiseAugmenter(self.config.get('augmentation', {})) \
#             if self.config.get('augmentation', {}).get('enabled', False) else None
#         
#         self.generator_model = None # This will hold the trained generator model
#         self.is_generator_trained = False
# 
#         logger.info(f"SyntheticDataGenerator initialized. Generator type: {self.generator_model_type}")
#         if self.augmenter:
#             logger.info(f"Data augmentation is enabled with type: {self.augmenter.noise_type}")
# 
#         self._load_generator_model()
# 
#     def _load_generator_model(self):
#         """
#         Loads a pre-trained generator model if available.
#         This is a placeholder for actual model loading (e.g., TensorFlow, PyTorch, or pickle).
#         """
#         model_file = self.model_save_path / f"{self.generator_model_type.lower()}_generator_model.pkl"
#         if model_file.exists():
#             try:
#                 self.generator_model = {} # A simple dictionary to mimic a loaded model
#                 self.is_generator_trained = True
#                 logger.info(f"Successfully loaded pre-trained {self.generator_model_type} generator from {model_file}.")
#             except Exception as e:
#                 logger.warning(f"Failed to load pre-trained {self.generator_model_type} generator from {model_file}: {e}")
#         else:
#             logger.info(f"No pre-trained {self.generator_model_type} generator found at {model_file}.")
# 
#     def _save_generator_model(self):
#         """
#         Saves the trained generator model.
#         Placeholder for actual model saving.
#         """
#         if self.generator_model:
#             model_file = self.model_save_path / f"{self.generator_model_type.lower()}_generator_model.pkl"
#             try:
#                 model_file.touch() # This will create an empty file
#                 logger.info(f"Successfully saved {self.generator_model_type} generator to {model_file}.")
#             except Exception as e:
#                 logger.error(f"Failed to save {self.generator_model_type} generator to {model_file}: {e}")
#         else:
#             logger.warning("No generator model to save.")
# 
#     def train_generator(self, raw_data: pd.DataFrame, model_type: str = None):
#         """
#         Trains the synthetic data generator model.
#         
#         Args:
#             raw_data (pd.DataFrame): The real time-series data to train the generator on.
#             model_type (str, optional): Override the configured generator model type.
#         """
#         train_model_type = model_type if model_type else self.generator_model_type
#         logger.info(f"Starting training for {train_model_type} generator with {len(raw_data)} samples...")
# 
#         if raw_data.empty:
#             logger.warning("Raw data is empty. Cannot train generator.")
#             return
# 
#         if self.augmenter:
#             logger.info("Applying augmentation to raw data before training.")
#             raw_data = self.augmenter.apply_augmentation(raw_data)
# 
#         training_data = raw_data[self.feature_cols].copy()
#         
#         if train_model_type == 'TimeGAN':
#             logger.info("Simulating TimeGAN training...")
#             time.sleep(self.training_params.get('epochs', 100) * 0.05)
#             self.generator_model = "DummyTimeGANTrainedModel"
#         elif train_model_type == 'CTGAN':
#             logger.info("Simulating CTGAN training...")
#             time.sleep(self.training_params.get('epochs', 100) * 0.03)
#             self.generator_model = {
#                 'means': training_data.mean().to_dict(),
#                 'stds': training_data.std().to_dict(),
#                 'min_vals': training_data.min().to_dict(),
#                 'max_vals': training_data.max().to_dict()
#             }
#         elif train_model_type == 'Gaussian':
#             logger.info("Training a simple Gaussian (statistical) model...")
#             self.generator_model = {
#                 'means': training_data.mean().to_dict(),
#                 'stds': training_data.std().to_dict(),
#                 'min_vals': training_data.min().to_dict(),
#                 'max_vals': training_data.max().to_dict()
#             }
#             logger.info("Gaussian model trained by storing statistics.")
#         else:
#             logger.error(f"Unsupported generator model type: {train_model_type}")
#             raise ValueError(f"Unsupported generator model type: {train_model_type}")
# 
#         self.is_generator_trained = True
#         self.generator_model_type = train_model_type
#         self._save_generator_model()
#         logger.info(f"Generator model '{self.generator_model_type}' training completed.")
# 
#     def generate_data(self, num_sequences: int, raw_data_base: Optional[pd.DataFrame] = None, system_characteristics: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
#         """
#         Generates synthetic time-series data.
#         """
#         if not self.is_generator_trained:
#             logger.warning("Generator model not trained. Attempting to train with provided raw_data_base or dummy data.")
#             if raw_data_base is not None and not raw_data_base.empty:
#                 self.train_generator(raw_data_base)
#             else:
#                 logger.error("Cannot generate synthetic data: Generator not trained and no raw_data_base provided.")
#                 raise RuntimeError("Generator not trained and no base data for quick training.")
# 
#         logger.info(f"Generating {num_sequences} synthetic sequences using {self.generator_model_type} generator...")
# 
#         synthetic_data_list = []
# 
#         if self.generator_model_type == 'TimeGAN':
#             logger.info("Simulating TimeGAN generation...")
#             for _ in range(num_sequences):
#                 sequence_data = {}
#                 base_values = {
#                     'price': 50.0,
#                     'volume': 100.0,
#                     'type': 'a'
#                 }
#                 
#                 for col in self.feature_cols:
#                     if col == 'date': continue
#                     if col == 'is_peak':
#                         sequence_data[col] = np.random.choice([0, 1], size=self.seq_len, p=[0.95, 0.05])
#                     elif col in base_values:
#                         if col == 'price':
#                             sequence_data[col] = np.cumsum(np.random.randn(self.seq_len) * 2) + base_values[col]
#                         elif col == 'volume':
#                             sequence_data[col] = np.cumsum(np.random.randn(self.seq_len) * 50) + base_values[col]
#                             
#                         elif col == 'type':
#                             sequence_data[col] = np.random.choice(['a', 'b'], size=self.seq_len, p=[0.7, 0.3])
#                         else:
#                             sequence_data[col] = np.cumsum(np.random.randn(self.seq_len) * 1) + base_values[col]
#                     else:
#                         sequence_data[col] = np.random.uniform(50, 100, self.seq_len)
#                 
#                 # Replace the pd.date_range line with:
#                 sequence = pd.DataFrame(sequence_data)#, index=range(_ * self.seq_len, (_ + 1) * self.seq_len))
# 
#                 # sequence = pd.DataFrame(sequence_data, index=pd.date_range(start=pd.Timestamp.now(), periods=self.seq_len, freq='h'))
#                 sequence['is_synthetic'] = True
#                 synthetic_data_list.append(sequence)
# 
#         elif self.generator_model_type == 'CTGAN':
#             logger.info("Simulating CTGAN generation...")
#             for _ in range(num_sequences):
#                 if not self.generator_model:
#                     logger.error("CTGAN model not trained. Cannot generate.")
#                     raise RuntimeError("CTGAN generator model is not trained.")
# 
#                 means = pd.Series(self.generator_model['means'])
#                 stds = pd.Series(self.generator_model['stds'])
#                 min_vals = pd.Series(self.generator_model['min_vals'])
#                 max_vals = pd.Series(self.generator_model['max_vals'])
# 
#                 sequence_data = {}
#                 for col in self.feature_cols:
#                     if col == 'timestamp': continue
#                     if col == 'is_peak':
#                         sequence_data[col] = np.random.choice([0, 1], size=self.seq_len, p=[0.95, 0.05])
#                     else:
#                         generated_col = np.random.normal(means.get(col, 0), stds.get(col, 1), size=self.seq_len)
#                         sequence_data[col] = np.clip(generated_col, min_vals.get(col, -np.inf), max_vals.get(col, np.inf))
# 
#                 sequence = pd.DataFrame(sequence_data)# index=pd.date_range(start=pd.Timestamp.now(), periods=self.seq_len, freq='h'))
#                 # Replace the pd.date_range line with:
#                 # sequence = pd.DataFrame(sequence_data, index=range(_ * self.seq_len, (_ + 1) * self.seq_len))
# 
#                 sequence['is_synthetic'] = True
#                 synthetic_data_list.append(sequence)
# 
#         elif self.generator_model_type == 'Gaussian':
#             logger.info("Generating data from Gaussian distribution...")
#             if not self.generator_model:
#                 logger.error("Gaussian model not trained. Cannot generate.")
#                 raise RuntimeError("Gaussian generator model is not trained.")
# 
#             means = pd.Series(self.generator_model['means'])
#             stds = pd.Series(self.generator_model['stds'])
#             min_vals = pd.Series(self.generator_model['min_vals'])
#             max_vals = pd.Series(self.generator_model['max_vals'])
# 
#             for _ in range(num_sequences):
#                 sequence_data = {}
#                 for col in self.feature_cols:
#                     if col == 'timestamp': continue
#                     if col == 'is_peak':
#                         sequence_data[col] = np.random.choice([0, 1], size=self.seq_len, p=[0.95, 0.05])
#                     else:
#                         generated_col = np.random.normal(means.get(col, 0), stds.get(col, 1), size=self.seq_len)
#                         sequence_data[col] = np.clip(generated_col, min_vals.get(col, -np.inf), max_vals.get(col, np.inf))
#                 
#                 sequence = pd.DataFrame(sequence_data)#, index=pd.date_range(start=pd.Timestamp.now(), periods=self.seq_len, freq='h'))
#                 # Replace the pd.date_range line with:
#                 # sequence = pd.DataFrame(sequence_data, index=range(_ * self.seq_len, (_ + 1) * self.seq_len))
# 
#                 sequence['is_synthetic'] = True
#                 synthetic_data_list.append(sequence)
# 
#         else:
#             logger.error(f"Unsupported generator model type for generation: {self.generator_model_type}")
#             raise ValueError(f"Unsupported generator model type for generation: {self.generator_model_type}")
# 
#         if not synthetic_data_list:
#             logger.warning("No synthetic data generated.")
#             return pd.DataFrame()
# 
#         # for i, df_seq in enumerate(synthetic_data_list):
#         #     unique_suffix = ''.join(random.choices('0123456789abcdef', k=6))
#         #     df_seq['sequence_id'] = f"synth_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}_{i}_{unique_suffix}"
#             
#         generated_df = pd.concat(synthetic_data_list, ignore_index=True)
#         
#         generated_df = generated_df.reset_index(drop=True)
#         
#         if 'is_peak' in generated_df.columns:
#             generated_df['is_peak'] = generated_df['is_peak'].astype(int)
#         
#         logger.info(f"Synthetic data generation completed. Total samples: {len(generated_df)}")
#         return generated_df
# 
