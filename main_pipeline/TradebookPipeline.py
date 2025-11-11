import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import os
import json
import random # Import the random module
from typing import Dict, Any, Tuple, Optional, Union
from loguru import logger

# --- Set up logging for the main pipeline ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "main_pipeline.log", rotation="10 MB", level="INFO")
logger.add(LOG_DIR / "main_pipeline_error.log", rotation="10 MB", level="ERROR")

# --- Updated Imports for Modern Synthetic Data Generation ---
_YDATA_PROFILING_AVAILABLE = False
_SYNTHETIC_DATA_AVAILABLE = False
# _SYNTHETIC_LIBRARY = "SDV"

try:
    # Modern ydata-profiling (formerly pandas-profiling)
    import ydata_profiling
    _YDATA_PROFILING_AVAILABLE = True
    _SYNTHETIC_LIBRARY = "SDV"
    logger.info("ydata-profiling available for data quality analysis.")
except ImportError as e:
    logger.warning(f"ydata-profiling not available: {e}")

try:
    # For synthetic data generation, use modern alternatives
    # Option 1: SDV (Synthetic Data Vault) - most actively maintained
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    #from sdv.timeseries import PARSynthesizer  # For time series   
    from sdv.sequential import PARSynthesizer
    from sdv.metadata import SingleTableMetadata
    
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    # from sdv.single_table.timeseries import LSTMSynthesizer, PARSynthesizer # Ensure these are imported if you plan to use them
    from sdv.sequential import PARSynthesizer

    _SYNTHETIC_DATA_AVAILABLE = True
    _SYNTHETIC_LIBRARY = "SDV"
    logger.info("Using SDV (Synthetic Data Vault) for synthetic data generation.")
except ImportError:
    try:
        # Option 2: synthcity as fallback
        from synthcity.plugins.core.dataloader import GenericDataLoader
        from synthcity.plugins import Plugins
        _SYNTHETIC_DATA_AVAILABLE = True
        _SYNTHETIC_LIBRARY = "SYNTHCITY"
        logger.info("Using synthcity for synthetic data generation.")
    except ImportError:
        try:
            # Option 3: gretel-client for cloud-based generation
            from gretel_client import configure_session
            from gretel_client.projects import create_or_get_project
            from gretel_client.helpers import poll
            _SYNTHETIC_DATA_AVAILABLE = True
            _SYNTHETIC_LIBRARY = "GRETEL"
            logger.info("Using Gretel for synthetic data generation.")
        except ImportError as e:
            logger.error(f"No synthetic data libraries available: {e}")
            logger.info("Install one of: 'pip install sdv', 'pip install synthcity', or 'pip install gretel-client'")

# --- Modern TensorFlow/Keras imports (if needed for custom models) ---
try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel('ERROR')
    logger.info("TensorFlow available for custom neural network models.")
except ImportError:
    logger.warning("TensorFlow not available. Custom neural networks will be disabled.")

# Import core modules (assuming these exist in your project)
# from tradebook_pipeline.config.ConfigLoader import ConfigLoader
# from config.ConfigLoader import ConfigLoader
# from tradebook_pipeline.synthetic_data.smart_integration import SmartSyntheticIntegration
# from tradebook_pipeline.peak_estimators.estimator_factory import PeakEstimatorFactory
# from tradebook_pipeline.peak_estimators.evaluation.metrics import evaluate_peak_detection
# from tradebook_pipeline.system_models.pump_characteristics import PumpCharacteristicParams
from config.ConfigLoader import ConfigLoader
from synthetic_data.smart_integration import SmartSyntheticIntegration
from peak_estimators.estimator_factory import PeakEstimatorFactory
from peak_estimators.evaluation.metrics import evaluate_peak_detection
from system_models.pump_characteristics import PumpCharacteristicParams

# in __main__.py or a similar orchestrator class
import numpy as np
# Assuming ConfigLoader and TradebookPipeline are imported

def convert_numpy_types(obj):
    # ... (paste the function I provided in the previous answer) ...
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj


class ModernSyntheticDataGenerator:
    """Modern synthetic data generator using current libraries."""
    
    def __init__(self, config: dict):
        self.config = config
        
        self.generator_type = config.get('generator_model_type', 'GaussianCopula')
        self.synthesizer = None
        self.metadata = None
        self.stats = {}
        self.categorical_stats = {}
        self.original_columns = []
        
        logger.info("ModernSyntheticDataGenerator initialized with generator type: %s", self.generator_type)

    def train(self, data: pd.DataFrame, time_col: str = None, entity_col: str = None, target_cols: list = None):
        """Train the synthetic data generator."""
        logger.info(f"Training {self.generator_type} synthetic data generator, library {_SYNTHETIC_LIBRARY}...")
        logger.info(f"Args with target_cols={target_cols}, entity_col={entity_col}, datetime_col={time_col}")
                
        if _SYNTHETIC_LIBRARY == "SDV":
            return self._train_sdv(data, time_col, entity_col, target_cols)
        elif _SYNTHETIC_LIBRARY == "SYNTHCITY":
            return self._train_synthcity(data, time_col, entity_col, target_cols)
        elif _SYNTHETIC_LIBRARY == "GRETEL":
            return self._train_gretel(data, time_col, entity_col, target_cols)
        else:
            return self._train_fallback(data, time_col, entity_col, target_cols)
    
    def _train_sdv(self, data: pd.DataFrame, time_col: str, entity_col: str, target_cols: list):
        """Train using SDV library."""

        try:
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(data)

            # Always set datetime type for the time column if it exists
            if time_col and time_col in data.columns:
                # Use update_column instead of set_column_type
                self.metadata.update_column(column_name=time_col, sdtype='datetime')
                logger.info(f"Column '{time_col}' set as datetime.")
                
            # If entity_col is provided by config, and it's 'type_encoded',
            # it should be treated as categorical unless explicitly used as sequence_key
            if entity_col and entity_col in data.columns:
                # If 'entity_col_name' is set in config (e.g., to 'type_encoded')
                # but it's not the primary sequence_key, treat it as categorical.
                # Only set to 'id' if it's truly a unique ID for a relational table.
                # Given our discussions, 'type_encoded' is a feature, so 'categorical' is appropriate here.
                self.metadata.update_column(column_name=entity_col, sdtype='categorical') # Use update_column
                logger.info(f"Column '{entity_col}' set as categorical.")
            

            # If sdv_sequence_key is configured and valid, set it AND its sdtype
            sdv_sequence_key = self.config.get('sdv_sequence_key') # Get directly from generator config
            if sdv_sequence_key and sdv_sequence_key in data.columns:
                self.metadata.set_sequence_key(column_name=sdv_sequence_key)
                # This is crucial for PARSynthesizer: the sequence_key column must be 'id' type
                self.metadata.update_column(column_name=sdv_sequence_key, sdtype='id') # Use update_column
                logger.info(f"SDV Metadata: Set sequence key to '{sdv_sequence_key}' and sdtype='id'.")
            else:
                logger.info("SDV Metadata: No valid sequence key configured or found in data. Assuming single sequence if time-series model chosen.")

            
            # If sdv_sequence_key is configured and valid, set it
            sdv_sequence_key = self.config.get('sdv_sequence_key') # Get directly from generator config
            if sdv_sequence_key and sdv_sequence_key in data.columns:
                self.metadata.set_sequence_key(column_name=sdv_sequence_key)
                # It's also good practice to ensure the sequence key column itself is treated as 'id' or appropriate
                self.metadata.set_column_type(column_name=sdv_sequence_key, sdtype='id') # Ensure it's 'id' as per error
                logger.info(f"SDV Metadata: Set sequence key to '{sdv_sequence_key}'.")
            else:
                logger.info("SDV Metadata: No valid sequence key configured or found in data.")
                
            # Choose synthesizer based on generator type from config
            generator_model_type = self.config.get('model_type', 'gaussiancopula').lower() # Use model_type from config

            if generator_model_type == 'ctgan':
                self.synthesizer = CTGANSynthesizer(
                    self.metadata,
                    epochs=self.config.get('training_params', {}).get('epochs', 300),
                    batch_size=self.config.get('training_params', {}).get('batch_size', 500),
                    verbose=True
                )
                logger.info("SDV: Instantiated CTGANSynthesizer.")
            elif generator_model_type == 'gaussiancopula':
                self.synthesizer = GaussianCopulaSynthesizer(self.metadata)
                logger.info("SDV: Instantiated GaussianCopulaSynthesizer.")
            elif generator_model_type in ['timegan', 'par']: # These models typically expect a sequence_key for PARSynthesizer
                # Check if a sequence key was successfully set.
                if self.metadata.get_sequence_key(): # Check if metadata already has a sequence key
                    self.synthesizer = PARSynthesizer(
                        metadata=self.metadata # metadata should already have sequence_key and datetime set
                    )
                    logger.info("SDV: Instantiated PARSynthesizer with sequence key.")
                elif time_col and not self.metadata.get_sequence_key(): # Single time series
                    # LSTMSynthesizer can be used for single time series without an explicit sequence_key
                    logger.warning(f"Time series model '{generator_model_type}' requested, but no sequence key provided. Falling back to LSTMSynthesizer for single sequence.")
                    # Ensure LSTMSynthesizer is imported if you want to use it
                    from sdv.single_table.timeseries import LSTMSynthesizer
                    self.synthesizer = LSTMSynthesizer(self.metadata)
                    logger.info("SDV: Instantiated LSTMSynthesizer for single sequence.")
                else:
                    logger.warning(f"Time series model '{generator_model_type}' requested, but time_col and/or valid sequence_key missing. Falling back to CTGAN.")
                    self.synthesizer = CTGANSynthesizer(self.metadata)
                    logger.info("SDV: Instantiated CTGANSynthesizer due to missing time_col/sequence_key.")
            else:
                logger.warning(f"Unrecognized generator_model_type '{generator_model_type}'. Defaulting to GaussianCopula.")
                self.synthesizer = GaussianCopulaSynthesizer(self.metadata)
                logger.info("SDV: Instantiated GaussianCopulaSynthesizer (default fallback).")

            # Fit the synthesizer
            self.synthesizer.fit(data)
            logger.info("SDV synthesizer training completed.")
            return True

            # # Choose synthesizer based on generator type from config
            # generator_model_type = self.generator_type.lower() # self.generator_type is set from 'model_type' in config
            # 
            # if generator_model_type == 'ctgan':
            #     self.synthesizer = CTGANSynthesizer(
            #         self.metadata,
            #         epochs=self.config.get('epochs', 300),
            #         batch_size=self.config.get('batch_size', 500),
            #         verbose=True
            #     )
            # elif generator_model_type == 'gaussiancopula':
            #     self.synthesizer = GaussianCopulaSynthesizer(self.metadata)
            # elif generator_model_type in ['timegan', 'par']: # These models typically expect a sequence_key
            #     # If you reach here, it implies you *intend* to use a time-series model.
            #     # If sequence_key was not set above, it will likely fail for PARSynthesizer.
            #     # LSTMSynthesizer can run without an explicit sequence_key for a single sequence.
            #     
            #     # Check if a sequence key was successfully set.
            #     if self.metadata.get_sequence_key(): # Check if metadata already has a sequence key
            #         # If sequence key exists, use a multi-sequence time series model (e.g., PARSynthesizer)
            #         # Note: PARSynthesizer needs sequence_key set in metadata, and optionally context_columns
            #         self.synthesizer = PARSynthesizer(
            #             metadata=self.metadata # metadata should already have sequence_key set
            #             # context_columns=[col for col in data.columns if col not in [time_col, sdv_sequence_key]] # example
            #         )
            #         logger.info("SDV: Instantiated PARSynthesizer with sequence key.")
            #     elif time_col: # Single time series
            #         # For a single time series without an explicit sequence key, LSTMSynthesizer is a better fit
            #         # or you could force a dummy sequence_key in your data and metadata earlier.
            #         logger.warning(f"Time series model '{generator_model_type}' requested, but no sequence key provided. Falling back to LSTMSynthesizer for single sequence.")
            #         from sdv.single_table.timeseries import LSTMSynthesizer
            #         self.synthesizer = LSTMSynthesizer(self.metadata)
            #     else:
            #         logger.warning(f"Time series model '{generator_model_type}' requested, but time_col and sequence_key missing. Falling back to CTGAN.")
            #         self.synthesizer = CTGANSynthesizer(self.metadata)
            # else:
            #     logger.warning(f"Unrecognized generator_model_type '{self.generator_type}'. Defaulting to GaussianCopula.")
            #     self.synthesizer = GaussianCopulaSynthesizer(self.metadata)
            # 
            # # Fit the synthesizer
            # self.synthesizer.fit(data)
            # logger.info("SDV synthesizer training completed.")
            # return True

        except Exception as e:
            logger.error(f"SDV training failed: {e}")
            return False
          
    def _train_synthcity(self, data: pd.DataFrame, time_col: str, entity_col: str, target_cols: list):
        """Train using synthcity library."""
        try:
            # Convert DataFrame to synthcity format
            loader = GenericDataLoader(data)
            
            # Choose plugin based on generator type
            if self.generator_type.lower() == 'ctgan':
                plugin_name = "ctgan"
            elif self.generator_type.lower() == 'timegan':
                plugin_name = "timegan"
            else:
                plugin_name = "marginal_distributions"  # Simple fallback
            
            # Create and train synthesizer
            self.synthesizer = Plugins().get(plugin_name)
            self.synthesizer.fit(loader)
            
            logger.info("Synthcity synthesizer training completed.")
            return True
            
        except Exception as e:
            logger.error(f"Synthcity training failed: {e}")
            return False
    
    def _train_gretel(self, data: pd.DataFrame, time_col: str, entity_col: str, target_cols: list):
        """Train using Gretel cloud service."""
        try:
            # This requires API key setup
            logger.warning("Gretel training requires API key configuration. Implement as needed.")
            return False
        except Exception as e:
            logger.error(f"Gretel training failed: {e}")
            return False
    
    def _train_fallback(self, data: pd.DataFrame, time_col: str, entity_col: str, target_cols: list):
        """Simple statistical fallback generator."""
        logger.info("Using simple statistical fallback generator.")
        
        # Store statistics for generation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.stats = {}
        
        for col in numeric_cols:
            self.stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max()
            }
        
        # Store categorical distributions
        self.categorical_stats = {}
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.categorical_stats[col] = data[col].value_counts(normalize=True).to_dict()
        
        self.original_columns = data.columns.tolist()
        return True
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data."""
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        if _SYNTHETIC_LIBRARY == "SDV" and self.synthesizer:
            # return self.synthesizer.sample(num_sequences=num_samples)
            # Check the type of SDV synthesizer to use the correct argument
            if isinstance(self.synthesizer, PARSynthesizer):
                return self.synthesizer.sample(num_sequences=num_samples)
            else: # GaussianCopulaSynthesizer, CTGANSynthesizer
                return self.synthesizer.sample(num_rows=num_samples)
                
        elif _SYNTHETIC_LIBRARY == "SYNTHCITY" and self.synthesizer:
            synthetic_data = self.synthesizer.generate(count=num_samples)
            return synthetic_data.dataframe()
        else:
            return self._generate_fallback(num_samples)
    
    def _generate_fallback(self, num_samples: int) -> pd.DataFrame:
        """Generate using simple statistical methods."""
        generated_data = {}
        
        # Generate numeric columns
        for col, stats in self.stats.items():
            generated_data[col] = np.random.normal(
                stats['mean'], 
                stats['std'], 
                num_samples
            )
            # Clip to original range
            generated_data[col] = np.clip(
                generated_data[col], 
                stats['min'], 
                stats['max']
            )
        
        # Generate categorical columns
        for col, dist in self.categorical_stats.items():
            values = list(dist.keys())
            probabilities = list(dist.values())
            generated_data[col] = np.random.choice(
                values, 
                size=num_samples, 
                p=probabilities
            )
        
        return pd.DataFrame(generated_data)

class TradebookPipeline:
    """
    Modern implementation of the Unified Peak Detection System.
    Updated to use current libraries and best practices.
    """
    
    def __init__(self):
        logger.info("Initializing modern TradebookPipeline...")
        self.config = ConfigLoader.get_config()
        self.synthetic_data_config = ConfigLoader.get_section("synthetic_data_generation")
        # self.peak_detection_config = ConfigLoader.get_section("peak_estimators")
        self.peak_detection_config = self.config.get('peak_estimators', {}) # Use .get with a default for safety
        self.system_configs = ConfigLoader.get_section("system_characteristics")
        
        # Get data paths
        self.synthetic_data_paths_config = ConfigLoader.get_section('data_paths')
        
        # Initialize components
        self.synthetic_integration = SmartSyntheticIntegration(self.config)
        # self.peak_estimator_factory = PeakEstimatorFactory(self.config)
        
        # Define paths
        self.synthetic_output_dir = Path(self.synthetic_data_paths_config.get('synthetic_output_dir', 'data/synthetic/datasets/'))
        self.raw_data_path = Path(self.synthetic_data_paths_config.get('raw_data', 'data/raw/sample_data.csv'))
        self.predictions_output_dir = Path(self.synthetic_data_paths_config.get('predictions_output_dir', 'data/predictions/'))
        
        # Create directories
        self.synthetic_output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the base root for all models
        base_models_root = Path(self.config.get('paths', {}).get('models_dir', 'models/'))
        # Define the specific directory for peak detection models
        # This will be 'models/peak_detection/'
        self.model_dir = base_models_root / "peak_detection"
        # self.model_dir = Path(self.config['paths']['models_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TradebookPipeline initialized. Peak model_dir for saving: {self.model_dir}")

        # Initialize the factory, passing the common peak_estimators config
        # AND crucially, passing the *same base path* so it can build consistent load paths.
        self.peak_estimator_factory = PeakEstimatorFactory(
            full_config=self.config, # Pass full config as factory expects it
            base_peak_model_path=self.model_dir # Pass the exact path where models are saved
        )

        # Load system characteristics
        self.system_char_instances = self._load_system_characteristics()
        
        # Initialize column names
        self.time_col_name = self.config.get("time_col_name", "timestamp")
        self.entity_col_name = self.config.get("entity_col_name", None)
        self.event_cols = self.config.get("event_cols", [])
        
        # Initialize modern synthetic generator
        self.synthetic_generator = ModernSyntheticDataGenerator(self.synthetic_data_config)
        
        logger.info("Modern TradebookPipeline initialized successfully.")
    
    def _load_system_characteristics(self) -> Dict[str, PumpCharacteristicParams]:
        """Load type characteristic instances from configuration."""
        type_configs = self.system_configs.get('event_types', {})
        type_instances = {}
        for type_id, params in type_configs.items():
            try:
                type_instances[type] = PumpCharacteristicParams.from_dict(params)
                logger.info(f"Loaded system characteristic for type: {type}")
            except Exception as e:
                logger.error(f"Failed to load pump characteristic for {type}: {e}")
        return type_instances
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw pump data from the specified path."""
        logger.info(f"Loading raw data from {self.raw_data_path}...")
        try:
            # Support multiple file formats
            if self.raw_data_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(self.raw_data_path)
            elif self.raw_data_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(self.raw_data_path)
            else:
                df = pd.read_csv(self.raw_data_path)
            
            logger.info(f"Raw data loaded: {df.shape}, {df.columns}, {df.head(2)}")
            # Convert Unix timestamp (milliseconds) to datetime
            # Check if the 'date' column contains these large numbers, otherwise infer.
            if self.time_col_name in df.columns and pd.api.types.is_numeric_dtype(df[self.time_col_name]):
                # Assuming milliseconds, convert to seconds, then to datetime
                df[self.time_col_name] = pd.to_datetime(df[self.time_col_name] / 1000, unit='s')

            # Set the time column as index
            # df = df.set_index(self.time_col_name).sort_index()
            
            # Handle time column
            if self.time_col_name in df.columns:
                df[self.time_col_name] = pd.to_datetime(df[self.time_col_name], errors='coerce')
                df.dropna(subset=[self.time_col_name], inplace=True) ##########################
                df = df.sort_values(by=self.time_col_name)#.set_index(self.time_col_name)
            else:
                logger.warning(f"Time column '{self.time_col_name}' not found.")
            
            logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.error(f"Raw data file not found at {self.raw_data_path}")
            raise
        except Exception as e:
            logger.exception(f"Error loading raw data: {e}")
            raise
    
    def preprocess_data(self, raw_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform modern data preprocessing with enhanced data quality checks.
        """
        logger.info("Starting modern data preprocessing...")
        processed_df = raw_data_df.copy()
        
        # Generate data quality report if ydata-profiling is available
        if _YDATA_PROFILING_AVAILABLE:
            try:
                logger.info("Generating data quality report...")
                profile = ydata_profiling.ProfileReport(
                    processed_df.head(1000),  # Sample for performance
                    title="Data Quality Report",
                    explorative=True,
                    minimal=True
                )
                profile_path = self.synthetic_output_dir / "data_quality_report.html"
                profile.to_file(profile_path)
                logger.info(f"Data quality report saved to {profile_path}")
            except Exception as e:
                logger.warning(f"Failed to generate data quality report: {e}")
        
        # Reset index for processing
        if processed_df.index.name == self.time_col_name:
            processed_df = processed_df.reset_index()
        
        # Handle time column
        if self.time_col_name in processed_df.columns:
            processed_df[self.time_col_name] = pd.to_datetime(processed_df[self.time_col_name], errors='coerce')
            processed_df.dropna(subset=[self.time_col_name], inplace=True)
            processed_df = processed_df.sort_values(by=self.time_col_name)
        
        # Handle target column
        if 'is_peak' not in processed_df.columns:
            processed_df['is_peak'] = 0
        
        # --- START OF MODIFICATION: Introduce random peaks ---
        peak_probability = self.peak_detection_config.get('peak_generation_probability', 0.05) # Get from config, default to 5%
        
        # Get indices where 'is_peak' is currently 0 (to avoid overwriting existing 1s if any)
        # Or, if you want to completely re-generate peaks, just operate on the entire column
        
        # For a simple random assignment:
        # We'll generate random numbers and set 'is_peak' to 1 where the random number is below the probability.
        # This will override any existing 'is_peak' values in the raw data.
        
        num_rows = len(processed_df)
        random_values = np.random.rand(num_rows) # Generate array of random floats between 0.0 and 1.0
        
        # Set 'is_peak' to 1 where random_value is less than peak_probability
        # Ensure it's only set to 1 if it's currently 0, or if you want to completely re-label.
        # For demonstration, we'll re-label the entire column based on probability.
        processed_df['is_peak'] = (random_values < peak_probability).astype(int)
        
        # --- END OF MODIFICATION ---

        processed_df['is_peak'] = processed_df['is_peak'].astype(int) # Ensure it's integer type
        
        logger.info(f"Preprocessing complete. 'is_peak' value counts: \n{processed_df['is_peak'].value_counts()}")
        
        
        # --- Encoding 'type' column to 'type_encoded' (0 or 1 for type_A and type_B) ---
        from sklearn.preprocessing import LabelEncoder # Make sure LabelEncoder is imported
        if 'type' in processed_df.columns:
            logger.info("Processing 'type' column for encoding.")
    
            if processed_df['type'].dtype == 'object' or pd.api.types.is_categorical_dtype(processed_df['type']):
                # Ensure the column is treated as categorical
                processed_df['type'] = processed_df['type'].astype('category')
    
                # Check if there are values other than 'type_A' or 'type_B'
                unique_types = processed_df['type'].unique()
                if not all(t in ['a', 'b'] for t in unique_types):
                    logger.warning(f"Unexpected 'type' values found during encoding: {unique_types}. Expected 'type_a', 'type_b'.")
                    # Decide how to handle unexpected types:
                    # Option 1: Filter them out
                    # processed_df = processed_df[processed_df['type'].isin(['type_A', 'type_B'])]
                    # Option 2: Map them to a default or treat as error
                    # For now, LabelEncoder will assign them a new integer.
    
                le = LabelEncoder()
                try:
                    # Fit and transform the 'type' column
                    processed_df['type_encoded'] = le.fit_transform(processed_df['type'])
                    # Optional: Store the mapping for debugging or future reference
                    logger.debug(f"LabelEncoder classes (mapping): {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
                    # Drop the original 'type' column as 'type_encoded' is created
                    processed_df = processed_df.drop(columns=['type'])
                    logger.info("Successfully encoded 'type' column to 'type_encoded'.")
                    logger.info(f"DataFrame head after 'type_encoded':\n{processed_df[['type_encoded']].head(3)}")
    
                except Exception as e:
                    logger.error(f"Error during Label Encoding 'type' column: {e}", exc_info=True)
                    # Decide how to handle the error: re-raise, return processed_df without encoding, etc.
                    # For robustness, you might want to ensure 'type' is dropped if encoding fails
                    if 'type' in processed_df.columns:
                        processed_df = processed_df.drop(columns=['type'])
                    if 'type_encoded' in processed_df.columns:
                        processed_df = processed_df.drop(columns=['type_encoded'])
    
    
            elif pd.api.types.is_numeric_dtype(processed_df['type']):
                # If 'type' is already numeric (e.g., 0s and 1s directly from source)
                # and named 'type', rename it to 'type_encoded'.
                # This assumes it already contains 0s and 1s representing type_A/B
                processed_df.rename(columns={'type': 'type_encoded'}, inplace=True)
                logger.info("Renamed numeric 'type' column to 'type_encoded'.")
                # logger.info(f"DataFrame head after 'type_encoded' rename:\n{processed_df[['type_encoded']].head(3)}")
                logger.info(f"DataFrame head after 'type_encoded' rename:\n{processed_df.head(3)}")
            else:
                logger.warning("No 'type' column found or it's not a recognized categorical/numeric type for encoding. Skipping 'type' encoding.")
        else:
            logger.warning("No 'type' column found in data for encoding.")
            


        # Modern categorical encoding
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in [self.time_col_name]:  # Don't encode time column
                encoded_col = f"{col}_encoded"
                processed_df[encoded_col] = pd.Categorical(processed_df[col]).codes
                logger.info(f"Encoded '{col}' as '{encoded_col}'")
        
        # Advanced missing value handling
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if processed_df[col].isnull().any():
                # Use median for skewed distributions, mean for normal
                if abs(processed_df[col].skew()) > 1:
                    fill_value = processed_df[col].median()
                    logger.info(f"Filling NaNs in skewed column '{col}' with median: {fill_value:.2f}")
                else:
                    fill_value = processed_df[col].mean()
                    logger.info(f"Filling NaNs in column '{col}' with mean: {fill_value:.2f}")
                processed_df[col].fillna(fill_value, inplace=True)
        
        # Feature engineering
        processed_df = self._add_engineered_features(processed_df)
        
        logger.info(f"Modern preprocessing completed. Shape: {processed_df.shape}")
        return processed_df
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add modern feature engineering."""
        logger.info("Adding engineered features...")
        
        # Time-based features
        if self.time_col_name in df.columns:
            time_col = pd.to_datetime(df[self.time_col_name])
            df['hour'] = time_col.dt.hour
            df['day_of_week'] = time_col.dt.dayofweek
            df['is_weekend'] = (time_col.dt.dayofweek >= 5).astype(int)
            df['month'] = time_col.dt.month
        
        # Rolling window features
        numeric_cols = ['price', 'amount']
        existing_numeric = [col for col in numeric_cols if col in df.columns]
        
        for col in existing_numeric:
            # Rolling statistics
            df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
            df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
            df[f'{col}_rolling_max_5'] = df[col].rolling(window=5, min_periods=1).max()
            df[f'{col}_rolling_min_5'] = df[col].rolling(window=5, min_periods=1).min()
            
            # Percentage change
            df[f'{col}_pct_change'] = df[col].pct_change().fillna(0)
            
            # Z-score (standardized values)
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
        
        return df
    
    def train_synthetic_generator(self, preprocessed_data_df: pd.DataFrame) -> bool:
        """Train the modern synthetic data generator."""
        logger.info("Training modern synthetic data generator...")
        
        if preprocessed_data_df.empty:
            logger.warning("Preprocessed data is empty. Skipping training.")
            return False
        
        # Prepare data for training
        training_data = preprocessed_data_df.copy()
        
        # Reset index to make time column available
        if training_data.index.name == self.time_col_name:
            training_data = training_data.reset_index()
        
        # Train the generator
        success = self.synthetic_generator.train(
            data=training_data,
            time_col=self.time_col_name,
            entity_col=self.entity_col_name,
            target_cols=self.event_cols
        )
        
        if success:
            logger.info("Synthetic data generator training completed successfully.")
        else:
            logger.error("Synthetic data generator training failed.")
        
        return success
    
    def generate_synthetic_data(self, num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic data using the trained generator."""
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        try:
            synthetic_data = self.synthetic_generator.generate(num_samples)
            
            # Save synthetic data
            output_file = self.synthetic_output_dir / self.synthetic_data_config.get(
                'output_filename', 'modern_synthetic_data.parquet'
            )
            synthetic_data.to_parquet(output_file, index=False)
            logger.info(f"Synthetic data saved to {output_file}. Shape: {synthetic_data.shape}")
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            return pd.DataFrame()
    
    def combine_training_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Combine real and synthetic data for training."""
        logger.info("Combining real and synthetic data...")
        
        if synthetic_data.empty:
            logger.warning("No synthetic data to combine. Using real data only.")
            return real_data
        
        # Ensure consistent columns
        real_cols = set(real_data.columns)
        synthetic_cols = set(synthetic_data.columns)
        common_cols = real_cols.intersection(synthetic_cols)
        
        if not common_cols:
            logger.error("No common columns between real and synthetic data.")
            return real_data
        
        # Select common columns and combine
        real_subset = real_data[list(common_cols)].copy()
        synthetic_subset = synthetic_data[list(common_cols)].copy()
        
        # Add source identifier
        real_subset['data_source'] = 'real'
        synthetic_subset['data_source'] = 'synthetic'
        
        combined = pd.concat([real_subset, synthetic_subset], ignore_index=True)
        logger.info(f"Combined data shape: {combined.shape}")
        
        return combined
    
    def prepare_data_for_peak_detection(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for peak detection."""
        logger.info("Preparing data for peak detection...")
        
        # Select feature columns
        feature_cols = []
        
        logger.info(f"Feature columns before engineering lags diff: {list(data.columns)}")
        
        # Basic features
        basic_features = ['price', 'amount']
        feature_cols.extend([col for col in basic_features if col in data.columns])
        
        # Engineered features
        engineered_patterns = ['_rolling_', '_pct_change', '_zscore', '_lag', '_diff']
        for col in data.columns:
            if any(pattern in col for pattern in engineered_patterns):
                feature_cols.append(col)
        
        # Time features
        time_features = ['hour', 'day_of_week', 'is_weekend', 'month']
        feature_cols.extend([col for col in time_features if col in data.columns])
        
        if not feature_cols:
            raise ValueError("No suitable features found for peak detection.")
        
        X = data[feature_cols].fillna(0)
        
        # Prepare labels
        if 'is_peak' not in data.columns:
            raise ValueError("'is_peak' column required for labels.")
        
        y = data['is_peak'].astype(int)
        
        logger.info(f"Features prepared: {X.shape}, Labels: {y.shape}")
        # Inside your prepare_data_for_peak_detection method, after 'is_peak' is generated:
        logger.info(f"Value counts of 'is_peak' after labeling:\n{data['is_peak'].value_counts()}")
        logger.info(f"Feature columns: {list(X.columns)}")
        
        return X, y
    
    def train_peak_estimators(self, training_data: pd.DataFrame):
        """Train peak detection estimators."""
        logger.info("Training peak estimators...")
        
        X_train, y_train = self.prepare_data_for_peak_detection(training_data)
        
        estimators_config = self.peak_detection_config.get('estimators', {})
    
        logger.info(f"train_peak_estimators: self.peak_detection_config at start of method: {self.peak_detection_config}")
        estimators_config = self.peak_detection_config.get('estimators', {})
        logger.info(f"train_peak_estimators: estimators_config after .get('estimators'): {estimators_config}")

        if not estimators_config:
            logger.warning("No estimators found or enabled in peak_detection_config.")
            return

        for est_name, est_params in estimators_config.items():
            if est_params.get('enabled', False):
                logger.info(f"Training estimator: {est_name}")
                try:
                    estimator = self.peak_estimator_factory.create_estimator(est_name)
                    
                    # --- ADD THESE LOGS ---
                    logger.info(f"Unique values in y_train for {est_name}: {y_train.unique()}")
                    logger.info(f"Value counts for y_train for {est_name}:\n{y_train.value_counts()}")
                    # --- END ADDED LOGS ---
                    
                    estimator.train(X_train, y_train)
                    
                    estimator.save_model(str(self.model_dir / est_name))
                    logger.info(f"Estimator '{est_name}' saved to {self.model_dir / est_name}")

                except Exception as e:
                    logger.error(f"Failed to train estimator '{est_name}': {e}")
    
    
    def run_inference(self, new_data: pd.DataFrame, estimator_name: str = "ml_time_series_v1") -> pd.DataFrame:
        """
        Run peak detection inference.
        Includes predicted probabilities in the output DataFrame if the estimator supports it.
        """
        logger.info(f"Running inference with estimator: {estimator_name}")

        try:
            estimator = self.peak_estimator_factory.load_estimator(estimator_name)

            # Add engineered features to new data if not present
            # Ensure self.time_col_name is properly initialized in your __init__
            if 'hour' not in new_data.columns and self.time_col_name in new_data.columns:
                new_data = self._add_engineered_features(new_data)
            
            # Prepare data for peak detection - this likely handles feature selection/transformation
            # X_inference should be the DataFrame with features for the model
            # The second return value (if any) from prepare_data_for_peak_detection is currently ignored (_).
            X_inference, _ = self.prepare_data_for_peak_detection(new_data)
            
            # Get binary predictions (0 or 1)
            predictions = estimator.predict(X_inference)
            new_data['predicted_peak'] = predictions
            
            # --- NEW CODE BLOCK: Get and store predicted probabilities ---
            if hasattr(estimator, 'predict_proba'):
                try:
                    # Call the predict_proba method on the estimator
                    # This should return a pd.Series
                    probabilities = estimator.predict_proba(X_inference)
                    new_data['predicted_proba'] = probabilities
                    logger.info(f"Probabilities also generated and included in output for {estimator_name}.")
                except NotImplementedError as e:
                    logger.warning(f"Estimator '{estimator_name}' explicitly states it does not support 'predict_proba', despite having the method. Error: {e}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred while getting probabilities for {estimator_name}: {e}", exc_info=True)
            else:
                logger.info(f"Estimator '{estimator_name}' does not have a 'predict_proba' method. Probabilities will not be included.")
            # --- END NEW CODE BLOCK ---

            # Save predictions
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            # Ensure self.predictions_output_dir is initialized, e.g., in __init__
            output_path = self.predictions_output_dir / f"predictions_{estimator_name}_{timestamp}.parquet"
            new_data.to_parquet(output_path, index=True) # Ensure index=True to save the datetime index

            logger.info(f"Inference completed. Found {new_data['predicted_peak'].sum()} peaks. Data saved to {output_path}")
            return new_data # This DataFrame will now include 'predicted_proba' if generated

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True) # Added exc_info=True for full traceback
            raise # Re-raise the exception to propagate it

    
    def evaluate_performance(self, predicted_df: pd.DataFrame, estimator_name: str):
        """Evaluate peak detection performance."""
        logger.info(f"Evaluating performance for: {estimator_name}")
        
        required_cols = ['is_peak', 'predicted_peak']
        if not all(col in predicted_df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        true_labels = predicted_df['is_peak']
        predicted_labels = predicted_df['predicted_peak']
        
        # Get predicted_probabilities if available
        predicted_probabilities = None
        if 'predicted_proba' in predicted_df.columns:
            predicted_probabilities = predicted_df['predicted_proba']
            logger.info(f"Found 'predicted_proba' column for {estimator_name}.")
        else:
            logger.info(f"Column 'predicted_proba' not found in predicted_df for {estimator_name}. ROC AUC and PR AUC might be skipped.")

        metrics = evaluate_peak_detection(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            predicted_probabilities=predicted_probabilities, # Pass None if not found
            prefix=f"{estimator_name}_"
        )

        # Convert NumPy types in the metrics dictionary before dumping to JSON
        serializable_metrics = convert_numpy_types(metrics) # <-- Apply conversion here

        logger.info(f"Evaluation metrics:\n{json.dumps(serializable_metrics, indent=2)}")
        
        # Save evaluation report
        report_path = Path(self.peak_detection_config.get('evaluation', {}).get(
            'report_path', 'reports/peak_detection_metrics.json'
        ))
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4) # Use the converted metrics
        logger.info(f"Evaluation report saved to {report_path}")

        return metrics # It's good practice to return the metrics for the caller to use

# Example usage and testing
if __name__ == "__main__":
    # Create sample data and config for testing
    import tempfile
    
    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Using temporary directory: {temp_dir}")
    
    # Create sample data
    sample_data_path = temp_dir / "sample_data.csv"
    num_points = 1000
    
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=num_points, freq='H'),
        'price': np.sin(np.linspace(0, 20, num_points)) * 10 + 50 + np.random.randn(num_points) * 2,
        'amount': np.abs(np.cos(np.linspace(0, 20, num_points))) * 100 + 50 + np.random.randn(num_points) * 10,
        'type': np.random.choice(['pump_A', 'pump_B'], num_points),
        'is_peak': np.zeros(num_points, dtype=int)
    })
    
    # Add some artificial peaks
    peak_indices = np.random.choice(num_points, 20, replace=False)
    sample_data.loc[peak_indices, 'is_peak'] = 1
    sample_data.loc[peak_indices, 'price'] *= 1.5
    
    sample_data.to_csv(sample_data_path, index=False)
    
    # Create sample config
    config_data = {
        'time_col_name': 'date',
        'entity_col_name': 'type_encoded',
        'event_cols': ['price', 'amount'],
        'paths': {
            'models_dir': str(temp_dir / 'models')
        },
        'data_paths': {
            'raw_data': str(sample_data_path),
            'synthetic_output_dir': str(temp_dir / 'synthetic'),
            'predictions_output_dir': str(temp_dir / 'predictions')
        },
        'synthetic_data_generation': {
            'generation_mode': 'local',
            'generator_model_type': 'GaussianCopula',  # Use modern SDV generator
            'output_filename': 'modern_synthetic_data.parquet',
            'training_params': {
                'epochs': 100,
                'batch_size': 128
            },
            'sequence_length': 24,
            'num_synthetic_samples': 500
        },
        'peak_estimators': {
            'estimators': {
                'ml_time_series_v1': {
                    'enabled': True,
                    'type': 'RandomForestClassifier',
                    'params': {
                        'n_estimators': 100,
                        'random_state': 42,
                        'max_depth': 10,
                        'min_samples_split': 5
                    }
                }
            },
            'evaluation': {
                'report_path': str(temp_dir / 'reports' / 'metrics.json')
            }
        },
        'system_characteristics': {
            'type': {
                'type_a': {
                    'nominal_amount_bar': 10.0
                },
                'type_b': {
                    'nominal_amount_bar': 12.0
                }
            }
        }
    }
    
    # Save config to file
    config_path = temp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    
    # Mock ConfigLoader for testing
    class MockConfigLoader:
        _config = config_data
        
        @classmethod
        def get_config(cls):
            return cls._config
        
        @classmethod
        def get_section(cls, section_name):
            return cls._config.get(section_name, {})
    
    # Replace ConfigLoader with mock for testing
    import sys
    sys.modules['config.ConfigLoader'].ConfigLoader = MockConfigLoader
    
    try:
        logger.info("Testing modern pipeline...")
        
        # Initialize pipeline
        pipeline = TradebookPipeline()
        
        # Test data loading
        raw_data = pipeline.load_raw_data()
        logger.info(f"Loaded raw data: {raw_data.shape}")
        
        # Test preprocessing
        processed_data = pipeline.preprocess_data(raw_data)
        logger.info(f"Processed data: {processed_data.shape}")
        
        # Test synthetic data generation
        success = pipeline.train_synthetic_generator(processed_data)
        if success:
            synthetic_data = pipeline.generate_synthetic_data(num_samples=100)
            logger.info(f"Generated synthetic data: {synthetic_data.shape}")
            
            # Test data combination
            combined_data = pipeline.combine_training_data(processed_data, synthetic_data)
            logger.info(f"Combined data: {combined_data.shape}")
        else:
            logger.warning("Using processed data only for training")
            combined_data = processed_data
        
        # Test feature preparation
        X, y = pipeline.prepare_data_for_peak_detection(combined_data)
        logger.info(f"Features: {X.shape}, Labels: {y.shape}")
        logger.info(f"Peak ratio: {y.mean():.3f}")
        
        logger.info("Modern pipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Test cleanup completed")
