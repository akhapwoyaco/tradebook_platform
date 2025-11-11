
import pandas as pd
from loguru import logger
from pathlib import Path
import sys
import os
import json
import warnings
from typing import Optional, Dict, Any
import argparse
from datetime import datetime
import numpy as np # Import numpy for type conversion

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to sys.path to make imports work
# Assuming train_all_models.py is in scripts/, and project root is 2 levels up
project_root = Path(__file__).resolve().parents[1] # Changed to parents[1] if scripts/ is directly in project root
sys.path.insert(0, str(project_root))

# # Import your pipeline components
# try:
#     from tradebook_pipeline.config.ConfigLoader import ConfigLoader
#     from tradebook_pipeline.main_pipeline.TradebookPipeline import TradebookPipeline
#     logger.info("Successfully imported pipeline components")
# except ImportError as e:
#     logger.error(f"Failed to import pipeline components: {e}")
#     logger.info("Make sure your project structure is correct and dependencies are installed.")
#     logger.info("If running from project root, ensure 'pip install -e .' was run.")
#     sys.exit(1)

# Import your pipeline components
try:
    from config.ConfigLoader import ConfigLoader
    from main_pipeline.TradebookPipeline import TradebookPipeline
    logger.info("Successfully imported pipeline components")
except ImportError as e:
    logger.error(f"Failed to import pipeline components: {e}")
    logger.info("Make sure your project structure is correct and dependencies are installed.")
    logger.info("If running from project root, ensure 'pip install -e .' was run.")
    sys.exit(1)


# Configure comprehensive logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove default logger to avoid duplicate logs
logger.remove()

# Add console logger with colors
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# Add file loggers
logger.add(
    LOG_DIR / "train_models.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)

logger.add(
    LOG_DIR / "train_models_error.log",
    rotation="10 MB",
    retention="7 days",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)


# Helper function for JSON serialization (from previous discussion)
def convert_numpy_types(obj):
    """Recursively converts NumPy types to native Python types for JSON serialization."""
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

class ModernTrainingOrchestrator:
    """
    Modern training orchestrator that handles the complete training pipeline
    with proper error handling, logging, and progress tracking.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize the training orchestrator.
        Args:
            config_path (Path): The path to the main configuration file.
        """
        self.config_path = config_path
        self.pipeline: Optional[TradebookPipeline] = None
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'steps_completed': [],
            'errors': [],
            'data_shapes': {},
            'model_metrics': {},
            'overall_status': 'unknown'
        }
        
    def setup_environment(self, custom_raw_data_path: Optional[str] = None) -> bool:
        """
        Setup the training environment, load config, and initialize the pipeline.
        Optionally overrides the raw data path in the config.
        """
        logger.info("Setting up training environment...")
        
        try:
            # Ensure config file exists, create default if not
            if not self.config_path.exists():
                logger.warning(f"Configuration file not found at {self.config_path}. Attempting to create a default one.")
                self._create_default_config()
                logger.info(f"Created default config at {self.config_path}")
                
            # Load configuration. ConfigLoader is a singleton.
            # We call the class directly to ensure its instance is set up with the correct path.
            # ConfigLoader._instance = None # Reset for fresh load, especially important in tests or reruns
            # ConfigLoader(str(self.config_path)) # Initialize the singleton with the config path
            ConfigLoader.load_config(str(self.config_path))
            config = ConfigLoader.get_config()
            
            # --- Apply custom raw data path override ---
            if custom_raw_data_path:
                logger.info(f"Overriding raw_data path in config to: {custom_raw_data_path}")
                config['data_paths']['raw_data'] = custom_raw_data_path
                # Important: Update the internal config of the singleton instance
                ConfigLoader._config_data = config 

            # Validate required sections
            required_sections = ['synthetic_data_generation', 'peak_estimators', 'data_paths', 'time_col_name', 'entity_col_name', 'event_cols']
            missing_sections = [section for section in required_sections 
                                 if section not in config]
            
            if missing_sections:
                logger.error(f"Missing required config sections: {missing_sections}")
                return False
                
            # Initialize pipeline after potentially overriding config
            self.pipeline = TradebookPipeline()
            logger.info("Training environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup training environment: {e}", exc_info=True)
            self.training_stats['errors'].append(f"Environment setup: {str(e)}")
            return False
    
    def _create_default_config(self):
        """
        Create a default configuration file (named config.yaml 
        to match expected config in pipeline logic).
        """
        default_config = {
            "time_col_name": "date",
            # "entity_col_name": "type_encoded",
            "event_cols": ['price', 'amount', "is_peak","type_encoded"],
            "paths": {
                "raw_data_dir": "data/raw/",
                "synthetic_output_dir": "data/synthetic/datasets/",
                "predictions_dir": "data/predictions/",
                "models_dir": "models/",
                "logs_dir": "logs/" # Ensure logs_dir is consistent
            },
            "data_paths": {
                "raw_data": "data/raw/sample_data",
                "synthetic_output_dir": "data/synthetic/datasets/",
                "predictions_output_dir": "data/predictions/"
            },
            "synthetic_data_generation": {
                "generation_mode": "local",
                # "generator_model_type": "timegan",
                "generator_model_type": "ctgan", # CHANGE: Use a more general model like CTGAN or GaussianCopula
                #// or if you need time-series explicitly, consider LSTMSynthesizer
                "output_filename": "synthetic_data.parquet",
                "training_params": {
                    "epochs": 100,
                    "batch_size": 128
                },
                "sequence_length": 24, # This might not be relevant for CTGAN, but harmless
                "num_synthetic_samples": 1000,
                "enabled": True,
                "library": "SDV",
                "model_type": "ctgan", #CHANGE: Align with generator_model_type. This is what your code uses.
                "time_col_name": "date",
                "entity_col_name": None,# // <--- CHANGE: Set to null to prevent SDV treating it as an ID
                "sdv_sequence_key": None# // <--- CHANGE: Set to null as 'type_encoded' is not a sequence key
                
            },
            "peak_estimators": {
                "peak_generation_probability": 0.05, # Add this line! Adjust as needed (e.g., 0.01 for 1% peaks)
                "estimators": {
                    "random_forest_v1": {
                        "enabled": True,
                        "type": "ml_estimator",
                        "model_type": "RandomForestClassifier", # Changed "type" to "model_type" for clarity
                        "params": {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "min_samples_split": 5,
                            "random_state": 42
                        }
                    },
                    "gradient_boosting_v1": {
                        "enabled": True,
                        "type": "ml_estimator",
                        "model_type": "GradientBoostingClassifier", # Changed "type" to "model_type" for clarity
                        "params": {
                            "n_estimators": 100,
                            "learning_rate": 0.1,
                            "max_depth": 6,
                            "random_state": 42
                        }
                    }
                },
                "evaluation": {
                    "report_path": "reports/peak_detection_metrics.json",
                    "cross_validation": {
                        "enabled": True,
                        "cv_folds": 5,
                        "scoring": ["accuracy", "precision", "recall", "f1"]
                    }
                }
            },
            "system_characteristics": {
                "trade_event_characteristics": {
                "event_types": [
                    "type_a",
                    "type_b"
                ],
                "amount_ranges": {
                    "overall_normal_range": [
                    100.0,
                    10000.0
                    ],
                    "type_specific_amount_ranges": {
                    "type_A": [
                        50.0,
                        5000.0
                    ],
                    "type_B": [
                        75.0,
                        7500.0
                    ]
                    }
                },
                "amount_distribution_params": {
                    "type_A": {
                    "mean": 1000.0,
                    "std_dev": 500.0,
                    "skew": 0.5
                    },
                    "type_B": {
                    "mean": 1200.0,
                    "std_dev": 600.0,
                    "skew": 0.6
                    }
                }
                }
            }
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
    
    def validate_data_quality(self, data: pd.DataFrame, stage: str) -> bool:
        """Validate data quality at different stages."""
        logger.info(f"Validating data quality for {stage}...")
        
        try:
            if data.empty:
                logger.error(f"{stage} data is empty")
                return False
                
            # Check for required columns based on stage and pipeline's config
            required_cols = []
            if self.pipeline: # Ensure pipeline is initialized before accessing its attributes
                if stage == "raw":
                    required_cols = [self.pipeline.time_col_name]
                elif stage == "processed":
                    required_cols = [self.pipeline.time_col_name, "is_peak"]
                    # Add relevant event_cols from config that should be present after preprocessing
                    config = ConfigLoader.get_config()
                    if 'event_cols' in config:
                        required_cols.extend([col for col in config['event_cols'] if col not in required_cols])
                elif stage == "features":
                    required_cols = ["is_peak"]
                    config = ConfigLoader.get_config()
                    if 'event_cols' in config:
                        required_cols.extend([col for col in config['event_cols'] if col not in required_cols])
                
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"{stage} data missing required columns: {missing_cols}")
                logger.error(f"Available columns: {data.columns.tolist()}")
                return False
                
            null_percentage = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if null_percentage > 0.5:
                logger.warning(f"{stage} data has high null percentage: {null_percentage:.2%}")
            
            logger.info(f"{stage} data shape: {data.shape}")
            logger.info(f"{stage} data null percentage: {null_percentage:.2%}")
            
            self.training_stats['data_shapes'][stage] = data.shape
            
            return True
            
        except Exception as e:
            logger.error(f"Data quality validation failed for {stage}: {e}", exc_info=True)
            return False
    
    def execute_training_step(self, step_name: str, step_function, *args, **kwargs) -> Any:
        """Execute a training step with error handling and logging."""
        logger.info(f"Executing step: {step_name}")
        
        try:
            result = step_function(*args, **kwargs)
            self.training_stats['steps_completed'].append(step_name)
            logger.info(f"✓ Step completed successfully: {step_name}")
            return result
            
        except Exception as e:
            error_msg = f"✗ Step failed: {step_name} - {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.training_stats['errors'].append(error_msg)
            raise # Re-raise the exception to stop the pipeline on failure
    
    def train_complete_pipeline(self) -> bool: # Removed custom_data_path argument as it's handled in setup_environment
        """Execute the complete training pipeline."""
        logger.info("=" * 60)
        logger.info("🚀 Starting Complete Pipeline Training")
        logger.info("=" * 60)
        
        self.training_stats['start_time'] = datetime.now()
        
        # Ensure pipeline is initialized after config has potentially been overridden
        if self.pipeline is None:
            logger.error("Pipeline not initialized. Call setup_environment first.")
            self.training_stats['overall_status'] = 'failed'
            self.training_stats['errors'].append("Pipeline not initialized before training.")
            self._save_training_summary()
            return False

        try:
            # Step 1: Load and validate raw data
            logger.info("📊 Step 1: Loading raw data...")
            raw_data = self.execute_training_step(
                "load_raw_data",
                self.pipeline.load_raw_data
            )
            
            if not self.validate_data_quality(raw_data, "raw"):
                raise ValueError("Raw data quality check failed.")
            
            # Step 2: Preprocess data
            logger.info("🔧 Step 2: Preprocessing data...")
            processed_data = self.execute_training_step(
                "preprocess_data",
                self.pipeline.preprocess_data,
                raw_data
            )
            
            if not self.validate_data_quality(processed_data, "processed"):
                raise ValueError("Processed data quality check failed.")
            
            # Step 3: Train synthetic data generator
            logger.info("🤖 Step 3: Training synthetic data generator...")
            # train_synthetic_generator can return None if generator_model_type is unsupported
            # or ydata-sdk is missing, but it raises an error.
            # So, if this step completes, generator_success should be True if it returned non-None.
            self.execute_training_step(
                "train_synthetic_generator",
                self.pipeline.train_synthetic_generator,
                processed_data
            )
            generator_trained = self.pipeline.synthetic_generator is not None
            
            # Step 4: Generate synthetic data (if generator training succeeded)
            synthetic_data = pd.DataFrame()
            if generator_trained:
                logger.info("📈 Step 4: Generating synthetic data...")
                synthetic_config = ConfigLoader.get_section("synthetic_data_generation")
                num_samples = synthetic_config.get('num_synthetic_samples', 1000)
                
                synthetic_data = self.execute_training_step(
                    "generate_synthetic_data",
                    self.pipeline.generate_synthetic_data,
                    num_samples
                )
                
                if not synthetic_data.empty:
                    self.validate_data_quality(synthetic_data, "synthetic")
                else:
                    logger.warning("No synthetic data generated, proceeding with real data only.")
            else:
                logger.warning("Synthetic generator was not trained or available. No synthetic data will be generated.")
            
            # Step 5: Combine training data (moved logic here from pipeline)
            logger.info("🔀 Step 5: Combining real and synthetic data...")
            combined_data = pd.DataFrame()
            if not synthetic_data.empty:
                # Align columns and concatenate
                time_col = self.pipeline.time_col_name
                
                # Ensure time_col is a regular column for concatenation, then set as index if desired
                preprocessed_data_for_concat = processed_data.copy()
                if preprocessed_data_for_concat.index.name == time_col:
                    preprocessed_data_for_concat = preprocessed_data_for_concat.reset_index()

                synthetic_data_for_concat = synthetic_data.copy()
                if synthetic_data_for_concat.index.name == time_col:
                    synthetic_data_for_concat = synthetic_data_for_concat.reset_index()

                # Get union of columns to handle potentially different columns
                all_cols = list(set(preprocessed_data_for_concat.columns) | set(synthetic_data_for_concat.columns))
                
                # Reindex and fill NaNs (e.g., with 0 or a sensible default)
                preprocessed_data_for_concat = preprocessed_data_for_concat.reindex(columns=all_cols, fill_value=0)
                synthetic_data_for_concat = synthetic_data_for_concat.reindex(columns=all_cols, fill_value=0)

                combined_data = pd.concat([preprocessed_data_for_concat, synthetic_data_for_concat], ignore_index=True)
                
                # Set time column as index for combined data if it exists and was the original index
                if time_col in combined_data.columns:
                    combined_data = combined_data.set_index(time_col).sort_index()

                logger.info(f"Combined real and synthetic data for training. Shape: {combined_data.shape}")
            else:
                logger.warning("No synthetic data to combine. Training peak estimators on real data only.")
                combined_data = processed_data.copy() # Use only processed real data

            if not self.validate_data_quality(combined_data, "combined"):
                raise ValueError("Combined data quality check failed.")
            
            # Step 6: Train peak detection estimators
            logger.info("🎯 Step 6: Training peak detection estimators...")
            self.execute_training_step(
                "train_peak_estimators",
                self.pipeline.train_peak_estimators,
                combined_data
            )
            
            # Step 7: Model validation (if validation data exists)
            logger.info("✅ Step 7: Model validation...")
            # We use processed_data for validation/inference here as it's the real data
            # You might want a dedicated validation_data_path in config for separate validation set.
            self.execute_training_step(
                "validate_trained_models",
                self._validate_trained_models,
                processed_data # Use real preprocessed data for validation
            )
            
            self.training_stats['end_time'] = datetime.now()
            duration = self.training_stats['end_time'] - self.training_stats['start_time']
            
            logger.info("=" * 60)
            logger.info("🎉 Training Pipeline Completed Successfully!")
            logger.info(f"⏱️ Total Duration: {duration}")
            logger.info(f"📊 Steps Completed: {len(self.training_stats['steps_completed'])}")
            logger.info("=" * 60)
            
            self.training_stats['overall_status'] = 'success'
            self._save_training_summary()
            
            return True
            
        except Exception as e:
            self.training_stats['end_time'] = datetime.now()
            logger.error(f"❌ Training pipeline failed: {e}", exc_info=True)
            self.training_stats['overall_status'] = 'failed'
            self.training_stats['errors'].append(str(e))
            self._save_training_summary()
            return False
    
    def _validate_trained_models(self, validation_data: pd.DataFrame):
        """
        Validate the trained models by running inference and evaluation.
        This is designed as a sub-step callable by execute_training_step.
        """
        logger.info("Validating trained models...")
        
        try:
            estimators_config = ConfigLoader.get_section("peak_estimators").get("estimators", {})
            enabled_estimators = [name for name, config in estimators_config.items()
                                  if config.get("enabled", False)]

            for estimator_name in enabled_estimators:
                logger.info(f"Validating estimator: {estimator_name}")

                try:
                    predictions = self.pipeline.run_inference(
                        validation_data.copy(), 
                        estimator_name
                    )

                    if 'is_peak' in validation_data.columns and not predictions.empty:
                        # Ensure 'is_peak' column exists in predictions for evaluation
                        if 'is_peak' not in predictions.columns:
                            predictions['is_peak'] = validation_data['is_peak'].loc[predictions.index] # Align if indexes match
                        
                        # Evaluate performance and get metrics (assuming evaluate_performance returns metrics)
                        metrics = self.pipeline.evaluate_performance(
                            predictions, 
                            estimator_name
                        )

                        peak_count = predictions['predicted_peak'].sum()
                        total_samples = len(predictions)
                        peak_ratio = peak_count / total_samples if total_samples > 0 else 0

                        # Store model metrics, ensuring JSON-compatible types
                        self.training_stats['model_metrics'][estimator_name] = {
                            'total_samples': int(total_samples),
                            'predicted_peaks': int(peak_count),
                            'peak_ratio': float(peak_ratio),
                            'evaluation_metrics': convert_numpy_types(metrics) # Convert any numpy types in metrics dict
                        }
                        logger.info(f"✓ {estimator_name}: {peak_count} peaks in {total_samples} samples ({peak_ratio:.2%})")

                    else:
                        logger.warning(f"Skipping evaluation for {estimator_name}: 'is_peak' column missing in validation data or predictions are empty.")

                except Exception as e:
                    logger.error(f"Validation failed for {estimator_name}: {e}", exc_info=True)
                    self.training_stats['model_metrics'][estimator_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }

        except Exception as e:
            logger.error(f"Overall model validation process failed: {e}", exc_info=True)
            self.training_stats['overall_status'] = 'failed'
            self.training_stats['overall_error'] = str(e)

        logger.info("Model validation completed.")
    
    def _save_training_summary(self):
        """Save training summary to file."""
        try:
            summary_path = Path("reports/training_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert training_stats to JSON-serializable format
            serializable_stats = convert_numpy_types(self.training_stats)
            
            # Convert datetime objects to strings for JSON serialization
            if serializable_stats['start_time'] and isinstance(serializable_stats['start_time'], datetime):
                serializable_stats['start_time'] = serializable_stats['start_time'].isoformat()
            if serializable_stats['end_time'] and isinstance(serializable_stats['end_time'], datetime):
                serializable_stats['end_time'] = serializable_stats['end_time'].isoformat()
            
            with open(summary_path, 'w') as f:
                json.dump(serializable_stats, f, indent=4) # default=str not needed if convert_numpy_types handles everything
                
            logger.info(f"Training summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}", exc_info=True)

def main():
    """Main function to parse arguments and run the training orchestrator."""
    parser = argparse.ArgumentParser(description="Train Tradebook Pipeline Models")
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("config/config.yaml"), #yaml
        help="Path to configuration file"
    )

    parser.add_argument(
        "--raw_data_path", 
        type=str, 
        default=None,
        help="Optional: Path to a custom raw data CSV file. Overrides the 'raw_data' path in the config file."
    )
    
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate the setup without training."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", colorize=True)
        logger.debug("Verbose logging enabled.")
    
    # Initialize training orchestrator
    orchestrator = ModernTrainingOrchestrator(args.config)
    
    # Setup environment, passing custom_data_path for override
    if not orchestrator.setup_environment(custom_raw_data_path=args.raw_data_path):
        logger.critical("Environment setup failed. Exiting.")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("✅ Validation completed successfully. Exiting without training.")
        return # Exit gracefully after validation
    
    # Execute training
    success = orchestrator.train_complete_pipeline() # No need to pass custom_data_path here
    
    if success:
        logger.info("🎉 Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
