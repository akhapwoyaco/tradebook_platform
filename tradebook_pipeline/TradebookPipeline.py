import sys
import os
import yaml
import warnings
import json
import pickle
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import threading
import time
from functools import wraps
from enum import Enum

import inspect
import dill  # Alternative to pickle that handles more complex objects


# Global variables for directories, will be set from config
LOG_DIR = Path("logs")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
DATA_DIR = Path("data") # New directory for all data files


class MonitoringLevel(Enum):
    """Monitoring levels for different environments"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class IngestionStatus(Enum):
    """Enumeration for ingestion status tracking"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    TIMEOUT = "timeout"

class ProductionMonitor:
    """Production monitoring and alerting system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('monitoring', {})
        self.alerts_enabled = self.config.get('alerts_enabled', True)
        self.retry_config = self.config.get('retry_config', {
            'max_retries': 3,
            'initial_delay': 1,
            'backoff_factor': 2,
            'max_delay': 60
        })
        self.ingestion_metrics = {
            'total_attempts': 0,
            'successful_ingestions': 0,
            'failed_ingestions': 0,
            'retry_attempts': 0,
            'last_success_time': None,
            'last_failure_time': None,
            'consecutive_failures': 0,
            'ingestion_durations': [],
            'failure_reasons': []
        }
        self.pipeline_metrics = {
            'pipeline_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run_time': None,
            'average_run_duration': 0,
            'run_durations': [],
            'step_failures': {}
        }
        
    def log_ingestion_attempt(self, source: str, symbol: str, data_type: str):
        """Log start of ingestion attempt"""
        self.ingestion_metrics['total_attempts'] += 1
        logger.info(f"INGESTION_START: {source} | {symbol} | {data_type} | Attempt #{self.ingestion_metrics['total_attempts']}")
    
    def log_ingestion_success(self, source: str, symbol: str, records_count: int, duration: float):
        """Log successful ingestion"""
        self.ingestion_metrics['successful_ingestions'] += 1
        self.ingestion_metrics['last_success_time'] = datetime.now()
        self.ingestion_metrics['consecutive_failures'] = 0
        self.ingestion_metrics['ingestion_durations'].append(duration)
        
        avg_duration = np.mean(self.ingestion_metrics['ingestion_durations'][-10:])  # Last 10 runs
        logger.info(f"INGESTION_SUCCESS: {source} | {records_count:,} records | {duration:.2f}s | Avg: {avg_duration:.2f}s")
        
        # Alert on performance degradation
        if duration > avg_duration * 2 and len(self.ingestion_metrics['ingestion_durations']) > 5:
            self._send_performance_alert(f"Ingestion duration ({duration:.2f}s) significantly above average ({avg_duration:.2f}s)", "warning")
    
    def log_ingestion_failure(self, source: str, symbol: str, error: Exception, duration: float):
        """Log failed ingestion"""
        self.ingestion_metrics['failed_ingestions'] += 1
        self.ingestion_metrics['last_failure_time'] = datetime.now()
        self.ingestion_metrics['consecutive_failures'] += 1
        self.ingestion_metrics['failure_reasons'].append({
            'timestamp': datetime.now(),
            'source': source,
            'symbol': symbol,
            'error': str(error),
            'error_type': type(error).__name__
        })
        
        logger.error(f"INGESTION_FAILURE: {source} | {symbol} | {error} | Duration: {duration:.2f}s | Consecutive failures: {self.ingestion_metrics['consecutive_failures']}")
        
        # Critical alert on consecutive failures
        if self.ingestion_metrics['consecutive_failures'] >= 3:
            self._send_critical_alert(f"Critical: {self.ingestion_metrics['consecutive_failures']} consecutive ingestion failures")
    
    def log_pipeline_start(self, pipeline_id: str):
        """Log pipeline execution start"""
        self.pipeline_metrics['pipeline_runs'] += 1
        self.pipeline_start_time = datetime.now()
        logger.info(f"PIPELINE_START: {pipeline_id} | Run #{self.pipeline_metrics['pipeline_runs']} | {self.pipeline_start_time}")
    
    def log_pipeline_success(self, pipeline_id: str):
        """Log successful pipeline execution"""
        duration = (datetime.now() - self.pipeline_start_time).total_seconds()
        self.pipeline_metrics['successful_runs'] += 1
        self.pipeline_metrics['last_run_time'] = datetime.now()
        self.pipeline_metrics['run_durations'].append(duration)
        self.pipeline_metrics['average_run_duration'] = np.mean(self.pipeline_metrics['run_durations'])
        
        logger.info(f"PIPELINE_SUCCESS: {pipeline_id} | Duration: {duration:.2f}s | Success rate: {self._get_pipeline_success_rate():.1%}")
    
    def log_pipeline_failure(self, pipeline_id: str, step: str, error: Exception):
        """Log failed pipeline execution"""
        duration = (datetime.now() - self.pipeline_start_time).total_seconds()
        self.pipeline_metrics['failed_runs'] += 1
        
        if step not in self.pipeline_metrics['step_failures']:
            self.pipeline_metrics['step_failures'][step] = 0
        self.pipeline_metrics['step_failures'][step] += 1
        
        logger.error(f"PIPELINE_FAILURE: {pipeline_id} | Step: {step} | Error: {error} | Duration: {duration:.2f}s")
        self._send_critical_alert(f"Pipeline failure in step '{step}': {error}")
    
    def log_step_start(self, step_name: str, additional_info: str = ""):
        """Log pipeline step start"""
        logger.info(f"STEP_START: {step_name} | {additional_info}")
        return time.time()  # Return start time for duration calculation
    
    def log_step_success(self, step_name: str, start_time: float, additional_info: str = ""):
        """Log pipeline step success"""
        duration = time.time() - start_time
        logger.info(f"STEP_SUCCESS: {step_name} | Duration: {duration:.2f}s | {additional_info}")
    
    def log_step_warning(self, step_name: str, warning_msg: str):
        """Log pipeline step warning"""
        logger.warning(f"STEP_WARNING: {step_name} | {warning_msg}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            'ingestion_metrics': {
                **self.ingestion_metrics,
                'success_rate': self._get_ingestion_success_rate(),
                'avg_duration': np.mean(self.ingestion_metrics['ingestion_durations']) if self.ingestion_metrics['ingestion_durations'] else 0
            },
            'pipeline_metrics': {
                **self.pipeline_metrics,
                'success_rate': self._get_pipeline_success_rate()
            },
            'system_health': self._assess_system_health()
        }
    
    def _get_ingestion_success_rate(self) -> float:
        """Calculate ingestion success rate"""
        total = self.ingestion_metrics['successful_ingestions'] + self.ingestion_metrics['failed_ingestions']
        return self.ingestion_metrics['successful_ingestions'] / total if total > 0 else 0
    
    def _get_pipeline_success_rate(self) -> float:
        """Calculate pipeline success rate"""
        total = self.pipeline_metrics['successful_runs'] + self.pipeline_metrics['failed_runs']
        return self.pipeline_metrics['successful_runs'] / total if total > 0 else 0
    
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        ingestion_rate = self._get_ingestion_success_rate()
        pipeline_rate = self._get_pipeline_success_rate()
        consecutive_failures = self.ingestion_metrics['consecutive_failures']
        
        if consecutive_failures >= 5:
            return "CRITICAL"
        elif ingestion_rate < 0.8 or pipeline_rate < 0.8:
            return "WARNING"
        elif ingestion_rate >= 0.95 and pipeline_rate >= 0.95:
            return "HEALTHY"
        else:
            return "DEGRADED"
    
    def _send_performance_alert(self, message: str, level: str = "info"):
        """Send performance alert (placeholder for actual alerting system)"""
        if self.alerts_enabled:
            logger.warning(f"PERFORMANCE_ALERT [{level.upper()}]: {message}")
    
    def _send_critical_alert(self, message: str):
        """Send critical alert (placeholder for actual alerting system)"""
        if self.alerts_enabled:
            logger.critical(f"CRITICAL_ALERT: {message}")


def retry_with_monitoring(monitor_getter: Callable, operation_name: str):
    """Decorator for retry logic with monitoring"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get monitor instance
            if hasattr(args[0], 'monitor'):
                monitor = args[0].monitor
            else:
                return func(*args, **kwargs)  # No monitoring available
                
            retry_config = monitor.retry_config
            max_retries = retry_config['max_retries']
            initial_delay = retry_config['initial_delay']
            backoff_factor = retry_config['backoff_factor']
            max_delay = retry_config['max_delay']
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        monitor.ingestion_metrics['retry_attempts'] += 1
                        logger.info(f"RETRY_ATTEMPT: {operation_name} | Attempt {attempt}/{max_retries}")
                    
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if attempt > 0:
                        logger.info(f"RETRY_SUCCESS: {operation_name} | Succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                        logger.warning(f"RETRY_FAILED: {operation_name} | Attempt {attempt + 1}/{max_retries + 1} failed: {e} | Retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"RETRY_EXHAUSTED: {operation_name} | All {max_retries + 1} attempts failed")
            
            # All retries exhausted
            raise last_exception
        return wrapper
    return decorator


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file with enhanced error handling.
    
    Args:
        config_path (str): The path to the configuration YAML file.
    
    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
    """
    try:
        config_path = Path(config_path).resolve()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        
        # Validate required configuration sections
        required_sections = ['data_ingestion', 'data_processing', 'core_analysis']
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            logger.warning(f"Missing configuration sections: {missing_sections}")
        
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise

def setup_directories(config: Dict[str, Any]):
    """
    Sets up and creates necessary directories based on the configuration.
    """
    global LOG_DIR, MODEL_DIR, REPORT_DIR, DATA_DIR
    
    paths = config.get("paths", {})
    LOG_DIR = Path(paths.get("logs_dir", "logs"))
    MODEL_DIR = Path(paths.get("models_dir", "models"))
    REPORT_DIR = Path(paths.get("predictions_dir", "reports"))
    DATA_DIR = Path(paths.get("data_dir", "data"))

    # Create directories with error handling
    directories = [LOG_DIR, MODEL_DIR, REPORT_DIR, DATA_DIR]
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory created/verified: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise

    # Create subdirectories
    subdirectories = [
        (DATA_DIR / "processed"),
        (DATA_DIR / "synthetic"),
        (LOG_DIR / "monitoring"),
        (REPORT_DIR / "monitoring")
    ]
    
    for subdir in subdirectories:
        try:
            subdir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create subdirectory {subdir}: {e}")

    # Setup enhanced logging with monitoring
    logger.add(LOG_DIR / "main_pipeline.log", rotation="10 MB", level="INFO")
    logger.add(LOG_DIR / "main_pipeline_error.log", rotation="10 MB", level="ERROR")
    logger.add(LOG_DIR / "monitoring" / "production_monitoring.log", rotation="5 MB", level="DEBUG")
    logger.info("Pipeline directories and enhanced logging configured.")


warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from tradebook_pipeline.data_ingestion.ingestion_manager import IngestionManager
    from tradebook_pipeline.data_processing.data_processor import DataProcessor
    from tradebook_pipeline.core_analysis.tools_manager import ToolsManager
    from tradebook_pipeline.synthetic_data.synthesizer import SyntheticDataGenerator
    from tradebook_pipeline.simulations.backtester import EnhancedBacktester
    from tradebook_pipeline.live_trading.live_trader import LiveTrader
    
    from tradebook_pipeline.model_retraining.perfomance_tracker import PerformanceTracker
    from tradebook_pipeline.model_retraining.retraining_manager import RetrainingManager
    from tradebook_pipeline.model_retraining.model_updater import ModelUpdater
    from tradebook_pipeline.model_retraining.evaluation_dashboard import EvaluationDashboard
    
    

except ModuleNotFoundError as e:
    logger.error(f"Failed to import a module: {e}")
    logger.info("This is likely due to the script not being run as a module. "
                "Please run using: `python -m tradebook_pipeline.TradebookPipeline`")
    sys.exit(1)


class TradebookPipeline:
    """
    The main class for the Tradebook Analysis Pipeline with production monitoring.

    This class orchestrates the entire workflow, from data ingestion and
    processing to model training, synthetic data generation, and
    strategy backtesting, with comprehensive monitoring and error handling.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the pipeline with a given configuration.

        Args:
            config (Dict[str, Any]): The pipeline's configuration, typically
                                     loaded from a YAML file.
        """
        self.config = config
        self.monitor = ProductionMonitor(config)
        self.pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize modules as None - will be created during initialization
        self.ingestion_manager = None
        self.data_processor = None
        self.tools_manager = None
        self.synthesizer = None
        self.backtester = None
        self.live_trader = None
        
        
        # Continuous learning components
        self.performance_tracker = None
        self.retraining_manager = None
        self.model_updater = None
        self.dashboard = None
        
        # Initialize if enabled
        if self.config.get('model_retraining', {}).get('enabled', False):
            self._initialize_retraining_system()
            
            
            
    def initialize_modules(self):
        """
        Initializes all pipeline modules based on the configuration with monitoring.
        """
        step_start_time = self.monitor.log_step_start("module_initialization")
        
        try:
            logger.info("Initializing pipeline modules...")
            
            # Initialize ingestion manager with monitoring wrapper
            self.ingestion_manager = self._initialize_ingestion_manager()
            
            # Initialize data processor
            processing_config = self.config.get('data_processing', {})
            self.data_processor = DataProcessor(config=processing_config)
            logger.info("Data processor initialized")
            
            # Initialize tools manager
            core_analysis_config = self.config.get('core_analysis', {})
            self.tools_manager = ToolsManager(config=core_analysis_config)
            logger.info("Tools manager initialized")
            
            # Initialize synthetic data generator if enabled
            synthetic_data_config = self.config.get('synthetic_data', {})
            if synthetic_data_config.get('enabled', False):
                self.synthesizer = SyntheticDataGenerator(config=synthetic_data_config)
                logger.info("Synthetic data generation module enabled.")
            else:
                logger.info("Synthetic data generation module disabled.")
            
            self.monitor.log_step_success("module_initialization", step_start_time, "All modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            self.monitor.log_pipeline_failure(self.pipeline_id, "module_initialization", e)
            raise
    
    def _initialize_ingestion_manager(self):
        """Initialize ingestion manager with monitoring wrapper"""
        try:
            ingestion_manager = IngestionManager(config=self.config)
            
            # Wrap ingestion methods with monitoring
            original_fetch_single_token = ingestion_manager.fetch_single_token
            
            def monitored_fetch_single_token(symbol, start_time, end_time, data_type):
                self.monitor.log_ingestion_attempt("exchange", symbol, data_type)
                start = time.time()
                
                try:
                    result = original_fetch_single_token(symbol, start_time, end_time, data_type)
                    duration = time.time() - start
                    
                    if result is not None and not result.empty:
                        self.monitor.log_ingestion_success("exchange", symbol, len(result), duration)
                    else:
                        self.monitor.log_ingestion_failure("exchange", symbol, Exception("No data returned"), duration)
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start
                    self.monitor.log_ingestion_failure("exchange", symbol, e, duration)
                    raise
            
            # Replace the original method with monitored version
            ingestion_manager.fetch_single_token = monitored_fetch_single_token
            
            logger.info("Ingestion manager initialized with monitoring wrapper")
            return ingestion_manager
            
        except Exception as e:
            logger.error(f"Failed to initialize ingestion manager: {e}")
            raise
    
    def _save_models(self, models: list):
        """
        Saves the trained models to the models directory with enhanced logging and diagnostics.
        
        Args:
            models (list): A list of dictionaries, where each dictionary
                           contains a 'model' key with the trained model object.
        """
        step_start_time = self.monitor.log_step_start("model_saving", f"Saving {len(models)} models")
        
        try:
            logger.info(f"Starting to save {len(models)} models...")
            
            saved_count = 0
            failed_count = 0
            
            for i, model_dict in enumerate(models):
                model_name = model_dict.get('name', f'unnamed_model_{i}')
                model_to_save = model_dict.get('model')
                
                logger.info(f"Processing model {i+1}/{len(models)}: '{model_name}'")
                
                if not model_to_save:
                    logger.warning(f"Model '{model_name}' is None or empty - skipping")
                    failed_count += 1
                    continue
                    
                # Enhanced diagnostics
                logger.info(f"Model type: {type(model_to_save)}")
                logger.info(f"Model dict keys: {list(model_dict.keys())}")
                
                # Analyze model structure
                self._analyze_model_structure(model_name, model_to_save)
                
                clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                file_path = MODEL_DIR / f"{clean_name}.pkl"
                
                # Try multiple serialization approaches
                saved_successfully = False
                
                # Approach 1: Standard pickle
                if not saved_successfully:
                    saved_successfully = self._try_pickle_save(model_name, model_to_save, file_path)
                
                # Approach 2: Dill (more powerful serialization)
                if not saved_successfully:
                    saved_successfully = self._try_dill_save(model_name, model_to_save, file_path)
                
                # Approach 3: Custom serialization for specific model types
                if not saved_successfully:
                    saved_successfully = self._try_custom_save(model_name, model_to_save, clean_name)
                
                # Fallback: Save metadata only
                if not saved_successfully:
                    self._save_metadata_only(model_dict, clean_name)
                    self.monitor.log_step_warning("model_saving", f"Only metadata saved for model '{model_name}'")
                    failed_count += 1
                else:
                    saved_count += 1
            
            self.monitor.log_step_success("model_saving", step_start_time, 
                                        f"Saved: {saved_count}, Failed: {failed_count}")
            
        except Exception as e:
            logger.error(f"Critical error in model saving: {e}")
            self.monitor.log_pipeline_failure(self.pipeline_id, "model_saving", e)
            raise
    
    def _analyze_model_structure(self, model_name: str, model):
        """Analyze the structure of the model to identify potential serialization issues."""
        logger.info(f"=== Analyzing model structure for '{model_name}' ===")
        
        try:
            # Check if it's a dictionary (common for custom models)
            if isinstance(model, dict):
                logger.info(f"Model is a dictionary with keys: {list(model.keys())}")
                for key, value in model.items():
                    logger.info(f"  {key}: {type(value)}")
                    if hasattr(value, '__call__') and not inspect.isbuiltin(value):
                        if 'lambda' in str(value):
                            logger.warning(f"  Found lambda function in key '{key}': {value}")
                        elif inspect.isfunction(value):
                            logger.info(f"  Found function in key '{key}': {value.__name__}")
            
            # Check attributes for objects
            elif hasattr(model, '__dict__'):
                attrs = vars(model)
                logger.info(f"Model has {len(attrs)} attributes: {list(attrs.keys())}")
                for attr_name, attr_value in attrs.items():
                    logger.info(f"  {attr_name}: {type(attr_value)}")
                    if hasattr(attr_value, '__call__') and not inspect.isbuiltin(attr_value):
                        if 'lambda' in str(attr_value):
                            logger.warning(f"  Found lambda function in attribute '{attr_name}': {attr_value}")
                        elif inspect.isfunction(attr_value):
                            logger.info(f"  Found function in attribute '{attr_name}': {attr_value.__name__}")
            
            # Check for common problematic attributes
            problematic_attrs = ['lambda', 'partial', 'functools']
            for attr_name in dir(model):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(model, attr_name)
                        attr_str = str(attr_value)
                        if any(prob in attr_str for prob in problematic_attrs):
                            logger.warning(f"  Potentially problematic attribute '{attr_name}': {attr_str}")
                    except Exception as e:
                        logger.debug(f"  Could not analyze attribute '{attr_name}': {e}")
                        
        except Exception as e:
            logger.error(f"Error analyzing model structure: {e}")
    
    def _try_pickle_save(self, model_name: str, model, file_path: Path) -> bool:
        """Attempt to save using standard pickle."""
        logger.info(f"Attempting standard pickle save for '{model_name}'...")
        
        try:
            # Test serialization first
            logger.debug("Testing pickle serialization...")
            serialized_data = pickle.dumps(model)
            logger.info(f"Pickle test successful. Serialized size: {len(serialized_data)} bytes")
            
            # Save to file
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"✓ Successfully saved model '{model_name}' using pickle to {file_path}")
            return True
            
        except Exception as e:
            logger.warning(f"✗ Pickle save failed for '{model_name}': {e}")
            logger.debug(f"Pickle error details: {type(e).__name__}: {e}")
            return False
    
    def _try_dill_save(self, model_name: str, model, file_path: Path) -> bool:
        """Attempt to save using dill (more powerful than pickle)."""
        logger.info(f"Attempting dill save for '{model_name}'...")
        
        try:
            import dill
            
            # Test serialization first
            logger.debug("Testing dill serialization...")
            serialized_data = dill.dumps(model)
            logger.info(f"Dill test successful. Serialized size: {len(serialized_data)} bytes")
            
            # Save to file with dill extension
            dill_path = file_path.with_suffix('.dill')
            with open(dill_path, 'wb') as f:
                dill.dump(model, f)
            
            logger.info(f"✓ Successfully saved model '{model_name}' using dill to {dill_path}")
            return True
            
        except ImportError:
            logger.warning("Dill not available. Install with: pip install dill")
            return False
        except Exception as e:
            logger.warning(f"✗ Dill save failed for '{model_name}': {e}")
            logger.debug(f"Dill error details: {type(e).__name__}: {e}")
            return False
    
    def _try_custom_save(self, model_name: str, model, clean_name: str) -> bool:
        """Attempt custom serialization based on model type."""
        logger.info(f"Attempting custom save for '{model_name}'...")
        
        try:
            # For dictionary-based models, try to save components separately
            if isinstance(model, dict):
                return self._save_dict_model(model_name, model, clean_name)
            
            # For sklearn-like models
            elif hasattr(model, 'get_params') and hasattr(model, 'set_params'):
                return self._save_sklearn_like_model(model_name, model, clean_name)
            
            # For other object types, try to extract serializable components
            else:
                return self._save_object_components(model_name, model, clean_name)
                
        except Exception as e:
            logger.warning(f"✗ Custom save failed for '{model_name}': {e}")
            return False
    
    def _save_dict_model(self, model_name: str, model_dict: dict, clean_name: str) -> bool:
        """Save dictionary-based models by separating serializable and non-serializable parts."""
        logger.info(f"Saving dictionary model '{model_name}' with component separation...")
        
        serializable_parts = {}
        non_serializable_parts = {}
        
        for key, value in model_dict.items():
            try:
                pickle.dumps(value)  # Test if this component can be pickled
                serializable_parts[key] = value
                logger.debug(f"  Component '{key}' is serializable")
            except Exception as e:
                non_serializable_parts[key] = {
                    'type': str(type(value)),
                    'string_repr': str(value)[:200] + ('...' if len(str(value)) > 200 else ''),
                    'error': str(e)
                }
                logger.warning(f"  Component '{key}' is not serializable: {e}")
        
        # Save serializable parts
        if serializable_parts:
            serializable_path = MODEL_DIR / f"{clean_name}_serializable.pkl"
            with open(serializable_path, 'wb') as f:
                pickle.dump(serializable_parts, f)
            logger.info(f"Saved serializable components to {serializable_path}")
        
        # Save information about non-serializable parts
        if non_serializable_parts:
            non_serializable_path = MODEL_DIR / f"{clean_name}_non_serializable.json"
            with open(non_serializable_path, 'w') as f:
                json.dump(non_serializable_parts, f, indent=2)
            logger.info(f"Saved non-serializable component info to {non_serializable_path}")
        
        return True
    
    def _save_sklearn_like_model(self, model_name: str, model, clean_name: str) -> bool:
        """Save sklearn-like models using their parameter extraction."""
        logger.info(f"Saving sklearn-like model '{model_name}'...")
        
        try:
            # Extract parameters
            params = model.get_params()
            
            # Try to save the model state
            model_state = {
                'params': params,
                'type': type(model).__name__,
                'module': type(model).__module__,
            }
            
            # Try to extract fitted attributes
            if hasattr(model, '__dict__'):
                fitted_attrs = {k: v for k, v in vars(model).items() 
                              if k.endswith('_') and not k.startswith('_')}
                if fitted_attrs:
                    model_state['fitted_attributes'] = fitted_attrs
            
            state_path = MODEL_DIR / f"{clean_name}_state.json"
            with open(state_path, 'w') as f:
                json.dump(model_state, f, indent=2, default=str)
            
            logger.info(f"Saved model state to {state_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save sklearn-like model state: {e}")
            return False
    
    def _save_object_components(self, model_name: str, model, clean_name: str) -> bool:
        """Save object by extracting its components."""
        logger.info(f"Saving object components for '{model_name}'...")
        
        try:
            components = {}
            
            # Extract basic information
            components['type'] = str(type(model))
            components['module'] = getattr(type(model), '__module__', 'unknown')
            components['name'] = getattr(type(model), '__name__', 'unknown')
            
            # Extract attributes if possible
            if hasattr(model, '__dict__'):
                for attr_name, attr_value in vars(model).items():
                    try:
                        pickle.dumps(attr_value)
                        components[f'attr_{attr_name}'] = attr_value
                    except:
                        components[f'attr_{attr_name}_info'] = {
                            'type': str(type(attr_value)),
                            'repr': str(attr_value)[:100]
                        }
            
            components_path = MODEL_DIR / f"{clean_name}_components.json"
            with open(components_path, 'w') as f:
                json.dump(components, f, indent=2, default=str)
            
            logger.info(f"Saved object components to {components_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save object components: {e}")
            return False
    
    def _save_metadata_only(self, model_dict: dict, clean_name: str):
        """Save comprehensive metadata when model cannot be serialized."""
        logger.info(f"Saving metadata only for '{model_dict.get('name', 'unnamed')}'...")
        
        try:
            metadata = {
                'name': model_dict.get('name', 'unnamed'),
                'strategy': model_dict.get('strategy', 'unknown'),
                'parameters': model_dict.get('parameters', {}),
                'type': str(type(model_dict.get('model', {}))),
                'note': 'Model contains non-serializable components',
                'available_keys': list(model_dict.keys()),
                'serialization_attempts': {
                    'pickle': 'failed',
                    'dill': 'failed',
                    'custom': 'failed'
                },
                'timestamp': str(pd.Timestamp.now()),
            }
            
            # Try to extract more information about the model
            model = model_dict.get('model')
            if model:
                if isinstance(model, dict):
                    metadata['model_keys'] = list(model.keys())
                    metadata['model_key_types'] = {k: str(type(v)) for k, v in model.items()}
                elif hasattr(model, '__dict__'):
                    metadata['model_attributes'] = list(vars(model).keys())
            
            metadata_path = MODEL_DIR / f"{clean_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Saved comprehensive metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            
    @retry_with_monitoring(lambda self: self.monitor, "data_save")        
    def _save_performance_report(self, results: Dict[str, Any], report_name: str):
        """
        Saves the performance report as a JSON file with retry logic.
        
        Args:
            results (Dict[str, Any]): The results dictionary from the evaluation.
            report_name (str): The name for the report file.
        """
        try:
            file_path = REPORT_DIR / f"{report_name}.json"
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Successfully saved performance report to {file_path}")
            
            # Save monitoring report as well
            monitoring_report_path = REPORT_DIR / "monitoring" / f"{report_name}_monitoring.json"
            monitoring_summary = self.monitor.get_monitoring_summary()
            with open(monitoring_report_path, 'w') as f:
                json.dump(monitoring_summary, f, indent=2, default=str)
            logger.info(f"Monitoring summary saved to {monitoring_report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")
            raise

    @retry_with_monitoring(lambda self: self.monitor, "dataframe_save")
    def _save_dataframe(self, df: pd.DataFrame, file_path: Path):
        """
        Saves a pandas DataFrame to a Parquet file with retry logic.
        
        Args:
            df (pd.DataFrame): The DataFrame to save.
            file_path (Path): The full path to the output file, including filename.
        """
        try:
            # Validate DataFrame before saving
            if df is None or df.empty:
                raise ValueError(f"Cannot save empty DataFrame to {file_path}")
                
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, file_path)
            logger.info(f"Successfully saved DataFrame ({df.shape[0]:,} rows, {df.shape[1]} cols) to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {file_path}: {e}")
            raise

    @retry_with_monitoring(lambda self: self.monitor, "csv_save")
    def _save_dataframe_to_csv(self, df: pd.DataFrame, file_path: Path):
        """
        Saves a pandas DataFrame to a CSV file with retry logic.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            file_path (Path): The full path to the output file, including filename.
        """
        try:
            # Validate DataFrame before saving
            if df is None or df.empty:
                raise ValueError(f"Cannot save empty DataFrame to {file_path}")
                
            df.to_csv(file_path, index=False)
            logger.info(f"Successfully saved DataFrame ({df.shape[0]:,} rows, {df.shape[1]} cols) to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {file_path}: {e}")
            raise

    def evaluate_model_performance(self, labeled_data: pd.DataFrame, trained_models: list) -> Dict[str, Any]:
        """
        Evaluates the performance of each trained model with enhanced monitoring.

        Calculates key classification metrics for each model and returns a
        dictionary of results.

        Args:
            labeled_data (pd.DataFrame): The DataFrame with the ground-truth
                                         'is_peak' labels.
            trained_models (list): A list of dictionaries, each representing
                                   a trained model.

        Returns:
            Dict[str, Any]: A dictionary containing performance metrics for
                            each model.
        """
        step_start_time = self.monitor.log_step_start("model_evaluation", f"Evaluating {len(trained_models)} models")
        
        try:
            logger.info("Evaluating model performance...")
            evaluation_results = {}
            true_labels = labeled_data['is_peak'].values
            
            successful_evaluations = 0
            failed_evaluations = 0

            for model_dict in trained_models:
                model_name = model_dict.get('name', 'Unnamed Model')
                try:
                    # Get the predictions from the model
                    predictions = self.tools_manager.apply_peak_labels(labeled_data.copy(), model_dict)['is_peak'].values
                    
                    # Calculate metrics
                    accuracy = accuracy_score(true_labels, predictions)
                    precision = precision_score(true_labels, predictions, zero_division=0)
                    recall = recall_score(true_labels, predictions, zero_division=0)
                    
                    # Get the confusion matrix as a list
                    cm = confusion_matrix(true_labels, predictions).tolist()
                    
                    results = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "confusion_matrix": {
                            "labels": ["Not Peak", "Peak"],
                            "matrix": cm
                        },
                        "evaluation_timestamp": datetime.now().isoformat(),
                        "data_size": len(labeled_data)
                    }
                    
                    evaluation_results[model_name] = results
                    successful_evaluations += 1
                    
                    logger.info(f"Performance for '{model_name}':")
                    logger.info(f"  Accuracy: {accuracy:.4f}")
                    logger.info(f"  Precision: {precision:.4f}")
                    logger.info(f"  Recall: {recall:.4f}")
                    
                    # Alert on poor performance
                    if accuracy < 0.6:
                        self.monitor.log_step_warning("model_evaluation", 
                                                    f"Low accuracy ({accuracy:.3f}) for model '{model_name}'")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate model '{model_name}': {e}")
                    evaluation_results[model_name] = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "evaluation_timestamp": datetime.now().isoformat()
                    }
                    failed_evaluations += 1

            # Add summary to results
            evaluation_results["evaluation_summary"] = {
                "total_models": len(trained_models),
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            self.monitor.log_step_success("model_evaluation", step_start_time, 
                                        f"Success: {successful_evaluations}, Failed: {failed_evaluations}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Critical error in model evaluation: {e}")
            self.monitor.log_pipeline_failure(self.pipeline_id, "model_evaluation", e)
            raise
    
    def _initialize_retraining_system(self):
        """Initialize continuous learning components"""
        try:
            logger.info("Initializing continuous learning system...")
            
            # Performance tracker
            self.performance_tracker = PerformanceTracker(
                config=self.config,
                output_dir="./monitoring/performance"
            )
            
            # Model updater (needs to be initialized before retraining manager)
            self.model_updater = ModelUpdater(
                config=self.config,
                models_dir=str(MODEL_DIR)
            )
            
            # Dashboard
            if self.config.get('model_retraining', {}).get('dashboard', {}).get('enabled', True):
                self.dashboard = EvaluationDashboard(
                    performance_tracker=self.performance_tracker,
                    retraining_manager=None,  # Will be set after retraining_manager init
                    model_updater=self.model_updater,
                    output_dir="./monitoring/dashboards"
                )
            
            logger.info("✓ Continuous learning system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize retraining system: {e}")
            self.performance_tracker = None
            self.retraining_manager = None
            self.model_updater = None
            self.dashboard = None
            
            
    def run_pipeline(self):
        """
        Executes the full end-to-end data pipeline workflow with comprehensive monitoring.
        This method defines the sequential logic of the pipeline.
        """
        try:
            self.monitor.log_pipeline_start(self.pipeline_id)
            logger.info(f"Starting the Tradebook Analysis Pipeline (ID: {self.pipeline_id})...")

            # Step 1: Data Ingestion with enhanced monitoring
            step_start_time = self.monitor.log_step_start("data_ingestion")
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=0.01730)
            data_type = 'tradebook'
            sym_BOL = 'BTC/USDT'
            
            logger.info(f"Ingestion parameters:")
            logger.info(f"  Time range: {start_time} to {end_time} ({(end_time - start_time).days} days)")
            logger.info(f"  Symbol: {sym_BOL}")
            logger.info(f"  Data type: {data_type}")
            
            # Fetch data with monitoring
            raw_data = self._fetch_data_with_monitoring(sym_BOL, start_time, end_time, data_type)
            
            if raw_data is None or raw_data.empty:
                error_msg = "No data was ingested. Pipeline cannot continue."
                logger.error(error_msg)
                self._log_ingestion_debug_info()
                self.monitor.log_pipeline_failure(self.pipeline_id, "data_ingestion", Exception(error_msg))
                return
            
            self.monitor.log_step_success("data_ingestion", step_start_time, 
                                        f"Ingested {raw_data.shape[0]:,} records")
            
            # Log comprehensive data analysis
            self._log_comprehensive_data_analysis(raw_data, start_time, end_time)
            
            # Step 2: Data Processing
            step_start_time = self.monitor.log_step_start("data_processing")
            
            processed_data = self._process_data_with_monitoring(raw_data)
            
            if processed_data is None or processed_data.empty:
                error_msg = "Data processing failed. Pipeline cannot continue."
                logger.error(error_msg)
                self.monitor.log_pipeline_failure(self.pipeline_id, "data_processing", Exception(error_msg))
                return
            
            # Save processed data
            processed_parquet_path = DATA_DIR / "processed" / "processed_data.parquet"
            processed_csv_path = DATA_DIR / "processed" / "processed_data.csv"
            self._save_dataframe(processed_data, processed_parquet_path)
            self._save_dataframe_to_csv(processed_data, processed_csv_path)
            
            self.monitor.log_step_success("data_processing", step_start_time, 
                                        f"Processed data shape: {processed_data.shape}")

            # Step 3: Model Training
            step_start_time = self.monitor.log_step_start("model_training")
            
            trained_models = self._train_models_with_monitoring(processed_data)
            
            if trained_models:
                self.monitor.log_step_success("model_training", step_start_time, 
                                            f"Trained {len(trained_models)} models")
                
                # Apply labels and save data
                model_to_label = trained_models[0]
                labeled_data = self.tools_manager.apply_peak_labels(processed_data.copy(), model_to_label)
                logger.info("Successfully labeled data with the 'is_peak' response variable.")

                labeled_parquet_path = DATA_DIR / "processed" / "labeled_data.parquet"
                labeled_csv_path = DATA_DIR / "processed" / "labeled_data.csv"
                self._save_dataframe(labeled_data, labeled_parquet_path)
                self._save_dataframe_to_csv(labeled_data, labeled_csv_path)
                
                # Step 4: Model Evaluation
                logger.info("Step 4: Evaluating model performance...")
                performance_results = self.evaluate_model_performance(labeled_data, trained_models)
                self._save_performance_report(performance_results, "model_performance_report")
                
                # Save models
                self._save_models(trained_models)
                
            else:
                self.monitor.log_step_warning("model_training", "No models were trained")
                labeled_data = processed_data

           
            # Step 5: Synthetic Data Generation (if enabled) - ENHANCED VERSION
            if self.synthesizer:
                step_start_time = self.monitor.log_step_start("synthetic_data_generation")
                try:
                    logger.info("="*60)
                    logger.info("STEP 5: SYNTHETIC DATA GENERATION")
                    logger.info("="*60)
                    
                    # Log input data characteristics
                    logger.info(f"Input data for training:")
                    logger.info(f"  Shape: {labeled_data.shape}")
                    logger.info(f"  Columns: {list(labeled_data.columns)}")
                    logger.info(f"  Dtypes: {labeled_data.dtypes.to_dict()}")
                    
                    # Train generator with enhanced validation
                    logger.info("Training synthetic data generator...")
                    self.synthesizer.train_generator(labeled_data)
                    
                    # Log training summary
                    training_summary = self.synthesizer.get_generation_summary()
                    logger.info("Training summary:")
                    logger.info(f"  Model type: {training_summary['model_type']}")
                    logger.info(f"  Features: {training_summary['feature_columns']}")
                    logger.info(f"  Sequence length: {training_summary['sequence_length']}")
                    logger.info(f"  Training samples: {training_summary['training_metadata'].get('n_samples', 'N/A')}")
                    
                    # Generate synthetic data with validation
                    num_sequences_to_generate = self.config.get('synthetic_data', {}).get('num_sequences', 1000)
                    logger.info(f"Generating {num_sequences_to_generate} synthetic sequences...")
                    
                    synthetic_data = self.synthesizer.generate_data(
                        num_sequences=num_sequences_to_generate,
                        raw_data_base=labeled_data,
                        validate=True  # Enable validation
                    )
                    
                    # Log generation results
                    logger.info(f"Synthetic data generated:")
                    logger.info(f"  Shape: {synthetic_data.shape}")
                    logger.info(f"  Memory usage: {synthetic_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    # Compare distributions
                    if not synthetic_data.empty:
                        logger.info("Distribution comparison (Real vs Synthetic):")
                        numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
                        
                        for col in numeric_cols[:5]:  # Show first 5 numeric columns
                            if col in labeled_data.columns and col != 'is_synthetic':
                                real_mean = labeled_data[col].mean()
                                synth_mean = synthetic_data[col].mean()
                                real_std = labeled_data[col].std()
                                synth_std = synthetic_data[col].std()
                                
                                logger.info(f"  {col}:")
                                logger.info(f"    Real:  mean={real_mean:.4f}, std={real_std:.4f}")
                                logger.info(f"    Synth: mean={synth_mean:.4f}, std={synth_std:.4f}")
                                logger.info(f"    Diff:  {abs(synth_mean-real_mean)/abs(real_mean)*100:.2f}% mean, "
                                          f"{abs(synth_std-real_std)/abs(real_std)*100:.2f}% std")
                    
                    # Save synthetic data
                    synthetic_parquet_path = DATA_DIR / "synthetic" / "synthetic_data.parquet"
                    synthetic_csv_path = DATA_DIR / "synthetic" / "synthetic_data.csv"
                    
                    self._save_dataframe(synthetic_data, synthetic_parquet_path)
                    self._save_dataframe_to_csv(synthetic_data, synthetic_csv_path)
                    
                    # Save generation metadata
                    metadata_path = DATA_DIR / "synthetic" / "generation_metadata.json"
                    generation_metadata = {
                        'generator_type': training_summary['model_type'],
                        'num_sequences': num_sequences_to_generate,
                        'total_samples': len(synthetic_data),
                        'feature_columns': training_summary['feature_columns'],
                        'real_data_samples': len(labeled_data),
                        'generation_timestamp': datetime.now().isoformat(),
                        'training_summary': training_summary
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(generation_metadata, f, indent=2, default=str)
                    
                    logger.info(f"Generation metadata saved to {metadata_path}")
                    
                    self.monitor.log_step_success(
                        "synthetic_data_generation", 
                        step_start_time,
                        f"Generated {len(synthetic_data)} synthetic records from {len(labeled_data)} real samples"
                    )
                    
                    logger.info("="*60)
                    logger.info("SYNTHETIC DATA GENERATION COMPLETED SUCCESSFULLY")
                    logger.info("="*60)
                    
                except KeyError as e:
                    logger.error(f"Column missing error in synthetic data generation: {e}")
                    logger.error(f"Available columns: {list(labeled_data.columns)}")
                    logger.error(f"Required columns: {self.config.get('synthetic_data', {}).get('feature_cols', 'Not specified')}")
                    self.monitor.log_pipeline_failure(self.pipeline_id, "synthetic_data_generation", e)
                    
                    # Save debug information
                    debug_info = {
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'available_columns': list(labeled_data.columns),
                        'data_shape': labeled_data.shape,
                        'data_dtypes': labeled_data.dtypes.astype(str).to_dict(),
                        'config_feature_cols': self.config.get('synthetic_data', {}).get('feature_cols', []),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    debug_path = DATA_DIR / "synthetic" / "generation_error_debug.json"
                    with open(debug_path, 'w') as f:
                        json.dump(debug_info, f, indent=2)
                    
                    logger.info(f"Debug information saved to {debug_path}")
                    
                except Exception as e:
                    logger.error(f"Synthetic data generation failed: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Error details:", exc_info=True)
                    self.monitor.log_pipeline_failure(self.pipeline_id, "synthetic_data_generation", e)
            


            # Step 6: Backtesting (if enabled)
            simulations_config = self.config.get('simulations', {})
            if simulations_config.get('enabled', False) and trained_models:
                self._run_backtesting_with_monitoring(trained_models, labeled_data, simulations_config)

            # Step 7: Live Trading (if enabled)
            live_trading_config = self.config.get('live_trading', {})
            if live_trading_config.get('enabled', False) and trained_models:
                self._run_live_trading_with_monitoring(trained_models, live_trading_config, sym_BOL, data_type)

            # Pipeline completed successfully
            self.monitor.log_pipeline_success(self.pipeline_id)
            logger.info(f"Pipeline execution completed successfully (ID: {self.pipeline_id})")
            
            # Save final monitoring report
            final_monitoring_report = self.monitor.get_monitoring_summary()
            final_report_path = REPORT_DIR / "monitoring" / f"final_monitoring_report_{self.pipeline_id}.json"
            with open(final_report_path, 'w') as f:
                json.dump(final_monitoring_report, f, indent=2, default=str)
            logger.info(f"Final monitoring report saved to {final_report_path}")

        except Exception as e:
            logger.critical(f"Critical pipeline failure: {e}")
            self.monitor.log_pipeline_failure(self.pipeline_id, "pipeline_execution", e)
            
            # Save failure report
            failure_report = {
                'pipeline_id': self.pipeline_id,
                'failure_time': datetime.now().isoformat(),
                'error_message': str(e),
                'error_type': type(e).__name__,
                'monitoring_summary': self.monitor.get_monitoring_summary()
            }
            
            failure_report_path = REPORT_DIR / "monitoring" / f"pipeline_failure_{self.pipeline_id}.json"
            with open(failure_report_path, 'w') as f:
                json.dump(failure_report, f, indent=2, default=str)
            
            logger.info(f"Failure report saved to {failure_report_path}")
            raise

    def _fetch_data_with_monitoring(self, symbol: str, start_time: datetime, end_time: datetime, data_type: str):
        """Fetch data with comprehensive monitoring"""
        try:
            raw_data = self.ingestion_manager.fetch_single_token(symbol, start_time, end_time, data_type)
            return raw_data
        except Exception as e:
            logger.error(f"Data fetching failed: {e}")
            raise

    def _process_data_with_monitoring(self, raw_data: pd.DataFrame):
        """Process data with monitoring"""
        try:
            logger.info("Step 2: Processing data and generating features...")
            processed_data = self.data_processor.process_data(raw_data)
            return processed_data
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise

    def _train_models_with_monitoring(self, processed_data: pd.DataFrame):
        """Train models with monitoring"""
        try:
            logger.info("Step 3: Training peak detection models...")
            trained_models = self.tools_manager.detect_peaks(processed_data)
            logger.info(f"Successfully trained {len(trained_models)} peak detection models.")
            return trained_models
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def _log_ingestion_debug_info(self):
        """Log debug information for ingestion failures"""
        logger.info("Debug information:")
        logger.info(f"Config keys: {list(self.config.keys())}")
        data_ingestion = self.config.get('data_ingestion', {})
        logger.info(f"Data ingestion config: {data_ingestion}")
        logger.info(f"Tokens to monitor: {data_ingestion.get('tokens_to_monitor', [])}")
        logger.info(f"Data sources: {data_ingestion.get('data_sources', {})}")

    def _log_comprehensive_data_analysis(self, raw_data: pd.DataFrame, start_time: datetime, end_time: datetime):
        """Log comprehensive analysis of ingested data"""
        logger.info(f"Successfully ingested data with {raw_data.shape[0]} records.")
        logger.info("="*50)
        logger.info("DATA ANALYSIS SUMMARY:")
        logger.info("="*50)
        
        # Dimensions and shape
        logger.info(f"Data dimensions: {raw_data.shape} (rows x columns)")
        logger.info(f"Memory usage: {raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Index information
        logger.info(f"Index type: {type(raw_data.index).__name__}")
        logger.info(f"Index range: {raw_data.index.min()} to {raw_data.index.max()}")
        logger.info(f"Index length: {len(raw_data.index)}")
        
        # Column information
        logger.info(f"Columns ({len(raw_data.columns)}): {list(raw_data.columns)}")
        logger.info(f"Column data types:")
        for col, dtype in raw_data.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # Time-based analysis (if timestamp columns exist)
        timestamp_cols = [col for col in raw_data.columns if 'time' in col.lower()]
        if timestamp_cols:
            for col in timestamp_cols:
                if raw_data[col].notna().any():
                    logger.info(f"Timestamp column '{col}':")
                    logger.info(f"  Min: {raw_data[col].min()}")
                    logger.info(f"  Max: {raw_data[col].max()}")
                    logger.info(f"  Span: {raw_data[col].max() - raw_data[col].min()}")
                    logger.info(f"  Non-null count: {raw_data[col].notna().sum()}")
        
        # Data quality checks
        null_counts = raw_data.isnull().sum()
        if null_counts.any():
            logger.info("Null value counts:")
            for col, null_count in null_counts[null_counts > 0].items():
                logger.info(f"  {col}: {null_count} ({null_count/len(raw_data)*100:.2f}%)")
        else:
            logger.info("No null values found in dataset")
        
        # Data transformations
        self._log_data_transformations(raw_data)
        
        # Time range verification
        self._verify_time_range(raw_data, start_time, end_time)
        
        logger.info("="*50)
        logger.info("Data ingestion and preprocessing completed successfully!")
        logger.info("="*50)

    def _log_data_transformations(self, raw_data: pd.DataFrame):
        """Log data transformation steps"""
        logger.info("="*50)
        logger.info("PERFORMING DATA TRANSFORMATIONS:")
        logger.info("="*50)
        
        # Side mapping transformation
        if 'side' in raw_data.columns:
            logger.info("Mapping 'side' column values...")
            side_counts_before = raw_data['side'].value_counts().to_dict()
            logger.info(f"  Before mapping: {side_counts_before}")
            
            raw_data['side'] = raw_data['side'].map({'ask': 'a', 'bid': 'b'})
            
            side_counts_after = raw_data['side'].value_counts().to_dict()
            logger.info(f"  After mapping: {side_counts_after}")
            
            # Check for unmapped values
            unmapped = raw_data['side'].isnull().sum()
            if unmapped > 0:
                logger.warning(f"  WARNING: {unmapped} values could not be mapped!")
        
        # Column renaming
        if 'side' in raw_data.columns:
            logger.info("Renaming column: 'side' -> 'type'")
            raw_data = raw_data.rename(columns={'side': 'type'})
            logger.info(f"  New columns: {list(raw_data.columns)}")
        
        # Column dropping
        if 'timestamp_ms' in raw_data.columns:
            logger.info("Dropping 'timestamp_ms' column...")
            logger.info(f"  Before drop - Shape: {raw_data.shape}")
            raw_data.drop('timestamp_ms', axis=1, inplace=True)
            logger.info(f"  After drop - Shape: {raw_data.shape}")
            logger.info(f"  Remaining columns: {list(raw_data.columns)}")

    def _verify_time_range(self, raw_data: pd.DataFrame, start_time: datetime, end_time: datetime):
        """Verify the time range of the data"""
        if any('time' in col.lower() for col in raw_data.columns):
            time_col = next((col for col in raw_data.columns if 'time' in col.lower()), None)
            if time_col and raw_data[time_col].notna().any():
                actual_start = raw_data[time_col].min()
                actual_end = raw_data[time_col].max()
                logger.info("Time range verification:")
                logger.info(f"  Requested: {start_time} to {end_time}")
                logger.info(f"  Actual: {actual_start} to {actual_end}")
                
                # Check if data spans the full requested range
                if hasattr(actual_start, 'to_pydatetime'):
                    actual_start = actual_start.to_pydatetime()
                if hasattr(actual_end, 'to_pydatetime'):
                    actual_end = actual_end.to_pydatetime()
                    
                coverage_start = actual_start >= start_time
                coverage_end = actual_end <= end_time
                logger.info(f"  Coverage check - Start: {'✓' if coverage_start else '✗'}, End: {'✓' if coverage_end else '✗'}")

    def _run_backtesting_with_monitoring(self, trained_models: list, labeled_data: pd.DataFrame, simulations_config: dict):
        """Run backtesting with monitoring"""
        step_start_time = self.monitor.log_step_start("backtesting")
        
        try:
            logger.info("Step 6: Running backtesting simulation...")
            model_to_backtest = None
            
            for model_dict in trained_models:
                model_name = model_dict.get('name', 'Unnamed Model')
                if 'model' in model_dict and isinstance(model_dict['model'], dict) and 'predict' in model_dict['model'] and callable(model_dict['model']['predict']):
                    model_to_backtest = model_dict['model']
                    logger.info(f"Using model for backtesting: {model_name}")
                    break
                else:
                    logger.warning(f"Skipping backtesting for model '{model_name}': No valid 'predict' method found.")
            
            if model_to_backtest and not isinstance(model_to_backtest, str):
                # Enhanced backtester configuration mapping
                enhanced_config = {
                    'initial_capital': simulations_config.get('initial_capital', 100000),
                    'position_size': simulations_config.get('position_size', 0.95),
                    'transaction_fee_rate': simulations_config.get('transaction_fee_rate', 0.001),
                    'min_trade_interval': simulations_config.get('min_trade_interval', 1),
                    'save_detailed_results': simulations_config.get('save_detailed_results', True),
                    'output_directory': simulations_config.get('output_directory', './backtest_outputs')
                }
                
                self.backtester = EnhancedBacktester(config=enhanced_config, model=model_to_backtest)
                backtest_results = self.backtester.run_backtest(labeled_data)
                
                # Extract metrics for logging
                metrics = backtest_results['performance_metrics']
                logger.info("Enhanced backtesting complete.")
                logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
                logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
                logger.info(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
                
                self._save_performance_report(backtest_results, "backtest_report")
                self.monitor.log_step_success("backtesting", step_start_time, 
                                            f"Return: {metrics['total_return_pct']:.2f}%")
                
            else:
                logger.warning("No valid model found for backtesting. Skipping simulation.")
                self.monitor.log_step_warning("backtesting", "No valid model found")
                
        except Exception as e:
            logger.error(f"Backtesting simulation failed: {e}")
            self.monitor.log_pipeline_failure(self.pipeline_id, "backtesting", e)
            raise

    def _run_live_trading_with_monitoring(self, trained_models: list, live_trading_config: dict, sym_BOL: str, data_type: str):
        """Run live trading with monitoring"""
        step_start_time = self.monitor.log_step_start("live_trading")
        
        try:
            logger.info("Step 7: Starting live trading simulation...")
            
            logger.info("Available models for live trading:")
            for i, model_dict in enumerate(trained_models):
                logger.info(f"  Model {i}: {model_dict.get('name', 'Unnamed')}")
                logger.info(f"    Keys: {list(model_dict.keys())}")
                if 'model' in model_dict:
                    logger.info(f"    Model type: {type(model_dict['model'])}")
                    if isinstance(model_dict['model'], dict):
                        logger.info(f"    Model dict keys: {list(model_dict['model'].keys())}")
            
            model_to_use = None
            for model_dict in trained_models:
                model_name = model_dict.get('name', 'Unnamed Model')
                
                # Check for Rule-Based Estimator (first priority)
                if 'Rule-Based Estimator' in model_name:
                    if 'model' in model_dict and isinstance(model_dict['model'], dict) and 'predict' in model_dict['model'] and callable(model_dict['model']['predict']):
                        model_to_use = model_dict['model']
                        logger.info(f"Found and validated Rule-Based model for live trading: {model_name}")
                        break
                    else:
                        logger.warning(f"Rule-Based model found but lacks valid 'predict' method. Skipping.")
                        continue
                
                # Check for ML Estimator (fallback)
                elif 'ML Estimator' in model_name:
                    if 'model' in model_dict and isinstance(model_dict['model'], dict) and 'object' in model_dict['model']:
                        ml_model = model_dict['model']['object']
                        if hasattr(ml_model, 'predict') and callable(ml_model.predict):
                            # Create a wrapper for the ML model to match expected interface
                            model_to_use = {
                                'predict': ml_model.predict,
                                'type': 'ml_model',
                                'scaler': model_dict['model'].get('scaler'),
                                'features': model_dict['model'].get('features')
                            }
                            logger.info(f"Found and validated ML model for live trading: {model_name}")
                            break
                    else:
                        logger.warning(f"ML model found but lacks valid sklearn object. Skipping.")
                        continue
            else:
                logger.warning("Could not find a suitable model for live trading. Skipping.")
                self.monitor.log_step_warning("live_trading", "No suitable model found")
                return

            symbol_to_trade = live_trading_config.get('symbol', sym_BOL)
            if model_to_use:
                # Initialize LiveTrader with enhanced configuration
                enhanced_config = {
                    **live_trading_config,
                    'save_trades': True,
                    'save_performance': True,
                    'performance_update_interval': 300,  # 5 minutes
                    'output_directory': './trading_outputs'
                }
                
                self.live_trader = LiveTrader(
                    config=enhanced_config, 
                    model=model_to_use,
                    ingestion_manager=self.ingestion_manager
                )
                
                
                
                
                
                # Connect performance tracker to live trader
                if self.performance_tracker and hasattr(self.live_trader, 'performance_tracker'):
                    self.live_trader.performance_tracker = self.performance_tracker
                
                # Initialize retraining manager now that tools_manager is available
                if self.performance_tracker and not self.retraining_manager:
                    self.retraining_manager = RetrainingManager(
                        config=self.config,
                        performance_tracker=self.performance_tracker,
                        tools_manager=self.tools_manager,
                        data_processor=self.data_processor
                    )
                    
                    # Update dashboard with retraining manager
                    if self.dashboard:
                        self.dashboard.retraining_manager = self.retraining_manager
                    
                    logger.info("✓ Retraining manager initialized")
                
                # Start dashboard generation thread
                if self.dashboard:
                    dashboard_thread = threading.Thread(
                        target=self._run_dashboard_updates,
                        daemon=True
                    )
                    dashboard_thread.start()
                    logger.info("Dashboard updates started")
                
                # Start continuous learning monitoring thread
                if self.retraining_manager and self.performance_tracker:
                    retraining_thread = threading.Thread(
                        target=self._run_continuous_learning,
                        args=(self.live_trader, sym_BOL, data_type),
                        daemon=True
                    )
                    retraining_thread.start()
                    logger.info("Continuous learning monitoring started")
                    
                
                
                
                    
                    
                # Start trading with monitoring
                self.live_trader.start_trading()
                
                # Set up monitoring thread
                monitoring_thread = threading.Thread(
                    target=self._monitor_live_trading, 
                    args=(self.live_trader,),
                    daemon=True
                )
                monitoring_thread.start()
                
                # Run trading with exception handling
                try:
                    self.live_trader.run_trading_loop(symbol=sym_BOL, data_type=data_type)
                except KeyboardInterrupt:
                    logger.info("Trading interrupted by user")
                except Exception as e:
                    logger.error(f"Trading loop error: {e}")
                    self.monitor.log_pipeline_failure(self.pipeline_id, "live_trading_loop", e)
                finally:
                    # Always stop trading gracefully
                    self.live_trader.stop_trading()
                    logger.info("Live trading stopped and session report generated")
                
                self.monitor.log_step_success("live_trading", step_start_time, "Live trading completed")
                logger.info("Live trading simulation completed. Check logs and output files for results.")
            
        except Exception as e:
            logger.error(f"Live trading simulation failed to start: {e}")
            self.monitor.log_pipeline_failure(self.pipeline_id, "live_trading", e)
            raise
    
    def _run_dashboard_updates(self):
        """Continuously update the evaluation dashboard"""
        update_interval = self.config.get('model_retraining', {}).get('dashboard', {}).get('auto_refresh_seconds', 30)
        
        while self.live_trader and self.live_trader.trading_active:
            try:
                self.dashboard.generate_dashboard(save_html=True)
                time.sleep(update_interval)
            except Exception as e:
                logger.error(f"Dashboard update failed: {e}")
                time.sleep(update_interval)
    
    def _run_continuous_learning(self, live_trader, symbol: str, data_type: str):
        """Monitor performance and trigger retraining when needed"""
        check_interval = 60  # Check every minute
        
        while live_trader.trading_active:
            try:
                time.sleep(check_interval)
                
                # Check if retraining needed
                should_retrain, reasons = self.retraining_manager.should_retrain()
                
                if should_retrain:
                    logger.info(f"🔄 Initiating automated retraining: {', '.join(reasons)}")
                    
                    # Pause trading temporarily (optional - depends on strategy)
                    # live_trader.trading_active = False
                    
                    # Fetch recent trading data for retraining
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=7)  # Last 7 days
                    
                    recent_data = self.ingestion_manager.fetch_single_token(
                        symbol, start_time, end_time, data_type
                    )
                    
                    # Retrain models
                    new_models = self.retraining_manager.retrain_models(recent_data)
                    
                    if new_models:
                        # Save new model versions
                        current_metrics = self.performance_tracker.get_current_metrics()
                        perf_metrics = {
                            'prediction_accuracy': current_metrics.prediction_accuracy if current_metrics else 0.0,
                            'win_rate': current_metrics.win_rate if current_metrics else 0.0
                        }
                        
                        version_ids = self.model_updater.save_new_model_version(
                            models=new_models,
                            training_samples=len(recent_data) if recent_data is not None else 0,
                            performance_metrics=perf_metrics
                        )
                        
                        # Deploy new models
                        for version_id in version_ids:
                            self.model_updater.deploy_model(version_id)
                        
                        logger.info(f"✓ Retraining complete. {len(version_ids)} new versions deployed")
                        
                        # Update live trader's model if in shadow mode
                        if self.model_updater.deployment_strategy == 'shadow_mode':
                            logger.info("New models deployed in shadow mode for validation")
                    else:
                        logger.warning("Retraining produced no models")
                    
                    # Resume trading
                    # live_trader.trading_active = True
                    
            except Exception as e:
                logger.error(f"Continuous learning error: {e}", exc_info=True)
                time.sleep(check_interval)
    
    def _save_models(self, models: list):
        """
        Enhanced model saving with versioning support.
        """
        step_start_time = self.monitor.log_step_start("model_saving", f"Saving {len(models)} models")
        
        try:
            # Use model updater if available
            if self.model_updater:
                current_metrics = self.performance_tracker.get_current_metrics() if self.performance_tracker else None
                perf_metrics = {
                    'prediction_accuracy': current_metrics.prediction_accuracy if current_metrics else 0.0,
                    'win_rate': current_metrics.win_rate if current_metrics else 0.0
                } if current_metrics else {}
                
                version_ids = self.model_updater.save_new_model_version(
                    models=models,
                    training_samples=0,  # Will be updated during retraining
                    performance_metrics=perf_metrics
                )
                
                logger.info(f"Saved {len(version_ids)} model versions with version control")
            else:
                # Fallback to original saving logic
                # ... (keep your existing _save_models code as fallback) ...
                pass
                
            self.monitor.log_step_success("model_saving", step_start_time, 
                                         f"Saved {len(models)} models")
                                         
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            self.monitor.log_pipeline_failure(self.pipeline_id, "model_saving", e)
            raise
          
    
    
          
    def _monitor_live_trading(self, live_trader):
        """Monitor live trading and log key metrics periodically with enhanced monitoring"""
        monitoring_interval = 60  # Log status every minute
        
        while live_trader.trading_active:
            try:
                status = live_trader.get_current_status()
                metrics_summary = live_trader.get_live_metrics_summary()
                
                # Log condensed status
                logger.info(f"TRADING STATUS - Portfolio: ${status['current_portfolio_value']:,.2f}, "
                           f"Return: {metrics_summary['total_return_pct']:+.2f}%, "
                           f"Trades: {status['total_trades']}, "
                           f"Errors: {status['consecutive_errors']}")
                
                # Retraining system status (with safety checks)
                if hasattr(self, 'retraining_manager') and self.retraining_manager:
                    try:
                        retrain_stats = self.retraining_manager.get_retraining_stats()
                        logger.info(f"RETRAINING - Total: {retrain_stats.get('total_retrains', 0)}, "
                                   f"Buffer: {retrain_stats.get('buffered_samples', 0)} samples")
                        
                        # Check retraining recommendation
                        if hasattr(self, 'performance_tracker') and self.performance_tracker:
                            recommendation = self.performance_tracker.get_retraining_recommendation()
                            if recommendation['should_retrain']:
                                logger.warning(f"⚠️ Retraining recommended: {', '.join(recommendation['reasons'])}")
                    except Exception as retrain_error:
                        logger.debug(f"Could not fetch retraining stats: {retrain_error}")
                
                # Model version info (with safety checks)
                if hasattr(self, 'model_updater') and self.model_updater:
                    try:
                        versions = self.model_updater.get_model_versions()
                        if versions:
                            active_versions = [v for v in versions if v.status == 'active']
                            logger.info(f"MODELS - Active: {len(active_versions)}, "
                                       f"Total versions: {len(versions)}")
                    except Exception as model_error:
                        logger.debug(f"Could not fetch model versions: {model_error}")
                
                # Enhanced monitoring checks with alerts
                if status['consecutive_errors'] >= 3:
                    self.monitor._send_critical_alert(f"High error count in live trading: {status['consecutive_errors']}")
                
                if metrics_summary['max_drawdown_pct'] > 10:
                    self.monitor._send_performance_alert(f"Significant drawdown in live trading: {metrics_summary['max_drawdown_pct']:.2f}%", "warning")
                
                # Check for unusual trading patterns
                if status['total_trades'] > 0:
                    try:
                        if hasattr(live_trader, 'session_start_time') and live_trader.session_start_time:
                            trading_duration = (datetime.now() - live_trader.session_start_time).total_seconds() / 3600
                        else:
                            trading_duration = max(1, monitoring_interval / 3600)
                        
                        recent_trade_rate = status['total_trades'] / max(1, trading_duration)
                        if recent_trade_rate > 50:
                            self.monitor._send_performance_alert(f"High trade frequency detected: {recent_trade_rate:.1f} trades/hour", "warning")
                            
                    except Exception as trade_rate_error:
                        logger.debug(f"Could not calculate trade rate: {trade_rate_error}")
                
                # Log additional monitoring metrics
                self.monitor.ingestion_metrics['live_trading_status'] = {
                    'timestamp': datetime.now().isoformat(),
                    'portfolio_value': status['current_portfolio_value'],
                    'total_return_pct': metrics_summary['total_return_pct'],
                    'total_trades': status['total_trades'],
                    'consecutive_errors': status['consecutive_errors'],
                    'max_drawdown_pct': metrics_summary['max_drawdown_pct']
                }
                    
                time.sleep(monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring thread error: {e}")
                self.monitor.log_step_warning("live_trading_monitoring", f"Monitoring error: {e}")
                time.sleep(monitoring_interval)
                
                
                
    def get_pipeline_health_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline health status"""
        return {
            'pipeline_id': self.pipeline_id,
            'monitoring_summary': self.monitor.get_monitoring_summary(),
            'system_health': self.monitor._assess_system_health(),
            'timestamp': datetime.now().isoformat()
        }

    def save_monitoring_checkpoint(self, checkpoint_name: str = None):
        """Save monitoring checkpoint for recovery purposes"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_data = {
            'pipeline_id': self.pipeline_id,
            'checkpoint_name': checkpoint_name,
            'timestamp': datetime.now().isoformat(),
            'monitoring_data': self.monitor.get_monitoring_summary(),
            'config_snapshot': self.config
        }
        
        checkpoint_path = REPORT_DIR / "monitoring" / f"{checkpoint_name}.json"
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            logger.info(f"Monitoring checkpoint saved to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save monitoring checkpoint: {e}")


if __name__ == '__main__':
    try:
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent
        config_file_path = project_root / "config" / "config.yaml"
        
        config = load_config(config_file_path)

        if config:
            setup_directories(config)
            pipeline = TradebookPipeline(config=config)
            
            # Save initial monitoring checkpoint
            pipeline.monitor = ProductionMonitor(config)
            pipeline.save_monitoring_checkpoint("pipeline_start")
            
            pipeline.initialize_modules()
            pipeline.run_pipeline()
            
            # Save final monitoring checkpoint
            pipeline.save_monitoring_checkpoint("pipeline_complete")
            
        else:
            logger.error("Failed to load configuration. Pipeline cannot start.")
    except FileNotFoundError as e:
        logger.error(f"Failed to run pipeline: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.critical(f"Pipeline terminated with critical error: {type(e).__name__}: {e}")
        
        # Attempt to save emergency monitoring report
        try:
            emergency_report = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat(),
                'emergency_shutdown': True
            }
            
            emergency_path = Path("logs/emergency_shutdown_report.json")
            emergency_path.parent.mkdir(exist_ok=True)
            with open(emergency_path, 'w') as f:
                json.dump(emergency_report, f, indent=2, default=str)
            logger.info(f"Emergency report saved to {emergency_path}")
        except:
            pass  # Don't fail on emergency reporting
        
        raise
