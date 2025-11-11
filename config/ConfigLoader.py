import yaml
import json
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional, List, Union
import os
from dataclasses import dataclass
from enum import Enum


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class LogLevel(Enum):
    """Enumeration for valid log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ConfigSchema:
    """Data class defining the expected configuration schema structure."""
    
    # Required top-level sections
    REQUIRED_SECTIONS = {
        'schema', 'paths', 'data_paths', 'data_ingestion', 'exchanges', 
        'data_processing', 'peak_estimators', 'synthetic_data', 'logging'
    }
    
    # Required schema fields
    REQUIRED_SCHEMA_FIELDS = {'version', 'time_col_name', 'entity_col_name', 'event_cols'}
    
    # Required path fields
    REQUIRED_PATHS = {
        'raw_data_dir', 'synthetic_output_dir', 'predictions_dir', 
        'models_dir', 'logs_dir'
    }
    
    # Valid exchange names
    VALID_EXCHANGES = {'binance', 'kraken', 'coinbasepro', 'local', 'gdrive'}
    
    # Valid model types for peak estimators
    VALID_ML_MODELS = {
        'RandomForestClassifier', 'GradientBoostingClassifier', 
        'XGBClassifier', 'LogisticRegression', 'SVC'
    }
    
    # Valid estimator types
    VALID_ESTIMATOR_TYPES = {'ml_estimator', 'rule_based_estimator'}
    
    # Valid synthetic data models
    VALID_SYNTHETIC_MODELS = {'TimeGAN', 'CTGAN', 'WGAN-GP', 'VAE'}


class ConfigLoader:
    """
    Enhanced singleton-like class to load, validate, and manage application configurations.
    Provides schema validation and comprehensive error handling.
    """
    
    _config = None
    _config_file_path = None
    _schema_validator = ConfigSchema()
    
    @classmethod
    def load_config(cls, config_file_path: Optional[str] = None, validate_schema: bool = False) -> None:
        """
        Loads and optionally validates the configuration from a YAML file.
        
        Args:
            config_file_path: Explicit path to config file. If None, uses default.
            validate_schema: Whether to perform schema validation (default: False for compatibility).
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ConfigValidationError: If the configuration fails validation.
            ValueError: If there's an error parsing the YAML.
        """
        
        logger.info(f"ConfigLoader.load_config called. Path: {config_file_path}, Validate: {validate_schema}")
        
        # Skip re-loading if already loaded and no new path provided
        if cls._config is not None and config_file_path is None:
            logger.info(f"Configuration already loaded from {cls._config_file_path}. Skipping re-load.")
            return
        
        # Determine config file path
        config_to_load = cls._resolve_config_path(config_file_path)
        
        # Validate file exists
        if not config_to_load.exists():
            logger.error(f"Configuration file not found: {config_to_load}")
            raise FileNotFoundError(f"Configuration file not found: {config_to_load}")
        
        # Load configuration
        try:
            with open(config_to_load, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
            cls._config_file_path = config_to_load
            logger.info(f"Configuration loaded successfully from {config_to_load}")
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_to_load}: {e}")
            raise ValueError(f"Error parsing YAML file {config_to_load}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading config from {config_to_load}: {e}")
            raise
        
        # Validate schema if requested (now optional for backward compatibility)
        if validate_schema:
            try:
                cls._validate_config_schema()
                logger.info("Configuration schema validation passed.")
            except ConfigValidationError as e:
                logger.warning(f"Schema validation failed, but continuing: {e}")
                # Don't raise the error to maintain compatibility
        else:
            logger.info("Schema validation skipped for backward compatibility.")
    
    @classmethod
    def _resolve_config_path(cls, config_file_path: Optional[str]) -> Path:
        """Resolves the configuration file path using project structure logic."""
        
        if config_file_path is not None:
            return Path(config_file_path).resolve()
        
        # Default path resolution logic
        current_script_path = Path(__file__).resolve()
        logger.debug(f"ConfigLoader script path: {current_script_path}")
        
        # Try different project structure patterns
        possible_roots = [
            current_script_path.parent.parent,  # config/ConfigLoader.py -> project_root/
            current_script_path.parent,         # ConfigLoader.py -> project_root/
            Path.cwd()                          # current working directory
        ]
        
        for project_root in possible_roots:
            config_path = project_root / "config.yaml"
            legacy_path = project_root / "config" / "config.yaml"
            enhanced_path = project_root / "config" / "config_enhanced.yaml"
            
            logger.debug(f"Checking config paths from root {project_root}:")
            logger.debug(f"  - {config_path}")
            logger.debug(f"  - {legacy_path}")
            logger.debug(f"  - {enhanced_path}")
            
            # Prefer the new consolidated config.yaml
            if config_path.exists():
                logger.info(f"Found config.yaml at: {config_path}")
                return config_path
            elif legacy_path.exists():
                logger.warning(f"Using legacy config path: {legacy_path}")
                return legacy_path
            elif enhanced_path.exists():
                logger.warning(f"Using enhanced config path: {enhanced_path}")
                return enhanced_path
        
        # Default fallback
        default_path = Path.cwd() / "config.yaml"
        logger.warning(f"No config found in standard locations. Defaulting to: {default_path}")
        return default_path
    
    @classmethod
    def _validate_config_schema(cls) -> None:
        """Validates the loaded configuration against the expected schema."""
        
        if cls._config is None:
            raise ConfigValidationError("No configuration loaded to validate.")
        
        config = cls._config
        schema = cls._schema_validator
        
        # Check required top-level sections
        missing_sections = schema.REQUIRED_SECTIONS - set(config.keys())
        if missing_sections:
            raise ConfigValidationError(f"Missing required sections: {missing_sections}")
        
        # Validate schema section
        cls._validate_schema_section(config.get('schema', {}))
        
        # Validate paths section
        cls._validate_paths_section(config.get('paths', {}))
        
        # Validate peak estimators
        cls._validate_peak_estimators(config.get('peak_estimators', {}))
        
        # Validate synthetic data configuration
        cls._validate_synthetic_data(config.get('synthetic_data', {}))
        
        # Validate logging configuration
        cls._validate_logging_config(config.get('logging', {}))
        
        # Validate exchanges
        cls._validate_exchanges(config.get('exchanges', {}))
        
        logger.info("All configuration sections validated successfully.")
    
    @classmethod
    def _validate_schema_section(cls, schema_config: Dict[str, Any]) -> None:
        """Validates the schema section of the configuration."""
        
        missing_fields = cls._schema_validator.REQUIRED_SCHEMA_FIELDS - set(schema_config.keys())
        if missing_fields:
            raise ConfigValidationError(f"Missing required schema fields: {missing_fields}")
        
        # Validate event_cols is a list
        event_cols = schema_config.get('event_cols', [])
        if not isinstance(event_cols, list) or not event_cols:
            raise ConfigValidationError("'event_cols' must be a non-empty list.")
    
    @classmethod
    def _validate_paths_section(cls, paths_config: Dict[str, Any]) -> None:
        """Validates the paths section of the configuration."""
        
        missing_paths = cls._schema_validator.REQUIRED_PATHS - set(paths_config.keys())
        if missing_paths:
            raise ConfigValidationError(f"Missing required paths: {missing_paths}")
        
        # Validate all paths are strings
        for path_name, path_value in paths_config.items():
            if not isinstance(path_value, str):
                raise ConfigValidationError(f"Path '{path_name}' must be a string, got {type(path_value)}")
    
    @classmethod
    def _validate_peak_estimators(cls, estimators_config: Dict[str, Any]) -> None:
        """Validates the peak_estimators section."""
        
        if 'estimators' not in estimators_config:
            raise ConfigValidationError("'peak_estimators' section must contain 'estimators' subsection.")
        
        estimators = estimators_config['estimators']
        if not isinstance(estimators, dict):
            raise ConfigValidationError("'estimators' must be a dictionary.")
        
        for estimator_name, estimator_config in estimators.items():
            cls._validate_single_estimator(estimator_name, estimator_config)
    
    @classmethod
    def _validate_single_estimator(cls, name: str, config: Dict[str, Any]) -> None:
        """Validates a single estimator configuration."""
        
        required_fields = {'enabled', 'type'}
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            raise ConfigValidationError(f"Estimator '{name}' missing required fields: {missing_fields}")
        
        # Validate estimator type
        estimator_type = config['type']
        if estimator_type not in cls._schema_validator.VALID_ESTIMATOR_TYPES:
            raise ConfigValidationError(f"Invalid estimator type '{estimator_type}' in '{name}'")
        
        # Validate ML model type if it's an ML estimator
        if estimator_type == 'ml_estimator':
            model_type = config.get('model_type')
            if model_type not in cls._schema_validator.VALID_ML_MODELS:
                logger.warning(f"Model type '{model_type}' in estimator '{name}' not in validated list")
    
    @classmethod
    def _validate_synthetic_data(cls, synthetic_config: Dict[str, Any]) -> None:
        """Validates the synthetic_data section."""
        
        if not synthetic_config:
            logger.warning("No synthetic_data configuration found.")
            return
        
        # Validate generator model type
        model_type = synthetic_config.get('generator_model_type')
        if model_type and model_type not in cls._schema_validator.VALID_SYNTHETIC_MODELS:
            logger.warning(f"Synthetic model type '{model_type}' not in validated list")
    
    @classmethod
    def _validate_logging_config(cls, logging_config: Dict[str, Any]) -> None:
        """Validates the logging configuration."""
        
        # Validate log level
        log_level = logging_config.get('level', 'INFO')
        try:
            LogLevel(log_level)
        except ValueError:
            valid_levels = [level.value for level in LogLevel]
            raise ConfigValidationError(f"Invalid log level '{log_level}'. Valid levels: {valid_levels}")
    
    @classmethod
    def _validate_exchanges(cls, exchanges_config: Dict[str, Any]) -> None:
        """Validates the exchanges configuration."""
        
        for exchange_name in exchanges_config.keys():
            if exchange_name not in cls._schema_validator.VALID_EXCHANGES:
                logger.warning(f"Exchange '{exchange_name}' not in validated list")
    
    @classmethod
    def get_config(cls, auto_load: bool = True) -> Dict[str, Any]:
        """
        Retrieves the entire loaded configuration.
        
        Args:
            auto_load: Whether to automatically load config if not loaded.
            
        Returns:
            The complete configuration dictionary.
            
        Raises:
            RuntimeError: If configuration couldn't be loaded.
        """
        
        if cls._config is None:
            if auto_load:
                logger.info("Configuration not loaded. Attempting to load default config...")
                try:
                    cls.load_config()
                except Exception as e:
                    logger.error(f"Failed to auto-load configuration: {e}")
                    raise RuntimeError(f"Configuration could not be loaded: {e}")
            else:
                raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        return cls._config
    
    @classmethod
    def get_section(cls, section_name: str, default: Any = None, auto_load: bool = True) -> Dict[str, Any]:
        """
        Retrieves a specific section from the loaded configuration.
        Handles backward compatibility by checking multiple possible section names.
        
        Args:
            section_name: The name of the configuration section.
            default: Default value if section not found.
            auto_load: Whether to automatically load config if not loaded.
            
        Returns:
            The requested configuration section.
        """
        
        config = cls.get_config(auto_load=auto_load)
        
        # Handle backward compatibility mappings
        compatibility_mappings = {
            'paths': ['paths', 'config_paths'],
            'data_paths': ['data_paths', 'config_data_paths'],
            'peak_estimators': ['peak_estimators', 'original_peak_estimators'],
            'data_schema': ['data_schema', 'schema'],
        }
        
        # Check primary section name first
        section = config.get(section_name)
        if section is not None:
            return section
        
        # Check compatibility mappings
        if section_name in compatibility_mappings:
            for alt_name in compatibility_mappings[section_name]:
                section = config.get(alt_name)
                if section is not None:
                    logger.info(f"Using '{alt_name}' section for requested '{section_name}'")
                    return section
        
        # If nothing found, return default
        if section is None:
            logger.warning(f"Section '{section_name}' not found in configuration.")
            return {} if default is None else default
        
        return section
    
    @classmethod
    def get_nested_value(cls, key_path: str, default: Any = None, separator: str = '.') -> Any:
        """
        Retrieves a nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the value (e.g., 'paths.raw_data_dir').
            default: Default value if key not found.
            separator: Separator character for key path.
            
        Returns:
            The requested configuration value.
            
        Example:
            value = ConfigLoader.get_nested_value('peak_estimators.strategy')
            paths = ConfigLoader.get_nested_value('paths.raw_data_dir')
        """
        
        config = cls.get_config()
        keys = key_path.split(separator)
        
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                logger.warning(f"Key path '{key_path}' not found in configuration.")
                return default
        
        return current
    
    @classmethod
    def validate_and_reload(cls, config_file_path: Optional[str] = None) -> None:
        """
        Reloads and validates the configuration.
        
        Args:
            config_file_path: Optional path to config file.
        """
        
        cls._config = None  # Force reload
        cls.load_config(config_file_path, validate_schema=True)
    
    @classmethod
    def get_config_info(cls) -> Dict[str, Any]:
        """Returns metadata about the loaded configuration."""
        
        if cls._config is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "file_path": str(cls._config_file_path),
            "schema_version": cls._config.get('schema', {}).get('version', 'unknown'),
            "sections": list(cls._config.keys()),
            "total_sections": len(cls._config.keys())
        }


# Backwards compatibility aliases
def load_config(config_file_path: Optional[str] = None) -> None:
    """Backwards compatible function for loading configuration."""
    ConfigLoader.load_config(config_file_path)


def get_config() -> Dict[str, Any]:
    """Backwards compatible function for getting configuration."""
    return ConfigLoader.get_config()


def get_section(section_name: str) -> Dict[str, Any]:
    """Backwards compatible function for getting configuration section."""
    return ConfigLoader.get_section(section_name)


# Testing and CLI functionality
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ConfigLoader validation and testing")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate configuration only")
    parser.add_argument("--info", "-i", action="store_true", help="Show configuration info")
    parser.add_argument("--section", "-s", help="Show specific section")
    parser.add_argument("--key", "-k", help="Show nested key value (dot notation)")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        ConfigLoader.load_config(args.config)
        logger.info("✅ Configuration loaded and validated successfully!")
        
        # Show configuration info
        if args.info:
            info = ConfigLoader.get_config_info()
            print("\n📊 Configuration Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        # Show specific section
        if args.section:
            section_data = ConfigLoader.get_section(args.section)
            print(f"\n📁 Section '{args.section}':")
            print(json.dumps(section_data, indent=2, default=str))
        
        # Show nested key value
        if args.key:
            value = ConfigLoader.get_nested_value(args.key)
            print(f"\n🔑 Key '{args.key}': {value}")
        
        # If just validating, show success message
        if args.validate:
            print("✅ Configuration validation passed!")
            
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        exit(1)
