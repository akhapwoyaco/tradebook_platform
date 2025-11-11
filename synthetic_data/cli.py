### `tradebook_pipeline/synthetic_data/cli.py`
import argparse
import pandas as pd
from loguru import logger
from pathlib import Path
import sys
import os

# Add project root to sys.path to ensure imports work from CLI
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import ConfigLoader
from synthetic_data.smart_integration import SmartSyntheticIntegration
from synthetic_data.manager import SyntheticDataManager # For direct local use or internal calls
from synthetic_data.api.server import app as flask_app_for_server_run # For running Flask server
from waitress import serve

# Configure logging for the CLI
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "synthetic_data_cli.log", rotation="10 MB", level="INFO")
logger.add(LOG_DIR / "synthetic_data_cli_error.log", rotation="10 MB", level="ERROR")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Helper function to load raw data."""
    logger.info(f"Loading raw data from {path}...")
    try:
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        logger.info(f"Raw data loaded: {df}")
        logger.info(f"Raw data loaded: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {path}")
        raise
    except Exception as e:
        logger.exception(f"Error loading raw data from {path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="CLI for Tradebook Pipeline's Synthetic Data module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config-path',
        type=str,
        default='synthetic_data/config_enhanced.yaml',
        help='Path to the main configuration YAML file.'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- Generate Subparser ---
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic data.')
    generate_parser.add_argument(
        '--num-sequences',
        type=int,
        default=1,
        help='Number of synthetic data sequences to generate.'
    )
    generate_parser.add_argument(
        '--client-type',
        type=str,
        choices=['local', 'remote'],
        default=None, # Will default to config value if not provided
        help='Specify whether to use local generation or remote API. Overrides config.'
    )
    generate_parser.add_argument(
        '--raw-data-path',
        type=str,
        default='data/raw/sample_data.csv',
        help='Path to the raw data CSV to base synthetic generation on.'
    )
    generate_parser.add_argument(
        '--output-file',
        type=str,
        default=None, # Will use config default if not provided
        help='Optional: Output file path for generated data relative to output_dir.'
    )

    # --- Train Subparser ---
    train_parser = subparsers.add_parser('train', help='Train the synthetic data generator model.')
    train_parser.add_argument(
        '--model-type',
        type=str,
        default=None, # Will default to config value
        help='Specify the type of generator model to train (e.g., TimeGAN, CTGAN).'
    )
    train_parser.add_argument(
        '--raw-data-path',
        type=str,
        default='data/raw/sample_data.csv',
        help='Path to the raw data CSV to train the generator on.'
    )
    train_parser.add_argument(
        '--client-type',
        type=str,
        choices=['local', 'remote'],
        default=None, # Will default to config value if not provided
        help='Specify whether to train locally or via remote API. Overrides config.'
    )

    # --- Server Subparser ---
    server_parser = subparsers.add_parser('server', help='Run the synthetic data generation API server.')
    server_parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host IP for the Flask server.'
    )
    server_parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for the Flask server.'
    )
    server_parser.add_argument(
        '--debug',
        action='store_true',
        help='Run Flask server in debug mode (development only).'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        ConfigLoader.load_config(Path(args.config_path))
        full_config = ConfigLoader.get_config()
        synthetic_config = ConfigLoader.get_section('synthetic_data_generation')
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        sys.exit(1)

    if args.command == 'generate':
        raw_data_path = Path(args.raw_data_path)
        if not raw_data_path.exists():
            logger.error(f"Raw data file not found: {raw_data_path}. Cannot generate synthetic data.")
            sys.exit(1)
            
        raw_data = load_raw_data(raw_data_path)
        
        # Determine client type
        client_type = args.client_type if args.client_type else synthetic_config.get('generation_mode', 'local')

        # Initialize SmartSyntheticIntegration
        integration = SmartSyntheticIntegration(full_config)

        try:
            logger.info(f"Starting synthetic data generation (mode: {client_type})...")
            generated_data = integration.generate_data(
                raw_data=raw_data,
                client_type=client_type,
                num_sequences=args.num_sequences,
                output_filename=args.output_file # Pass CLI specified output file
            )
            logger.success(f"Successfully generated {len(generated_data)} synthetic data points.")
            logger.info(f"Generated data saved to {synthetic_config.get('output_dir', 'data/synthetic/datasets/')}/{args.output_file if args.output_file else synthetic_config.get('output_filename', 'generated_data.parquet')}")
        except Exception as e:
            logger.exception(f"Error during synthetic data generation: {e}")
            sys.exit(1)

    elif args.command == 'train':
        raw_data_path = Path(args.raw_data_path)
        if not raw_data_path.exists():
            logger.error(f"Raw data file not found: {raw_data_path}. Cannot train generator.")
            sys.exit(1)
            
        raw_data = load_raw_data(raw_data_path)
        
        # Determine client type
        client_type = args.client_type if args.client_type else synthetic_config.get('generation_mode', 'local')
        model_type = args.model_type if args.model_type else synthetic_config.get('generator_model_type', 'TimeGAN')

        integration = SmartSyntheticIntegration(full_config)

        try:
            logger.info(f"Starting synthetic data generator training for {model_type} (mode: {client_type})...")
            integration.train_generator(
                raw_data=raw_data,
                model_type=model_type,
                client_type=client_type
            )
            logger.success(f"Successfully initiated training for {model_type} generator.")
        except Exception as e:
            logger.exception(f"Error during synthetic data generator training: {e}")
            sys.exit(1)

    elif args.command == 'server':
        logger.info(f"Starting Synthetic Data API server on {args.host}:{args.port} (Debug: {args.debug})...")
        try:
            if args.debug:
                flask_app_for_server_run.run(host=args.host, port=args.port, debug=True)
            else:
                # Use waitress for production deployment
                serve(flask_app_for_server_run, host=args.host, port=args.port)
        except Exception as e:
            logger.critical(f"Failed to start server: {e}")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

