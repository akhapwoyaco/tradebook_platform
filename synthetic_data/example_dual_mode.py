import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import os
import time

# Add project root to sys.path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import ConfigLoader
from synthetic_data.smart_integration import SmartSyntheticIntegration

# Configure logging for the example script
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "example_dual_mode.log", rotation="10 MB", level="INFO")
logger.add(LOG_DIR / "example_dual_mode_error.log", rotation="10 MB", level="ERROR")

def create_dummy_raw_data(num_points: int = 100) -> pd.DataFrame:
    """Creates a dummy DataFrame to simulate raw pump data."""
    logger.info(f"Creating dummy raw data with {num_points} points...")
    data = {
        'date': pd.date_range(start='2023-01-01', periods=num_points, freq='H'),
        'price': np.sin(np.linspace(0, 20, num_points)) * 10 + 50 + np.random.randn(num_points) * 2,
        'amount': np.abs(np.cos(np.linspace(0, 20, num_points))) * 100 + 50 + np.random.randn(num_points) * 10,
        
        'is_peak': np.zeros(num_points, dtype=int)
    }
    df_dummy = pd.DataFrame(data)

    # Introduce some artificial peaks
    peak_indices = np.random.choice(num_points, 5, replace=False)
    for idx in peak_indices:
        df_dummy.loc[idx:min(idx+5, num_points-1), 'is_peak'] = 1
        df_dummy.loc[idx, 'price'] *= 1.5
        df_dummy.loc[idx, 'amount'] *= 2.0
    
    df_dummy = df_dummy.set_index('date') # Set date as index
    logger.info("Dummy raw data created successfully.")
    return df_dummy

def run_dual_mode_example():
    """
    Demonstrates the SmartSyntheticIntegration in both local and remote modes.
    """
    logger.info("Starting dual mode synthetic data generation example.")

    # 1. Load Configuration
    try:
        # Load from the main configuration path (e.g., project_root/config.yaml)
        ConfigLoader.load_config() 
        full_config = ConfigLoader.get_config()
        synthetic_config = ConfigLoader.get_section("synthetic_data_generation")
        data_paths_config = ConfigLoader.get_section("data_paths") # Get data_paths section
        logger.info("Configuration loaded.")
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        return

    # 2. Create Dummy Raw Data
    raw_data_base = create_dummy_raw_data(num_points=50)

    # 3. Initialize SmartSyntheticIntegration
    # The integration object will be re-initialized for each mode to ensure config changes are picked up.
    integration = SmartSyntheticIntegration(full_config)

    # Determine the base output directory for synthetic data
    # This should be consistent with the 'data_paths.synthetic_output_dir' in config_enhanced.yaml
    output_base_dir = Path(data_paths_config.get('synthetic_output_dir', 'data/synthetic/datasets/'))
    output_base_dir.mkdir(parents=True, exist_ok=True) # Ensure the base directory exists

    # --- Test Local Generation ---
    logger.info("\n--- Running Local Synthetic Data Generation ---")
    synthetic_config['generation_mode'] = 'local' # Force local mode for this test
    integration = SmartSyntheticIntegration(full_config) # Re-initialize integration

    try:
        start_time = time.time()
        local_generated_data = integration.generate_data(
            raw_data=raw_data_base,
            client_type='local', # Explicitly request local for this test
            num_sequences=3
        )
        end_time = time.time()
        logger.info(f"Local generation completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Local generated data shape: {local_generated_data.shape}")
        
        # Save local generated data for inspection
        output_file_local = output_base_dir / "local_generated_data.parquet"
        local_generated_data.to_parquet(output_file_local, index=True)
        logger.info(f"Local generated data saved to {output_file_local}")

    except Exception as e:
        logger.error(f"Error during local synthetic data generation: {e}")

    # --- Test Remote Generation ---
    logger.info("\n--- Running Remote Synthetic Data Generation ---")
    synthetic_config['generation_mode'] = 'remote' # Force remote mode for this test
    integration = SmartSyntheticIntegration(full_config) # Re-initialize integration

    # IMPORTANT: Ensure your synthetic data API server is running in a separate terminal!
    # You can start it using the CLI:
    # `python -m tradebook_pipeline.synthetic_data.cli server`
    # Or directly:
    # `waitress-serve --listen=0.0.0.0:5000 synthetic_data.api.server:app`

    try:
        start_time = time.time()
        remote_generated_data = integration.generate_data(
            raw_data=raw_data_base,
            client_type='remote', # Explicitly request remote for this test
            num_sequences=2
        )
        end_time = time.time()
        logger.info(f"Remote generation completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Remote generated data shape: {remote_generated_data.shape}")

        # Save remote generated data for inspection
        output_file_remote = output_base_dir / "remote_generated_data.parquet"
        remote_generated_data.to_parquet(output_file_remote, index=True)
        logger.info(f"Remote generated data saved to {output_file_remote}")

    except Exception as e:
        logger.error(f"Error during remote synthetic data generation. "
                     f"Please ensure the API server is running at {synthetic_config.get('remote_api_url')}."
                     f" Details: {e}")
    
    logger.info("\nDual mode example finished.")

if __name__ == "__main__":
    run_dual_mode_example()