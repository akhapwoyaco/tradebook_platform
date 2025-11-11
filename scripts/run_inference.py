import pandas as pd
from loguru import logger
from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# # --- CORRECTED IMPORT ---
# from tradebook_pipeline.config.ConfigLoader import ConfigLoader
# # ------------------------
# 
# from tradebook_pipeline.main_pipeline.TradebookPipeline import TradebookPipeline # Reuse main pipeline components
# 

# --- CORRECTED IMPORT ---
from config.ConfigLoader import ConfigLoader
# ------------------------

from main_pipeline.TradebookPipeline import TradebookPipeline # Reuse main pipeline components


# Configure logging for the inference script
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "run_inference.log", rotation="10 MB", level="INFO")
# Add backtrace and diagnose for error logs to get full context on critical failures
logger.add(LOG_DIR / "run_inference_error.log", rotation="10 MB", level="ERROR", backtrace=True, diagnose=True)

def run_inference_script(
    input_data_path: str,
    estimator_name: str = "ml_time_series_v1",
    output_filename: str = None,
    evaluate: bool = True # Flag to indicate if true labels are expected for evaluation
):
    """
    Loads new data, runs peak detection inference, and optionally evaluates performance.
    
    Args:
        input_data_path (str): Path to the CSV/Parquet file containing new data.
        estimator_name (str): The name of the trained estimator to use for inference.
        output_filename (str, optional): Custom filename for saving predictions.
        evaluate (bool): If True, expects 'is_peak' column in input_data_path for evaluation.
    """
    logger.info(f"Starting inference script for estimator: {estimator_name}")
    
    try:
        # Load configuration
        ConfigLoader.load_config()
        full_config = ConfigLoader.get_config()
        
        pipeline = TradebookPipeline()

        # 1. Load Input Data
        input_path = Path(input_data_path)
        logger.info(f"Loading inference data from {input_path}...")
        if input_path.suffix == '.csv':
            input_df = pd.read_csv(input_path)
        elif input_path.suffix == '.parquet':
            input_df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported input data format: {input_path.suffix}. Please use .csv or .parquet.")
        
        if input_df.empty:
            logger.error(f"Input data file '{input_data_path}' is empty. Cannot proceed with inference.")
            return

        # Handle indexing: prioritize MultiIndex for multi-sequence data
        if 'sequence_id' in input_df.columns and 'timestamp' in input_df.columns:
            logger.info("Detected 'sequence_id' and 'timestamp' columns. Setting MultiIndex.")
            input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
            input_df = input_df.set_index(['sequence_id', 'timestamp'])
        elif 'timestamp' in input_df.columns:
            logger.info("Detected 'timestamp' column. Setting as DatetimeIndex.")
            input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
            input_df = input_df.set_index('timestamp')
        else:
            logger.warning("Neither 'timestamp' nor ('sequence_id', 'timestamp') found as columns. Proceeding with default DataFrame index. Ensure your estimator can handle this input format.")
            # Depending on strictness, you might raise an error here if timestamp is mandatory.

        logger.info(f"Inference data loaded. Shape: {input_df.shape}, Index: {input_df.index.names if input_df.index.names != [None] else 'Default (integer)'}")

        # 2. Run Inference
        # Pass a copy to run_inference to prevent accidental in-place modifications of input_df
        predicted_df = pipeline.run_inference(input_df.copy(), estimator_name=estimator_name)
        
        # Override default output path if custom filename is provided
        if output_filename:
            output_dir = pipeline.predictions_output_dir
            custom_output_path = output_dir / output_filename
            custom_output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
            predicted_df.to_parquet(custom_output_path, index=True) # Save with index (timestamp/MultiIndex)
            logger.info(f"Predictions saved to custom path: {custom_output_path}")

        # 3. Evaluate Performance (if true labels are available and evaluation is requested)
        # predicted_df should contain 'is_peak' if it was present in input_df
        if evaluate and 'is_peak' in predicted_df.columns:
            logger.info("True labels ('is_peak' column) found. Proceeding with evaluation.")
            pipeline.evaluate_performance(predicted_df, estimator_name=estimator_name)
        elif evaluate and 'is_peak' not in predicted_df.columns:
            logger.warning("Evaluation requested but 'is_peak' column (true labels) not found in input data or predictions. Skipping evaluation.")
        else:
            logger.info("Evaluation skipped as requested or true labels not needed for this run.")

        logger.info("Inference script completed successfully.")

    except Exception as e:
        logger.critical(f"Inference script failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run peak detection inference on new data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-data-path',
        type=str,
        required=True,
        help='Path to the input data file (CSV or Parquet) for inference.'
    )
    parser.add_argument(
        '--estimator-name',
        type=str,
        default='ml_time_series_v1',
        help='Name of the trained peak estimator to use (e.g., "rule_based_v1", "ml_time_series_v1").'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default=None,
        help='Optional: Custom filename for the output predictions (e.g., "live_predictions.parquet"). Predictions are saved to the configured predictions output directory.'
    )
    parser.add_argument(
        '--no-evaluate',
        action='store_true',
        help='Do not perform evaluation even if true labels are present in the input data.'
    )

    args = parser.parse_args()

    # Call the main function with parsed arguments
    run_inference_script(
        input_data_path=args.input_data_path,
        estimator_name=args.estimator_name,
        output_filename=args.output_filename,
        evaluate=not args.no_evaluate # 'evaluate' is True by default, --no-evaluate sets it to False
    )
    
# import pandas as pd
# from loguru import logger
# from pathlib import Path
# import sys
# import os

# # Add project root to sys.path
# sys.path.append(str(Path(__file__).resolve().parents[2]))

# from config import ConfigLoader
# from main_pipeline import TradebookPipeline # Reuse main pipeline components

# # Configure logging for the inference script
# LOG_DIR = Path("logs")
# LOG_DIR.mkdir(parents=True, exist_ok=True)
# logger.add(LOG_DIR / "run_inference.log", rotation="10 MB", level="INFO")
# # Add backtrace and diagnose for error logs to get full context on critical failures
# logger.add(LOG_DIR / "run_inference_error.log", rotation="10 MB", level="ERROR", backtrace=True, diagnose=True)

# def run_inference_script(
#     input_data_path: str,
#     estimator_name: str = "ml_time_series_v1",
#     output_filename: str = None,
#     evaluate: bool = True # Flag to indicate if true labels are expected for evaluation
# ):
#     """
#     Loads new data, runs peak detection inference, and optionally evaluates performance.
    
#     Args:
#         input_data_path (str): Path to the CSV/Parquet file containing new data.
#         estimator_name (str): The name of the trained estimator to use for inference.
#         output_filename (str, optional): Custom filename for saving predictions.
#         evaluate (bool): If True, expects 'is_peak' column in input_data_path for evaluation.
#     """
#     logger.info(f"Starting inference script for estimator: {estimator_name}")
    
#     try:
#         # Load configuration
#         ConfigLoader.load_config()
#         full_config = ConfigLoader.get_config()
        
#         pipeline = TradebookPipeline()

#         # 1. Load Input Data
#         input_path = Path(input_data_path)
#         logger.info(f"Loading inference data from {input_path}...")
#         if input_path.suffix == '.csv':
#             input_df = pd.read_csv(input_path)
#         elif input_path.suffix == '.parquet':
#             input_df = pd.read_parquet(input_path)
#         else:
#             raise ValueError(f"Unsupported input data format: {input_path.suffix}. Please use .csv or .parquet.")
        
#         if input_df.empty:
#             logger.error(f"Input data file '{input_data_path}' is empty. Cannot proceed with inference.")
#             return

#         # Handle indexing: prioritize MultiIndex for multi-sequence data
#         if 'sequence_id' in input_df.columns and 'timestamp' in input_df.columns:
#             logger.info("Detected 'sequence_id' and 'timestamp' columns. Setting MultiIndex.")
#             input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
#             input_df = input_df.set_index(['sequence_id', 'timestamp'])
#         elif 'timestamp' in input_df.columns:
#             logger.info("Detected 'timestamp' column. Setting as DatetimeIndex.")
#             input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
#             input_df = input_df.set_index('timestamp')
#         else:
#             logger.warning("Neither 'timestamp' nor ('sequence_id', 'timestamp') found as columns. Proceeding with default DataFrame index. Ensure your estimator can handle this input format.")
#             # Depending on strictness, you might raise an error here if timestamp is mandatory.

#         logger.info(f"Inference data loaded. Shape: {input_df.shape}, Index: {input_df.index.names if input_df.index.names != [None] else 'Default (integer)'}")

#         # 2. Run Inference
#         # Pass a copy to run_inference to prevent accidental in-place modifications of input_df
#         predicted_df = pipeline.run_inference(input_df.copy(), estimator_name=estimator_name)
        
#         # Override default output path if custom filename is provided
#         if output_filename:
#             output_dir = pipeline.predictions_output_dir
#             custom_output_path = output_dir / output_filename
#             custom_output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
#             predicted_df.to_parquet(custom_output_path, index=True) # Save with index (timestamp/MultiIndex)
#             logger.info(f"Predictions saved to custom path: {custom_output_path}")

#         # 3. Evaluate Performance (if true labels are available and evaluation is requested)
#         # predicted_df should contain 'is_peak' if it was present in input_df
#         if evaluate and 'is_peak' in predicted_df.columns:
#             logger.info("True labels ('is_peak' column) found. Proceeding with evaluation.")
#             pipeline.evaluate_performance(predicted_df, estimator_name=estimator_name)
#         elif evaluate and 'is_peak' not in predicted_df.columns:
#             logger.warning("Evaluation requested but 'is_peak' column (true labels) not found in input data or predictions. Skipping evaluation.")
#         else:
#             logger.info("Evaluation skipped as requested or true labels not needed for this run.")

#         logger.info("Inference script completed successfully.")

#     except Exception as e:
#         logger.critical(f"Inference script failed: {e}", exc_info=True)
#         sys.exit(1)

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Run peak detection inference on new data.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument(
#         '--input-data-path',
#         type=str,
#         required=True,
#         help='Path to the input data file (CSV or Parquet) for inference.'
#     )
#     parser.add_argument(
#         '--estimator-name',
#         type=str,
#         default='ml_time_series_v1',
#         help='Name of the trained peak estimator to use (e.g., "rule_based_v1", "ml_time_series_v1").'
#     )
#     parser.add_argument(
#         '--output-filename',
#         type=str,
#         default=None,
#         help='Optional: Custom filename for the output predictions (e.g., "live_predictions.parquet"). Predictions are saved to the configured predictions output directory.'
#     )
#     parser.add_argument(
#         '--no-evaluate',
#         action='store_true',
#         help='Do not perform evaluation even if true labels are present in the input data.'
#     )

#     args = parser.parse_args()

#     # Call the main function with parsed arguments
#     run_inference_script(
#         input_data_path=args.input_data_path,
#         estimator_name=args.estimator_name,
#         output_filename=args.output_filename,
#         evaluate=not args.no_evaluate # 'evaluate' is True by default, --no-evaluate sets it to False
#     )
