from flask import Flask, request, jsonify
import pandas as pd
from loguru import logger
import io
import sys
from waitress import serve # For production deployment
import base64 # Added for base64 decoding
from pathlib import Path # Ensure Path is imported

# Add parent directory to path to allow imports from synthetic_data and config
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import ConfigLoader
from synthetic_data.manager import SyntheticDataManager
# from synthetic_data.augmenters.noise_augmenter import NoiseAugmenter # Removed: Not used
# from system_models.pump_characteristics import PumpCharacteristicParams # Removed: Not used in active code

# Configure logging for the API server
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "synthetic_data_api.log", rotation="10 MB", level="INFO")
logger.add(LOG_DIR / "synthetic_data_api_error.log", rotation="10 MB", level="ERROR")


app = Flask(__name__)

# Load configuration once when the server starts
app_config = ConfigLoader.get_config()
synthetic_config = ConfigLoader.get_section("synthetic_data_generation")

# Initialize SyntheticDataManager globally or per request if stateful
# For simplicity, initialized once here. If manager holds mutable state for training,
# consider re-initializing or passing state correctly.
synthetic_manager = SyntheticDataManager(synthetic_config)

# @app.route("/health")
# def health_check():
#     return {"status": "ok"}, 200
#   
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks."""
    return jsonify({"status": "healthy", "service": "synthetic_data_api"}), 200

@app.route('/generate', methods=['POST'])
def generate_synthetic_data_endpoint():
    """
    API endpoint to generate synthetic data.
    Expects a JSON payload with 'num_sequences' and optionally 'raw_data_base64'.
    If raw_data_base64 is not provided, manager uses its default/trained data.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400

        num_sequences = data.get('num_sequences')
        if not isinstance(num_sequences, int) or num_sequences <= 0:
            return jsonify({"error": "Invalid or missing 'num_sequences'. Must be a positive integer."}), 400

        raw_data_df = None
        if 'raw_data_base64' in data:
            # Decode base64 string to bytes, then load with pandas from a StringIO or BytesIO
            try:
                decoded_bytes = base64.b64decode(data['raw_data_base64'])
                # Assuming the decoded content is a JSON string suitable for pd.read_json
                raw_data_df = pd.read_json(io.BytesIO(decoded_bytes))
                logger.info(f"Received raw data (shape: {raw_data_df.shape}) for synthetic generation.")
            except Exception as e:
                logger.error(f"Failed to decode/load raw_data_base64: {e}")
                return jsonify({"error": "Failed to decode 'raw_data_base64'. Ensure it's valid base64-encoded JSON."}), 400

        # System characteristics can be passed in the request if dynamic,
        # otherwise manager uses defaults/pre-loaded ones.
        # For this example, we'll assume manager already has access to system_characteristics
        # via the config if needed for generation.
        # system_characteristics = data.get('system_characteristics', {}) # Or load from manager's config
        # For example:
        # system_characteristics = {k: PumpCharacteristicParams.from_dict(v) for k,v in system_characteristics.items()}
        
        synthetic_df = synthetic_manager.generate_synthetic_data(
            raw_data_base=raw_data_df, # Pass if provided, otherwise manager uses its default
            num_sequences=num_sequences,
            # Pass system_characteristics if they are dynamic per request
        )
        
        # Return synthetic data as JSON (or base64 CSV/Parquet for larger data)
        # Using JSON records for simplicity; for large datasets, consider Parquet binary via base64
        return jsonify({
            "status": "success",
            "num_generated_sequences": len(synthetic_df),
            "synthetic_data": synthetic_df.to_json(orient="records", date_format="iso")
        }), 200

    except Exception as e:
        logger.exception(f"Error during synthetic data generation request: {e}")
        return jsonify({"error": "Internal server error during generation.", "details": str(e)}), 500

@app.route('/train_generator', methods=['POST'])
def train_generator_endpoint():
    """
    API endpoint to trigger training of the synthetic data generator.
    Expects 'raw_data_base64' for training.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400

        raw_data_df = None
        if 'raw_data_base64' in data:
            try:
                decoded_bytes = base64.b64decode(data['raw_data_base64'])
                raw_data_df = pd.read_json(io.BytesIO(decoded_bytes))
            except Exception as e:
                logger.error(f"Failed to decode/load raw_data_base64 for training: {e}")
                return jsonify({"error": "Failed to decode 'raw_data_base64'. Ensure it's valid base64-encoded JSON."}), 400
        else:
            return jsonify({"error": "Missing 'raw_data_base64' for training."}), 400
        
        model_type = data.get('model_type', synthetic_config.get('generator_model_type', 'TimeGAN')) # Default from config
        
        logger.info(f"Starting training for synthetic data generator (type: {model_type})...")
        synthetic_manager.train_generator(raw_data_df, model_type=model_type)
        logger.info("Synthetic data generator training completed.")

        return jsonify({"status": "success", "message": "Synthetic data generator training initiated."}), 200

    except Exception as e:
        logger.exception(f"Error during generator training request: {e}")
        return jsonify({"error": "Internal server error during training.", "details": str(e)}), 500


if __name__ == '__main__':
    # When running with 'python server.py', use waitress for production-ready server
    # For development, you can use app.run(debug=True)
    logger.info("Starting Synthetic Data API Server...")
    try:
        # For production use, or a more robust local test
        serve(app, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Failed to start Synthetic Data API Server: {e}")
        sys.exit(1)
    # For local development without waitress (less robust):
    # app.run(debug=True, host='0.0.0.0', port=5000)
