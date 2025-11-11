import requests
import pandas as pd
import json
import io
import base64 # Added for base64 encoding/decoding
from loguru import logger
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path to allow imports from config
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.ConfigLoader import ConfigLoader
#from config import ConfigLoader

class SyntheticDataClient:
    """
    Client for interacting with the remote Synthetic Data API.
    Handles communication for generation and training requests.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('synthetic_data_generation', {})
        self.api_base_url = self.config.get('remote_api_url', 'http://localhost:5000')
        logger.info(f"SyntheticDataClient initialized. API Base URL: {self.api_base_url}")

    def generate_data(self, num_sequences: int, raw_data_base: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Sends a request to the remote API to generate synthetic data.
        
        Args:
            num_sequences (int): The number of synthetic sequences to generate.
            raw_data_base (pd.DataFrame, optional): Raw data to base the generation on.
                                                    If None, the generator uses its default/trained data.
                                                    Expected to be sent as base64-encoded JSON.
        Returns:
            pd.DataFrame: The generated synthetic data.
        
        Raises:
            requests.exceptions.RequestException: If there's an issue with the API request.
            ValueError: If the API returns an error.
        """
        endpoint = f"{self.api_base_url}/generate"
        payload = {"num_sequences": num_sequences}

        if raw_data_base is not None:
            # Convert DataFrame to JSON string, then base64 encode it
            json_str = raw_data_base.to_json(orient="records", date_format="iso")
            payload['raw_data_base64'] = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
            logger.info(f"Sending raw data base of shape {raw_data_base.shape} to remote generator.")
        else:
            logger.info("Generating synthetic data remotely without a raw data base.")

        logger.info(f"Requesting {num_sequences} synthetic sequences from {endpoint}...")
        try:
            response = requests.post(endpoint, json=payload, timeout=self.config.get('api_timeout_seconds', 300))
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            if response_data.get('status') == 'success':
                synthetic_data_json = response_data.get('synthetic_data')
                if synthetic_data_json:
                    # Load JSON string back into DataFrame
                    synthetic_df = pd.read_json(io.StringIO(synthetic_data_json))
                    logger.info(f"Successfully generated {len(synthetic_df)} synthetic samples remotely.")
                    # Assuming 'date' column exists and setting it as index
                    if 'date' in synthetic_df.columns:
                        synthetic_df['date'] = pd.to_datetime(synthetic_df['date'])
                        synthetic_df = synthetic_df.set_index('date')
                    return synthetic_df
                else:
                    raise ValueError("API returned success but no 'synthetic_data' in response.")
            else:
                error_msg = response_data.get('error', 'Unknown error from API.')
                raise ValueError(f"API Error: {error_msg}")

        except requests.exceptions.Timeout:
            logger.error(f"Request to {endpoint} timed out after {self.config.get('api_timeout_seconds', 300)} seconds.")
            raise requests.exceptions.Timeout("Remote API request timed out.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to remote API at {endpoint}: {e}")
            raise requests.exceptions.ConnectionError(f"Failed to connect to remote API: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request error during generation: {e}")
            raise
        except ValueError as e:
            logger.error(f"Data processing error after API response: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during remote data generation: {e}")
            raise

    def train_generator(self, raw_data: pd.DataFrame, model_type: str = "TimeGAN"):
        """
        Sends a request to the remote API to train the synthetic data generator.
        
        Args:
            raw_data (pd.DataFrame): The raw data to train the generator on.
            model_type (str): The type of generator model to train (e.g., 'TimeGAN').
            
        Raises:
            requests.exceptions.RequestException: If there's an issue with the API request.
            ValueError: If the API returns an error.
        """
        endpoint = f"{self.api_base_url}/train_generator"
        
        # Convert DataFrame to JSON string, then base64 encode it
        json_str = raw_data.to_json(orient="records", date_format="iso")
        base64_encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')

        payload = {
            "model_type": model_type,
            "raw_data_base64": base64_encoded_data
        }

        logger.info(f"Requesting remote training for {model_type} generator on {len(raw_data)} samples...")
        try:
            response = requests.post(endpoint, json=payload, timeout=self.config.get('api_timeout_seconds', 600)) # Longer timeout for training
            response.raise_for_status()

            response_data = response.json()
            if response_data.get('status') == 'success':
                logger.info(f"Remote generator training initiated successfully: {response_data.get('message')}")
            else:
                error_msg = response_data.get('error', 'Unknown error from API.')
                raise ValueError(f"API Error during training: {error_msg}")

        except requests.exceptions.Timeout:
            logger.error(f"Request to {endpoint} timed out after {self.config.get('api_timeout_seconds', 600)} seconds.")
            raise requests.exceptions.Timeout("Remote API training request timed out.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to remote API at {endpoint}: {e}")
            raise requests.exceptions.ConnectionError(f"Failed to connect to remote API: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request error during training: {e}")
            raise
        except ValueError as e:
            logger.error(f"Data processing error after API response during training: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during remote generator training: {e}")
            raise

# Example Usage (for testing purposes)
if __name__ == "__main__":
    from pathlib import Path
    import numpy as np

    # Dummy config for client
    client_config = {
        'synthetic_data_generation': {
            'remote_api_url': 'http://localhost:5000', # Ensure your server.py is running!
            'api_timeout_seconds': 30
        }
    }
    
    # Mocking ConfigLoader to inject dummy config for client test
    class MockConfigLoader:
        @classmethod
        def get_config(cls): return client_config
        @classmethod
        def get_section(cls, section_name): return client_config.get(section_name, {})

    # Overriding the global ConfigLoader for this test script only
    ConfigLoader = MockConfigLoader
    
    client = SyntheticDataClient(ConfigLoader.get_config())

    # Create dummy raw data for testing
    num_points_raw = 50
    dummy_raw_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=num_points_raw, freq='H'),
        'price': np.random.rand(num_points_raw) * 100,
        'amount': np.random.rand(num_points_raw) * 1000,
        'is_peak': np.zeros(num_points_raw)
    }).set_index('date')

    logger.info("Attempting to generate data remotely...")
    try:
        # Test generation with raw data base
        generated_data = client.generate_data(num_sequences=2, raw_data_base=dummy_raw_data)
        logger.info(f"Client received generated data of shape: {generated_data.shape}")
        logger.info(generated_data.head())
        
        # Test generation without raw data base (server should use its default)
        generated_data_no_base = client.generate_data(num_sequences=1)
        logger.info(f"Client received generated data (no base) of shape: {generated_data_no_base.shape}")
        logger.info(generated_data_no_base.head())

        logger.info("\nAttempting to train generator remotely...")
        client.train_generator(raw_data=dummy_raw_data, model_type="TimeGAN")
        logger.info("Remote training request sent.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Remote client test failed due to network/API issue: {e}")
        logger.error("Please ensure synthetic_data/api/server.py is running on localhost:5000")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during client test: {e}")
