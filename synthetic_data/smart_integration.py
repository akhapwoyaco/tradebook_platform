import pandas as pd
from loguru import logger
from typing import Dict, Any, Optional
from pathlib import Path # Import Path for directory operations

# Import local manager and remote client
from synthetic_data.manager import SyntheticDataManager
from synthetic_data.remote.client import SyntheticDataClient
from system_models.pump_characteristics import PumpCharacteristicParams # For type hinting system_characteristics

class SmartSyntheticIntegration:
    """
    Acts as an intelligent router for synthetic data operations.
    It abstracts away whether generation/training happens locally or via a remote API,
    based on the configuration.
    """
    def __init__(self, full_config: Dict[str, Any]):
        self.full_config = full_config
        self.synthetic_config = full_config.get('synthetic_data_generation', {})
        self.generation_mode = self.synthetic_config.get('generation_mode', 'local')
        
        self.local_manager = None
        self.remote_client = None

        logger.info(f"SmartSyntheticIntegration initialized. Default generation mode: {self.generation_mode}")

    def _get_local_manager(self) -> SyntheticDataManager:
        """Initializes and returns the local SyntheticDataManager."""
        if self.local_manager is None:
            # SyntheticDataManager expects the 'synthetic_data_generation' section of the config
            self.local_manager = SyntheticDataManager(self.synthetic_config)
            logger.debug("Initialized local SyntheticDataManager.")
        return self.local_manager

    def _get_remote_client(self) -> SyntheticDataClient:
        """Initializes and returns the remote SyntheticDataClient."""
        if self.remote_client is None:
            # SyntheticDataClient might need broader config for API endpoints etc.
            self.remote_client = SyntheticDataClient(self.full_config)
            logger.debug("Initialized remote SyntheticDataClient.")
        return self.remote_client

    def generate_data(
        self,
        raw_data: pd.DataFrame,
        num_sequences: int,
        client_type: Optional[str] = None, # Overrides configured mode
        output_filename: Optional[str] = None, # For local saving
        system_characteristics: Optional[Dict[str, PumpCharacteristicParams]] = None # For informed generation
    ) -> pd.DataFrame:
        """
        Generates synthetic data, routing the request based on `client_type`
        or configured `generation_mode`.
        
        Args:
            raw_data (pd.DataFrame): The raw data to base synthetic generation on.
            num_sequences (int): Number of synthetic sequences to generate.
            client_type (str, optional): 'local' or 'remote'. Overrides the config's `generation_mode`.
            output_filename (str, optional): If provided and using local mode, save to this filename.
            system_characteristics (Dict[str, PumpCharacteristicParams], optional): Pump characteristic
                                                                                    objects to guide generation.
                                                                                    
        Returns:
            pd.DataFrame: The generated synthetic data.
        """
        effective_client_type = client_type if client_type else self.generation_mode
        logger.info(f"Generating synthetic data using '{effective_client_type}' client.")

        generated_data = pd.DataFrame()
        output_path = None # Initialize outside try-block

        if effective_client_type == 'local':
            manager = self._get_local_manager()
            generated_data = manager.generate_synthetic_data(
                num_sequences=num_sequences,
                raw_data_base=raw_data,
                system_characteristics=system_characteristics # Pass system characteristics
            )
            # Save locally if output_filename is specified or default is set
            # Access the output directory from the 'data_paths' section of the full config
            output_dir = Path(self.full_config.get('data_paths', {}).get('synthetic_output_dir', 'data/synthetic/datasets/'))
            
            if output_filename:
                output_path = output_dir / output_filename
            else:
                # Use a dynamic filename based on timestamp for uniqueness if not specified
                timestamp_str = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
                default_filename = f"generated_data_{timestamp_str}.parquet"
                output_path = output_dir / default_filename
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generated_data.to_parquet(output_path, index=True) # Save with timestamp index
            logger.info(f"Local generated data saved to {output_path}. Shape: {generated_data.shape}")

        elif effective_client_type == 'remote':
            client = self._get_remote_client()
            generated_data = client.generate_data(
                num_sequences=num_sequences,
                raw_data_base=raw_data # Remote client handles JSON conversion
            )
            # Remote generation might not save locally by default, but you could add
            # a local caching mechanism if needed. For now, it just returns the DF.
            logger.info(f"Remote generated data received. Shape: {generated_data.shape}")

        else:
            raise ValueError(f"Unsupported synthetic data client type: {effective_client_type}")
        
        return generated_data

    def train_generator(
        self,
        raw_data: pd.DataFrame,
        model_type: str,
        client_type: Optional[str] = None # Overrides configured mode
    ):
        """
        Triggers training of the synthetic data generator, routing the request.
        
        Args:
            raw_data (pd.DataFrame): The raw data to train the generator on.
            model_type (str): The type of generator model to train (e.g., 'TimeGAN').
            client_type (str, optional): 'local' or 'remote'. Overrides the config's `generation_mode`.
        """
        effective_client_type = client_type if client_type else self.generation_mode
        logger.info(f"Training synthetic data generator using '{effective_client_type}' client.")

        if effective_client_type == 'local':
            manager = self._get_local_manager()
            manager.train_generator(raw_data=raw_data, model_type=model_type)
        elif effective_client_type == 'remote':
            client = self._get_remote_client()
            client.train_generator(raw_data=raw_data, model_type=model_type)
        else:
            raise ValueError(f"Unsupported synthetic data client type for training: {effective_client_type}")
        
        logger.info(f"Synthetic data generator training initiated successfully via {effective_client_type}.")

# Example usage is in example_dual_mode.py
