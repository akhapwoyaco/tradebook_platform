from typing import Dict, Any, Optional
from loguru import logger
from uuid import uuid4
import threading
import time
import pandas as pd # Added: Used in example usage
import numpy as np # Added: Used in example usage
import random # Added: Used in example usage

# This is a highly simplified JobManager for demonstration.
# In a real-world scenario, you would integrate with a proper
# distributed task queue system like Celery, RQ, or Apache Airflow.

class SyntheticDataJobManager:
    """
    Manages asynchronous synthetic data generation and training jobs.
    Uses simple in-memory tracking for demonstration.
    """
    _jobs: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("SyntheticDataJobManager initialized (in-memory, single-process).")

    def _update_job_status(self, job_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None):
        """Internal method to update job status safely."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]['status'] = status
                self._jobs[job_id]['last_updated'] = time.time()
                if result is not None: # Check for None explicitly
                    self._jobs[job_id]['result'] = result
                if error is not None: # Check for None explicitly
                    self._jobs[job_id]['error'] = error
            logger.info(f"Job {job_id} updated to status: {status}")

    def _run_job_async(self, job_id: str, target_func, *args, **kwargs):
        """Internal helper to run a function in a new thread and update job status."""
        try:
            self._update_job_status(job_id, "RUNNING")
            result = target_func(*args, **kwargs)
            self._update_job_status(job_id, "COMPLETED", result=result)
        except Exception as e:
            logger.exception(f"Job {job_id} failed with error: {e}")
            self._update_job_status(job_id, "FAILED", error=str(e))

    def submit_job(self, job_type: str, target_func, *args, **kwargs) -> str:
        """
        Submits a new asynchronous job.
        
        Args:
            job_type (str): Type of job (e.g., 'generate', 'train').
            target_func (callable): The function to execute in the job.
            *args, **kwargs: Arguments to pass to the target_func.
            
        Returns:
            str: The unique ID of the submitted job.
        """
        job_id = str(uuid4())
        with self._lock:
            self._jobs[job_id] = {
                'id': job_id,
                'type': job_type,
                'status': 'PENDING',
                'submitted_at': time.time(),
                'last_updated': time.time(),
                'result': None,
                'error': None,
                'args': args, # For debugging/auditing
                'kwargs': kwargs # For debugging/auditing
            }
        
        logger.info(f"Job {job_id} ({job_type}) submitted.")
        # Start the job in a new thread. For production, replace with Celery/RQ task.
        thread = threading.Thread(target=self._run_job_async, args=(job_id, target_func, *args), kwargs=kwargs)
        thread.daemon = True # Allow main program to exit even if threads are running
        thread.start()
        
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current status of a job.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves status of all managed jobs.
        """
        with self._lock:
            return self._jobs.copy()

    def cleanup_completed_jobs(self, older_than_seconds: int = 3600):
        """Removes completed or failed jobs older than specified seconds."""
        current_time = time.time()
        jobs_to_remove = []
        with self._lock:
            for job_id, job_info in self._jobs.items():
                if job_info['status'] in ['COMPLETED', 'FAILED'] and \
                   (current_time - job_info['last_updated']) > older_than_seconds:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                logger.info(f"Cleaned up job: {job_id}")

# Example Usage (for testing purposes)
if __name__ == "__main__":
    from synthetic_data.manager import SyntheticDataManager
    from config import ConfigLoader # Changed from synthetic_data.config_enhanced

    # Dummy configuration
    # For standalone execution, mock ConfigLoader or provide a direct config dict.
    # Here, we'll try to load from ConfigLoader assuming a default path,
    # or use a mock if ConfigLoader can't find config.
    try:
        from pathlib import Path
        # Adjust path for ConfigLoader if running this script directly
        # assuming config.py is two levels up from synthetic_data/jobs
        sys.path.append(str(Path(__file__).resolve().parents[3])) 
        app_config_for_test = ConfigLoader.get_config()
        mock_config = ConfigLoader.get_section("synthetic_data_generation")
        if not mock_config: # Fallback if section not found in default config
            raise ValueError("synthetic_data_generation section not found in config.")
    except Exception as e:
        logger.warning(f"Could not load config via ConfigLoader for example: {e}. Using hardcoded mock_config.")
        mock_config = { # Hardcoded fallback config
            'output_dir': 'data/synthetic/datasets/',
            'generator_model_type': 'TimeGAN',
            'seq_len': 24,
            'feature_cols': ['price', 'amount'],
            'output_filename': 'test_generated_data.parquet'
        }
        # Wrap this in a 'synthetic_data_generation' key to match expected structure
        mock_config = {'synthetic_data_generation': mock_config}
    
    # Mock SyntheticDataManager for the job manager to call
    class MockSyntheticDataManager:
        def __init__(self, config):
            self.config = config
            logger.info("Mock SyntheticDataManager initialized.")
        
        def generate_synthetic_data(self, raw_data_base, num_sequences, **kwargs):
            logger.info(f"Mocking generation of {num_sequences} sequences...")
            time.sleep(2) # Simulate work
            seq_len = self.config['synthetic_data_generation']['seq_len']
            feature_cols = self.config['synthetic_data_generation']['feature_cols']
            
            mock_data_dict = {
                'date': pd.date_range(start='2023-01-01', periods=num_sequences * seq_len, freq='H'),
            }
            for col in feature_cols:
                mock_data_dict[col] = np.random.rand(num_sequences * seq_len) * 100
            
            mock_data = pd.DataFrame(mock_data_dict)
            mock_data['is_synthetic'] = True
            mock_data = mock_data.set_index('date')
            logger.info(f"Mock generation complete. Shape: {mock_data.shape}")
            return mock_data

        def train_generator(self, raw_data, model_type):
            logger.info(f"Mocking training for {model_type} on {len(raw_data)} samples...")
            time.sleep(5) # Simulate work
            if random.random() < 0.2: # Simulate occasional failure
                raise ValueError("Mock training failed!")
            logger.info(f"Mock training for {model_type} complete.")
    
    # Initialize the job manager
    job_manager = SyntheticDataJobManager(mock_config) # Pass the (possibly wrapped) mock_config
    mock_sd_manager = MockSyntheticDataManager(mock_config) # Pass full config to mock manager

    # Create dummy raw data for testing
    num_points_raw = 100
    dummy_raw_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=num_points_raw, freq='H'),
        'price': np.random.rand(num_points_raw) * 100,
        'amount': np.random.rand(num_points_raw) * 1000,
        'is_peak': np.zeros(num_points_raw)
    }).set_index('date')

    # Submit a generation job
    gen_job_id = job_manager.submit_job(
        "generate", 
        mock_sd_manager.generate_synthetic_data, 
        raw_data_base=dummy_raw_data, 
        num_sequences=3
    )
    logger.info(f"Submitted generation job: {gen_job_id}")

    # Submit a training job
    train_job_id = job_manager.submit_job(
        "train", 
        mock_sd_manager.train_generator, 
        raw_data=dummy_raw_data, 
        model_type="TimeGAN"
    )
    logger.info(f"Submitted training job: {train_job_id}")

    # Monitor job statuses
    logger.info("Monitoring jobs (wait up to 10 seconds)...")
    for _ in range(10):
        gen_status = job_manager.get_job_status(gen_job_id)
        train_status = job_manager.get_job_status(train_job_id)
        logger.info(f"Gen Job ({gen_job_id[:8]}...): {gen_status['status'] if gen_status else 'N/A'}")
        logger.info(f"Train Job ({train_job_id[:8]}...): {train_status['status'] if train_status else 'N/A'}")
        if (gen_status and gen_status['status'] in ['COMPLETED', 'FAILED']) and \
           (train_status and train_status['status'] in ['COMPLETED', 'FAILED']):
            break
        time.sleep(1)

    # Check final results
    final_gen_status = job_manager.get_job_status(gen_job_id)
    final_train_status = job_manager.get_job_status(train_job_id)

    logger.info(f"\nFinal Generation Job Status: {final_gen_status}")
    logger.info(f"Final Training Job Status: {final_train_status}")

    # Clean up old jobs
    job_manager.cleanup_completed_jobs(older_than_seconds=1)
    logger.info(f"Jobs after cleanup: {job_manager.get_all_jobs()}")