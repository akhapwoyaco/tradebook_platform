# tests/test_pipeline.py - FINAL CORRECTED VERSION

import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from tradebook_pipeline.TradebookPipeline import TradebookPipeline, setup_directories
import pandas as pd
import time
import threading
from functools import wraps # Needed for patching


# Use the full path for internal imports if simple name patching fails
# Note: Patching the classes in the namespace of the TradebookPipeline module (e.g., 'tradebook_pipeline.TradebookPipeline.SyntheticDataGenerator')
@patch('tradebook_pipeline.TradebookPipeline.SyntheticDataGenerator')
@patch('tradebook_pipeline.TradebookPipeline.LiveTrader')
@patch('tradebook_pipeline.TradebookPipeline.EnhancedBacktester')
@patch('tradebook_pipeline.TradebookPipeline.DataProcessor')
@patch('tradebook_pipeline.TradebookPipeline.IngestionManager')
@patch('tradebook_pipeline.TradebookPipeline.ToolsManager') 
def test_pipeline_initialization(
    MockToolsManager,
    MockIngestionManager, 
    MockDataProcessor, 
    MockBacktester, 
    MockLiveTrader, 
    MockSyntheticDataGenerator, 
    mock_config
):
    """Tests the TradebookPipeline class initializes its modules correctly."""
    
    # Enable synthetic data for initialization check
    mock_config['synthetic_data'] = {'enabled': True}
    
    pipeline = TradebookPipeline(config=mock_config)
    pipeline.initialize_modules()

    # Assert that all main components are instantiated
    MockIngestionManager.assert_called_once_with(config=mock_config)
    MockDataProcessor.assert_called_once_with(config=mock_config.get('data_processing', {}))
    MockToolsManager.assert_called_once_with(config=mock_config.get('core_analysis', {}))
    
    # Assert synthesizer is called since it's enabled in the mock config above
    MockSyntheticDataGenerator.assert_called_once_with(config=mock_config.get('synthetic_data', {}))
    
    # Asserting correct attributes are set
    assert pipeline.ingestion_manager is MockIngestionManager.return_value
    assert pipeline.data_processor is MockDataProcessor.return_value
    assert pipeline.tools_manager is MockToolsManager.return_value
    assert pipeline.synthesizer is MockSyntheticDataGenerator.return_value

def test_setup_directories_creation_and_cleanup(tmp_path, mock_config):
    """Tests the setup_directories helper function creates necessary directories."""
    
    # Set config paths to use the temp directory
    temp_config = mock_config.copy()
    # These path keys map directly to the global variables used in setup_directories
    temp_config['paths']['models_dir'] = str(tmp_path / 'models')
    temp_config['paths']['logs_dir'] = str(tmp_path / 'logs')
    temp_config['paths']['predictions_dir'] = str(tmp_path / 'reports')
    temp_config['paths']['data_dir'] = str(tmp_path / 'data')
    
    setup_directories(temp_config)
    
    # Assert directories were created, matching the logic in TradebookPipeline.py
    assert (tmp_path / 'data').is_dir()
    assert (tmp_path / 'models').is_dir()
    assert (tmp_path / 'logs').is_dir()
    assert (tmp_path / 'reports').is_dir()
    assert (tmp_path / 'data/processed').is_dir()
    assert (tmp_path / 'data/synthetic').is_dir()


