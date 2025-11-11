# tests/synthetic_data/test_synthesizer.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from tradebook_pipeline.synthetic_data.synthesizer import SyntheticDataGenerator
import json
from pathlib import Path


@pytest.fixture
def mock_config():
    """Returns a mock configuration matching the actual config.yaml structure."""
    return {
        'enabled': True,
        'generator_model_type': 'Gaussian',
        'model_name': 'Gaussian',
        'model_path': 'models/synthetic_data/gaussian_model.pkl',
        'num_sequences': 1000,
        'seq_len': 24,
        'feature_cols': [],  # Empty for auto-detection
        'training_params': {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 0.001,
            'model_save_path': 'models/synthetic_data/'
        },
        'timegan_params': {
            'sequence_length': 24,
            'hidden_dim': 24,
            'gamma_param': 1.0,
            'noise_dim': 32,
            'num_layers': 3
        },
        'augmentation': {
            'enabled': False,
            'noise_type': 'gaussian',
            'noise_level': 0.01,
            'apply_to_columns': []
        },
        'validation': {
            'enabled': True,
            'max_mean_diff_pct': 20,
            'max_std_diff_pct': 25,
            'min_ks_pvalue': 0.01
        },
        'benchmarking': {
            'enabled': True,
            'save_reports': True,
            'report_path': 'reports/synthetic_data/',
            'metrics': [
                'distribution_similarity',
                'statistical_fidelity',
                'correlation_preservation'
            ]
        },
        'output_filename': 'synthetic_tradebook_data.parquet',
        'output_directory': 'data/synthetic/datasets/'
    }


@pytest.fixture
def mock_tradebook_df():
    """Returns a mock tradebook DataFrame matching actual pipeline data."""
    np.random.seed(42)
    data = {
        'price': np.random.uniform(90, 110, 100),
        'volume': np.random.uniform(50, 150, 100),
        'type': np.random.choice(['a', 'b'], 100),  # Already mapped from 'side'
        'is_peak': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    # Add index to simulate time-based data
    df.index = pd.date_range(start='2024-01-01', periods=100, freq='h')
    return df


@pytest.fixture
def synthesizer_instance(mock_config, tmp_path):
    """Returns an initialized SyntheticDataGenerator instance."""
    # Update config to use temporary path
    config = mock_config.copy()
    config['training_params']['model_save_path'] = str(tmp_path)
    
    # Patch NoiseAugmenter since augmentation is disabled
    with patch('tradebook_pipeline.synthetic_data.synthesizer.NoiseAugmenter') as MockAugmenter:
        instance = SyntheticDataGenerator(config=config)
        # Augmenter should be None when disabled
        assert instance.augmenter is None
        return instance


def test_synthesizer_initialization(mock_config, tmp_path):
    """Tests the SyntheticDataGenerator initializes properties correctly."""
    config = mock_config.copy()
    config['training_params']['model_save_path'] = str(tmp_path)
    
    with patch('tradebook_pipeline.synthetic_data.synthesizer.NoiseAugmenter'):
        synthesizer = SyntheticDataGenerator(config=config)
        
        assert synthesizer.generator_model_type == 'Gaussian'
        assert synthesizer.seq_len == 24
        assert synthesizer.feature_cols is None  # Not set until training
        assert synthesizer.augmenter is None  # Disabled in config
        assert not synthesizer.is_generator_trained
        assert synthesizer.generator_model is None
        assert synthesizer.model_save_path == Path(tmp_path)


def test_detect_feature_columns(synthesizer_instance, mock_tradebook_df):
    """Tests automatic feature column detection."""
    df = mock_tradebook_df.copy()
    feature_cols = synthesizer_instance._detect_feature_columns(df)
    
    # Should include numeric and categorical columns
    assert 'price' in feature_cols
    assert 'volume' in feature_cols
    assert 'type' in feature_cols
    
    # Should exclude metadata columns
    assert 'is_peak' not in feature_cols


def test_validate_training_data(synthesizer_instance, mock_tradebook_df):
    """Tests training data validation."""
    is_valid, training_data = synthesizer_instance._validate_training_data(mock_tradebook_df)
    
    assert is_valid
    assert synthesizer_instance.feature_cols is not None
    assert len(synthesizer_instance.feature_cols) > 0
    assert 'n_samples' in synthesizer_instance.training_metadata
    assert 'n_features' in synthesizer_instance.training_metadata
    assert synthesizer_instance.training_metadata['n_samples'] == len(mock_tradebook_df)



def test_validate_training_data_empty_dataframe(synthesizer_instance):
    """Tests validation with empty DataFrame."""
    empty_df = pd.DataFrame()
    is_valid, _ = synthesizer_instance._validate_training_data(empty_df)
    
    assert not is_valid


def test_train_generator_gaussian(synthesizer_instance, mock_tradebook_df, tmp_path):
    """Tests the Gaussian training logic and model saving."""
    synthesizer_instance.model_save_path = tmp_path
    
    # Train the generator (uses real file I/O)
    synthesizer_instance.train_generator(mock_tradebook_df)
    
    # Verify training completed
    assert synthesizer_instance.is_generator_trained
    assert synthesizer_instance.generator_model is not None
    assert synthesizer_instance.generator_model['model_type'] == 'Gaussian'
    
    # Verify statistics were stored
    assert 'numeric_stats' in synthesizer_instance.generator_model
    assert 'categorical_stats' in synthesizer_instance.generator_model
    
    # Verify model file was created
    model_file = tmp_path / "gaussian_generator.json"
    assert model_file.exists()
    
    # Verify file contents
    with open(model_file, 'r') as f:
        saved_data = json.load(f)
    assert saved_data['model_type'] == 'Gaussian'
    assert 'generator_model' in saved_data


def test_train_generator_ctgan(synthesizer_instance, mock_tradebook_df, tmp_path):
    """Tests CTGAN training (falls back to Gaussian implementation)."""
    synthesizer_instance.model_save_path = tmp_path
    
    synthesizer_instance.train_generator(mock_tradebook_df, model_type='CTGAN')
    
    assert synthesizer_instance.is_generator_trained
    assert synthesizer_instance.generator_model['model_type'] == 'CTGAN'
    assert synthesizer_instance.generator_model_type == 'CTGAN'
    
    # Should still have Gaussian-style statistics
    assert 'numeric_stats' in synthesizer_instance.generator_model


def test_train_generator_timegan(synthesizer_instance, mock_tradebook_df, tmp_path):
    """Tests TimeGAN training (falls back to Gaussian implementation)."""
    synthesizer_instance.model_save_path = tmp_path
    
    synthesizer_instance.train_generator(mock_tradebook_df, model_type='TimeGAN')
    
    assert synthesizer_instance.is_generator_trained
    assert synthesizer_instance.generator_model['model_type'] == 'TimeGAN'
    assert synthesizer_instance.generator_model_type == 'TimeGAN'


def test_train_generator_invalid_data(synthesizer_instance):
    """Tests training with invalid (empty) data."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="Training data validation failed"):
        synthesizer_instance.train_generator(empty_df)


def test_train_generator_unsupported_model_type(synthesizer_instance, mock_tradebook_df):
    """Tests training with unsupported model type."""
    with pytest.raises(ValueError, match="Unsupported generator model type"):
        synthesizer_instance.train_generator(mock_tradebook_df, model_type='GAN')


def test_generate_data_gaussian(synthesizer_instance, mock_tradebook_df):
    """Tests synthetic data generation using Gaussian model."""
    # Train first
    synthesizer_instance.train_generator(mock_tradebook_df)
    
    # Generate synthetic data
    num_sequences = 5
    synthetic_data = synthesizer_instance.generate_data(num_sequences=num_sequences, validate=False)
    
    # Verify output
    assert isinstance(synthetic_data, pd.DataFrame)
    assert not synthetic_data.empty
    assert len(synthetic_data) == num_sequences * synthesizer_instance.seq_len
    
    # Verify required columns
    assert 'is_synthetic' in synthetic_data.columns
    assert 'sequence_id' in synthetic_data.columns
    assert all(synthetic_data['is_synthetic'])
    
    # Verify feature columns are present
    for col in synthesizer_instance.feature_cols:
        assert col in synthetic_data.columns


def test_generate_data_distribution_quality(synthesizer_instance, mock_tradebook_df):
    """Tests that generated data has similar distribution to training data."""
    synthesizer_instance.train_generator(mock_tradebook_df)
    synthetic_data = synthesizer_instance.generate_data(num_sequences=10, validate=False)
    
    # Check numeric columns have reasonable distributions
    for col in ['price', 'volume']:
        real_mean = mock_tradebook_df[col].mean()
        synth_mean = synthetic_data[col].mean()
        real_std = mock_tradebook_df[col].std()
        synth_std = synthetic_data[col].std()
        
        # Mean should be within 30% (generous for small sample)
        assert abs(synth_mean - real_mean) / real_mean < 0.3
        # Std should be within 50% (more variable for small sample)
        assert abs(synth_std - real_std) / real_std < 0.5


def test_generate_data_without_training(synthesizer_instance, mock_tradebook_df):
    """Tests generation when model is not trained (should auto-train)."""
    synthetic_data = synthesizer_instance.generate_data(
        num_sequences=2, 
        raw_data_base=mock_tradebook_df,
        validate=False
    )
    
    assert synthesizer_instance.is_generator_trained
    assert not synthetic_data.empty


def test_generate_data_without_training_no_base_data(synthesizer_instance):
    """Tests generation without training and no base data (should raise error)."""
    with pytest.raises(RuntimeError, match="Generator not trained"):
        synthesizer_instance.generate_data(num_sequences=2, raw_data_base=None)


def test_generate_data_unsupported_model(synthesizer_instance, mock_tradebook_df):
    """Tests generation with unsupported model type."""
    # Train with a valid type first
    synthesizer_instance.train_generator(mock_tradebook_df)
    
    # Change to unsupported type
    synthesizer_instance.generator_model_type = 'GAN'
    
    with pytest.raises(ValueError, match="Generation not implemented"):
        synthesizer_instance.generate_data(num_sequences=1)


def test_generate_data_with_validation(synthesizer_instance, mock_tradebook_df, tmp_path):
    """Tests synthetic data generation with validation enabled."""
    synthesizer_instance.model_save_path = tmp_path
    
    # Train
    synthesizer_instance.train_generator(mock_tradebook_df)
    
    # Generate with validation
    synthetic_data = synthesizer_instance.generate_data(
        num_sequences=3, 
        validate=True
    )
    
    assert not synthetic_data.empty
    assert synthesizer_instance.validator is not None


def test_get_generation_summary(synthesizer_instance, mock_tradebook_df):
    """Tests getting generation summary."""
    synthesizer_instance.train_generator(mock_tradebook_df)
    
    summary = synthesizer_instance.get_generation_summary()
    
    assert 'model_type' in summary
    assert 'is_trained' in summary
    assert 'feature_columns' in summary
    assert 'sequence_length' in summary
    assert 'training_metadata' in summary
    assert 'augmentation_enabled' in summary
    assert summary['is_trained'] == True
    assert summary['model_type'] == 'Gaussian'
    assert summary['sequence_length'] == 24
    assert summary['augmentation_enabled'] == False


def test_validator_distribution_similarity(mock_tradebook_df):
    """Tests the SyntheticDataValidator distribution similarity calculation."""
    from tradebook_pipeline.synthetic_data.synthesizer import SyntheticDataValidator
    
    validator = SyntheticDataValidator(mock_tradebook_df)
    
    # Create similar synthetic data
    synthetic_df = mock_tradebook_df.copy()
    synthetic_df['price'] = synthetic_df['price'] + np.random.normal(0, 1, len(synthetic_df))
    
    similarities = validator.calculate_distribution_similarity(synthetic_df)
    
    assert 'price' in similarities
    assert 'ks_statistic' in similarities['price']
    assert 'ks_pvalue' in similarities['price']
    assert 'mean_diff_pct' in similarities['price']
    assert 'std_diff_pct' in similarities['price']


def test_validator_validate_synthetic_data(mock_tradebook_df):
    """Tests the SyntheticDataValidator validation."""
    from tradebook_pipeline.synthetic_data.synthesizer import SyntheticDataValidator
    
    validator = SyntheticDataValidator(mock_tradebook_df)
    
    # Create synthetic data with same structure
    synthetic_df = mock_tradebook_df.copy()
    synthetic_df['price'] = np.random.uniform(90, 110, len(synthetic_df))
    
    is_valid, results = validator.validate_synthetic_data(synthetic_df)
    
    assert 'is_valid' in results
    assert 'issues' in results
    assert 'warnings' in results
    assert 'metrics' in results
    assert 'distribution_similarity' in results['metrics']


def test_validator_detects_missing_columns(mock_tradebook_df):
    """Tests validator detects missing columns."""
    from tradebook_pipeline.synthetic_data.synthesizer import SyntheticDataValidator
    
    validator = SyntheticDataValidator(mock_tradebook_df)
    
    # Create synthetic data missing columns
    synthetic_df = mock_tradebook_df[['price', 'volume']].copy()
    
    is_valid, results = validator.validate_synthetic_data(synthetic_df)
    
    assert not is_valid
    assert len(results['issues']) > 0


def test_model_save_and_metadata(synthesizer_instance, mock_tradebook_df, tmp_path):
    """Tests model saving and metadata generation."""
    synthesizer_instance.model_save_path = tmp_path
    
    # Train (uses real file operations)
    synthesizer_instance.train_generator(mock_tradebook_df)
    
    # Check that model file was created
    model_file = tmp_path / "gaussian_generator.json"
    assert model_file.exists()
    
    # Verify file contents
    with open(model_file, 'r') as f:
        saved_data = json.load(f)
    
    assert 'model_type' in saved_data
    assert 'generator_model' in saved_data
    assert 'training_metadata' in saved_data
    assert 'feature_columns' in saved_data
    assert 'seq_len' in saved_data
    assert saved_data['model_type'] == 'Gaussian'
    assert saved_data['seq_len'] == 24


def test_augmentation_disabled(mock_config, tmp_path):
    """Tests that augmentation is properly disabled when configured."""
    config = mock_config.copy()
    config['training_params']['model_save_path'] = str(tmp_path)
    config['augmentation']['enabled'] = False
    
    with patch('tradebook_pipeline.synthetic_data.synthesizer.NoiseAugmenter'):
        synthesizer = SyntheticDataGenerator(config=config)
        assert synthesizer.augmenter is None


def test_augmentation_enabled(mock_config, tmp_path):
    """Tests that augmentation is properly enabled when configured."""
    config = mock_config.copy()
    config['training_params']['model_save_path'] = str(tmp_path)
    config['augmentation']['enabled'] = True
    
    with patch('tradebook_pipeline.synthetic_data.synthesizer.NoiseAugmenter') as MockAugmenter:
        synthesizer = SyntheticDataGenerator(config=config)
        assert synthesizer.augmenter is not None
        MockAugmenter.assert_called_once()


def test_numeric_and_categorical_statistics(synthesizer_instance, mock_tradebook_df):
    """Tests that both numeric and categorical statistics are properly captured."""
    synthesizer_instance.train_generator(mock_tradebook_df)
    
    model = synthesizer_instance.generator_model
    
    # Check numeric stats
    assert 'numeric_stats' in model
    assert 'price' in model['numeric_stats']
    assert 'volume' in model['numeric_stats']
    
    # Each numeric stat should have mean, std, min, max, etc.
    for col in ['price', 'volume']:
        assert 'mean' in model['numeric_stats'][col]
        assert 'std' in model['numeric_stats'][col]
        assert 'min' in model['numeric_stats'][col]
        assert 'max' in model['numeric_stats'][col]
    
    # Check categorical stats
    assert 'categorical_stats' in model
    assert 'type' in model['categorical_stats']
    assert 'values' in model['categorical_stats']['type']
    assert 'probabilities' in model['categorical_stats']['type']
