import pytest
from pathlib import Path
from loguru import logger
from unittest.mock import patch
import pandas as pd
import math

from system_models.base_characteristics import CharacteristicParams
from system_models.pump_characteristics import PumpCharacteristicParams


# Test suite for the base class
def test_characteristic_params_initialization():
    params = CharacteristicParams()
    assert params.name == "BaseSystemComponent"
    assert params.description == "General system component characteristics."
    assert params.additional_params == {}

def test_characteristic_params_to_dict():
    params = CharacteristicParams(
        name="Test",
        additional_params={"key": "value"}
    )
    result = params.to_dict()
    assert result['name'] == "Test"
    assert result['additional_params']['key'] == "value"

# Test suite for the Pump-specific class
def test_pump_characteristic_params_initialization():
    params = PumpCharacteristicParams()
    assert params.name == "GenericPump"
    assert params.max_flow_rate_lps == 100.0
    assert params.min_flow_rate_lps == 0.0
    assert params.efficiency_percentage == 85.0
    assert params.flow_rate_normal_range == [30.0, 90.0]

def test_pump_characteristic_params_custom_values():
    params = PumpCharacteristicParams(
        name="HighFlowPump",
        max_flow_rate_lps=500.0,
        efficiency_percentage=95.0
    )
    assert params.name == "HighFlowPump"
    assert params.max_flow_rate_lps == 500.0
    assert params.efficiency_percentage == 95.0

@patch.object(logger, 'warning')
@patch.object(logger, 'error')
def test_pump_characteristic_params_post_init_validation_warnings(mock_error, mock_warning):
    """
    Tests that the __post_init__ method logs warnings and errors for invalid data.
    """
    # Test invalid efficiency (WARNING)
    params_low_eff = PumpCharacteristicParams(efficiency_percentage=105.0)
    mock_warning.assert_called_with(
        f"Pump '{params_low_eff.name}': Efficiency percentage 105.0 is out of typical range (0-100)."
    )

    # Test invalid flow range (ERROR)
    mock_error.reset_mock()
    params_invalid_flow = PumpCharacteristicParams(min_flow_rate_lps=150.0, max_flow_rate_lps=100.0)
    mock_error.assert_called_with(
        f"Pump '{params_invalid_flow.name}': Min flow rate (150.0) should be less than max flow rate (100.0)."
    )
    
    # Test invalid normal ranges (WARNING)
    mock_warning.reset_mock()
    params_invalid_range = PumpCharacteristicParams(flow_rate_normal_range=[100, 50])
    mock_warning.assert_called_with(
        f"Pump '{params_invalid_range.name}': Invalid flow_rate_normal_range: [100, 50]. Resetting to min/max flow rates."
    )
    assert params_invalid_range.flow_rate_normal_range == [0.0, 100.0]
    
def test_pump_characteristic_params_get_operating_limits():
    params = PumpCharacteristicParams(nominal_flow_rate_lps=80.0)
    limits = params.get_operating_limits()
    
    assert limits['nominal_flow_rate_lps'] == 80.0
    assert 'min_flow_rate_lps' in limits
    assert 'max_pressure_bar' in limits

@patch.object(logger, 'warning')
def test_pump_characteristic_params_calculate_power_consumption_nominal(mock_warning):
    params = PumpCharacteristicParams(
        power_rating_kw=10.0,
        nominal_flow_rate_lps=50.0,
        nominal_pressure_bar=4.0
    )
    # At nominal conditions, power should be close to the rating
    power = params.calculate_power_consumption(flow_rate=50.0, pressure=4.0)
    assert math.isclose(power, 10.0, rel_tol=1e-2)
    mock_warning.assert_not_called()

@patch.object(logger, 'warning')
def test_pump_characteristic_params_calculate_power_consumption_outside_range(mock_warning):
    params = PumpCharacteristicParams(
        power_rating_kw=10.0,
        nominal_flow_rate_lps=50.0,
        nominal_pressure_bar=4.0,
        min_flow_rate_lps=10.0,
        max_flow_rate_lps=100.0,
        min_pressure_bar=1.0,
        max_pressure_bar=8.0
    )
    # Test for a value far outside the allowed range
    power = params.calculate_power_consumption(flow_rate=200.0, pressure=5.0)
    
    # The method should return a penalized nominal power (1.2 * rating)
    assert power == pytest.approx(12.0)
    mock_warning.assert_called_with(
        "Calculated power for flow 200.0 and pressure 5.0 outside nominal pump operating envelope for GenericPump. Returning penalized nominal power."
    )
    
    # Test for a value within the range but far from nominal
    power_low = params.calculate_power_consumption(flow_rate=20.0, pressure=2.0)
    assert power_low < params.power_rating_kw
    
# import pandas as pd
# import numpy as np
# import pytest
# from pathlib import Path
# import shutil
# import logging # Import logging to configure caplog levels if needed, though pytest handles it
# 
# # Assuming these imports are correct relative to your project structure
# from system_models.base_characteristics import CharacteristicParams
# from system_models.pump_characteristics import PumpCharacteristicParams
# # Assuming RuleBasedPeakEstimator is still part of your project and its tests are here
# from peak_estimators.strategies.rule_based_estimator import RuleBasedPeakEstimator
# from peak_estimators.strategies.base_estimator import BasePeakEstimator # For type check
# 
# 
# # Fixture for a temporary directory to save/load models
# @pytest.fixture(scope="module")
# def temp_test_models_dir():
#     """
#     Provides a temporary directory for saving and loading models during tests.
#     Ensures the directory is clean before and after tests.
#     """
#     # Changed name to be more generic as this file tests multiple components
#     test_dir = Path("models/test_characteristics_temp/")
#     if test_dir.exists():
#         shutil.rmtree(test_dir)
#     test_dir.mkdir(parents=True, exist_ok=True)
#     yield test_dir
#     shutil.rmtree(test_dir)
# 
# # Fixture for a basic config for the RuleBasedPeakEstimator (kept for completeness of original file)
# @pytest.fixture
# def base_rule_config():
#     """
#     Provides a standard configuration dictionary for the RuleBasedPeakEstimator.
#     Aligns with expected column names 'price' and 'amount'.
#     """
#     return {
#         'price_column': 'price',
#         'amount_column': 'amount',
#         'price_rise_threshold': 0.10, # 10% rise
#         'lookback_window': 3,         # 3 timesteps for lookback
#         'price_drop_threshold': 0.03, # 3% drop
#         'min_amount_increase': 0.20,  # 20% amount increase
#         'check_next_timesteps': 2     # Check next 2 timesteps for drop
#     }
# 
# # Fixture for sample data with a clear peak (kept for completeness of original file)
# @pytest.fixture
# def sample_data_with_peak():
#     """
#     Provides sample DataFrame with a 'date' column as index,
#     and includes 'price', 'amount', 'other_feature', and 'is_peak'.
#     Designed to trigger a peak based on base_rule_config.
#     """
#     data = {
#         'date': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00',
#                                 '2023-01-01 03:00:00', '2023-01-01 04:00:00', '2023-01-01 05:00:00',
#                                 '2023-01-01 06:00:00', '2023-01-01 07:00:00', '2023-01-01 08:00:00',
#                                 '2023-01-01 09:00:00']),
#         'price': [100.0, 101.0, 102.0, 105.0, 115.0, 125.0, 130.0, 128.0, 120.0, 110.0],
#         'amount': [1000, 1050, 1100, 1200, 1500, 1800, 2500, 2000, 1800, 1500],
#         'other_feature': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#         'is_peak': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # Peak at index 6 (130.0 price, 2500 amount)
#     }
#     df = pd.DataFrame(data).set_index('date') # Changed 'timestamp' to 'date'
#     return df
# 
# # Fixture for sample data with no peak (kept for completeness of original file)
# @pytest.fixture
# def sample_data_no_peak():
#     """
#     Provides sample DataFrame with a 'date' column as index,
#     and includes 'price' and 'amount'.
#     Designed to NOT trigger any peaks.
#     """
#     data = {
#         'date': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00',
#                                 '2023-01-01 03:00:00', '2023-01-01 04:00:00', '2023-01-01 05:00:00',
#                                 '2023-01-01 06:00:00', '2023-01-01 07:00:00', '2023-01-01 08:00:00',
#                                 '2023-01-01 09:00:00']),
#         'price': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
#         'amount': [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090]
#     }
#     df = pd.DataFrame(data).set_index('date') # Changed 'timestamp' to 'date'
#     return df
# 
# # Test cases for CharacteristicParams
# 
# def test_characteristic_params_initialization():
#     """
#     Tests basic initialization of CharacteristicParams with name and description.
#     """
#     params = CharacteristicParams(name="TestComponent", description="A test component.")
#     assert params.name == "TestComponent"
#     assert params.description == "A test component."
#     assert params.additional_params == {}
# 
# def test_characteristic_params_with_additional_params():
#     """
#     Tests initialization of CharacteristicParams with explicit additional parameters.
#     """
#     params = CharacteristicParams(name="TestComponent", additional_params={"key": "value"})
#     assert params.additional_params == {"key": "value"}
# 
# def test_characteristic_params_to_dict():
#     """
#     Tests conversion of CharacteristicParams instance to a dictionary.
#     """
#     params = CharacteristicParams(name="DictTest", description="Convert to dict", additional_params={"rate": 10.5})
#     expected_dict = {
#         'name': 'DictTest',
#         'description': 'Convert to dict',
#         'additional_params': {'rate': 10.5}
#     }
#     assert params.to_dict() == expected_dict
# 
# def test_characteristic_params_from_dict():
#     """
#     Tests creating a CharacteristicParams instance from a dictionary.
#     Verifies that extra keys in the input dict are correctly merged into additional_params.
#     """
#     data = {
#         'name': 'FromDictTest',
#         'description': 'Create from dict',
#         'additional_params': {'id': 123},
#         'extra_key_not_in_dataclass': 'should_be_merged_into_additional_params'
#     }
#     params = CharacteristicParams.from_dict(data)
#     assert params.name == "FromDictTest"
#     assert params.description == "Create from dict"
#     # This behavior (merging extra keys into additional_params) is a design choice
#     # and this test correctly asserts it.
#     assert params.additional_params == {'id': 123, 'extra_key_not_in_dataclass': 'should_be_merged_into_additional_params'}
# 
# 
# # Test cases for PumpCharacteristicParams
# 
# def test_pump_characteristic_params_initialization():
#     """
#     Tests basic initialization of PumpCharacteristicParams with default values.
#     """
#     pump = PumpCharacteristicParams(name="TestPump")
#     assert pump.name == "TestPump"
#     assert pump.max_flow_rate_lps == 100.0 # Default value
#     assert pump.efficiency_percentage == 85.0 # Default value
#     assert pump.flow_rate_normal_range == [30.0, 90.0]
# 
# def test_pump_characteristic_params_custom_values():
#     """
#     Tests initialization of PumpCharacteristicParams with custom values for all attributes.
#     """
#     pump = PumpCharacteristicParams(
#         name="CustomPump",
#         max_flow_rate_lps=200.0,
#         nominal_pressure_bar=10.0,
#         power_rating_kw=50.0,
#         flow_rate_normal_range=[100.0, 180.0]
#     )
#     assert pump.name == "CustomPump"
#     assert pump.max_flow_rate_lps == 200.0
#     assert pump.nominal_pressure_bar == 10.0
#     assert pump.power_rating_kw == 50.0
#     assert pump.flow_rate_normal_range == [100.0, 180.0]
# 
# def test_pump_characteristic_params_post_init_validation_warnings(caplog):
#     """
#     Tests post-initialization validation logic for PumpCharacteristicParams,
#     verifying warnings and errors are logged correctly for invalid inputs.
#     """
#     # Test efficiency out of range (WARNING)
#     with caplog.at_level('WARNING'):
#         pump = PumpCharacteristicParams(name="BadEfficiencyPump", efficiency_percentage=105.0)
#         assert "Efficiency percentage 105.0 is out of typical range (0-100)" in caplog.text
# 
#     caplog.clear() # Clear logs for next test
# 
#     # Test min_flow_rate_lps >= max_flow_rate_lps (ERROR)
#     with caplog.at_level('ERROR'):
#         pump = PumpCharacteristicParams(name="BadFlowRangePump", min_flow_rate_lps=100.0, max_flow_rate_lps=90.0)
#         assert "Min flow rate (100.0) should be less than max flow rate (90.0)" in caplog.text
# 
#     caplog.clear() # Clear logs for next test
# 
#     # Test nominal_flow_rate_lps outside min/max range (WARNING)
#     with caplog.at_level('WARNING'):
#         pump = PumpCharacteristicParams(name="BadNominalFlowPump", min_flow_rate_lps=10.0, max_flow_rate_lps=20.0, nominal_flow_rate_lps=5.0)
#         assert "Nominal flow rate (5.0) is outside min/max flow range." in caplog.text
# 
#     caplog.clear() # Clear logs for next test
#     
#     # Test invalid normal range (WARNING)
#     with caplog.at_level('WARNING'):
#         pump = PumpCharacteristicParams(name="BadNormalRange", flow_rate_normal_range=[90.0, 30.0])
#         assert "Invalid flow_rate_normal_range: [90.0, 30.0]. Resetting to nominal range." in caplog.text
# 
# 
# def test_pump_characteristic_params_get_operating_limits():
#     """
#     Tests the get_operating_limits method of PumpCharacteristicParams.
#     """
#     pump = PumpCharacteristicParams(name="LimitsTest", max_flow_rate_lps=120.0, nominal_flow_rate_lps=80.0)
#     limits = pump.get_operating_limits()
#     assert limits['max_flow_rate_lps'] == 120.0
#     assert limits['nominal_flow_rate_lps'] == 80.0
#     assert 'efficiency_percentage' in limits
# 
# def test_pump_characteristic_params_calculate_power_consumption_nominal():
#     """
#     Tests power consumption calculation at nominal operating points.
#     Uses pytest.approx for floating-point comparisons.
#     """
#     pump = PumpCharacteristicParams(nominal_flow_rate_lps=70.0, nominal_pressure_bar=5.0, power_rating_kw=15.0, efficiency_percentage=85.0)
#     # At nominal flow and pressure, the calculated power should be very close to the power_rating_kw.
#     power = pump.calculate_power_consumption(70.0, 5.0)
#     assert power == pytest.approx(15.0, rel=0.05) # Allow 5% relative difference for simplified model calculation
# 
# def test_pump_characteristic_params_calculate_power_consumption_outside_range(caplog):
#     """
#     Tests power consumption calculation when operating outside nominal range,
#     verifying warnings are logged and power is penalized.
#     """
#     pump = PumpCharacteristicParams(nominal_flow_rate_lps=70.0, nominal_pressure_bar=5.0, power_rating_kw=15.0)
#     
#     with caplog.at_level('WARNING'):
#         # Flow rate too high
#         power = pump.calculate_power_consumption(150.0, 5.0)
#         assert "outside nominal pump operating envelope" in caplog.text
#         assert power > 15.0 # Should be penalized (higher than nominal rated power)
#     caplog.clear()
# 
#     with caplog.at_level('WARNING'):
#         # Pressure too low
#         power = pump.calculate_power_consumption(70.0, 0.5)
#         assert "outside nominal pump operating envelope" in caplog.text
#         assert power > 15.0 # Should be penalized (higher than nominal rated power)
#     caplog.clear()
# 
# def test_pump_characteristic_params_from_dict():
#     """
#     Tests creating a PumpCharacteristicParams instance from a dictionary.
#     Verifies custom values are set, default values are applied for missing keys,
#     and extra keys are captured in additional_params.
#     """
#     data = {
#         'name': 'DictPump',
#         'max_flow_rate_lps': 180.0,
#         'nominal_pressure_bar': 8.0,
#         'maintenance_interval_hours': 1000,
#         'extra_key': 'some_value'
#     }
#     pump = PumpCharacteristicParams.from_dict(data)
#     assert pump.name == "DictPump"
#     assert pump.max_flow_rate_lps == 180.0
#     assert pump.nominal_pressure_bar == 8.0
#     assert pump.maintenance_interval_hours == 1000
#     assert pump.additional_params == {'extra_key': 'some_value'} # Ensure extra keys are captured in additional_params
#     
#     # Check default values are still applied if not provided in dict
#     assert pump.efficiency_percentage == 85.0
# 
# 
# # Test cases for RuleBasedPeakEstimator (from original user input, kept for context)
# 
# def test_estimator_initialization(base_rule_config):
#     """
#     Tests if the RuleBasedPeakEstimator initializes correctly
#     and inherits from BasePeakEstimator.
#     """
#     estimator = RuleBasedPeakEstimator(base_rule_config)
#     assert isinstance(estimator, BasePeakEstimator)
#     assert estimator.price_rise_threshold == 0.10
#     assert estimator.lookback_window == 3
#     assert not estimator.is_fitted # is_fitted is an attribute, not a method
# 
# def test_train_method_sets_fitted_status(base_rule_config, sample_data_with_peak):
#     """
#     Tests that the train method correctly sets the estimator's fitted status.
#     For rule-based, train usually just validates and sets internal state.
#     """
#     estimator = RuleBasedPeakEstimator(base_rule_config)
#     # Rule-based estimators typically don't 'learn' from labels in train,
#     # but the method signature requires it.
#     labels = sample_data_with_peak['is_peak']
#     # Pass features only (drop 'is_peak' from X)
#     estimator.train(sample_data_with_peak.drop(columns=['is_peak']), labels)
#     assert estimator.is_fitted # is_fitted is an attribute
# 
# def test_predict_with_peak_data(base_rule_config, sample_data_with_peak):
#     """
#     Tests prediction on data containing a defined peak.
#     Verifies the rule-based logic correctly identifies the peak.
#     """
#     estimator = RuleBasedPeakEstimator(base_rule_config)
#     # For rule-based, train mostly confirms setup; rules are inherent in config.
#     # The 'is_peak' column is used for comparison, not for training the rules themselves.
#     estimator.train(sample_data_with_peak, sample_data_with_peak['is_peak'])
#     
#     # Drop the 'is_peak' column from the data used for prediction, as it's the label
#     predictions = estimator.predict(sample_data_with_peak.drop(columns=['is_peak']))
#     
#     assert isinstance(predictions, pd.Series)
#     assert len(predictions) == len(sample_data_with_peak)
#     
#     # Expected peak at index with '2023-01-01 06:00:00' based on fixture data and rules
#     # Price at index 6 (130) vs min in [105, 115, 125] (105) -> (130-105)/105 = 0.238 > 0.10 (rise threshold)
#     # Price at index 7 (128) vs current 130 -> 128 is NOT < 130 * (1-0.03) = 126.1
#     # Price at index 8 (120) vs current 130 -> 120 IS < 130 * (1-0.03) = 126.1 (drop threshold met by index 8, within 2 timesteps)
#     # amount at index 6 (2500) vs mean in [1200, 1500, 1800] (1500) -> (2500-1500)/1500 = 0.66 > 0.20 (amount threshold)
#     
#     # Create expected predictions Series with the correct index (date)
#     expected_predictions = pd.Series([0,0,0,0,0,0,1,0,0,0], index=sample_data_with_peak.index, dtype=int)
#     pd.testing.assert_series_equal(predictions, expected_predictions, check_names=False) # check_names=False for Series comparison if name differs
# 
# def test_predict_with_no_peak_data(base_rule_config, sample_data_no_peak):
#     """
#     Tests prediction on data expected to have no peaks.
#     Verifies no peaks are detected.
#     """
#     estimator = RuleBasedPeakEstimator(base_rule_config)
#     # Labels are not strictly used by rule-based train, but required by method signature
#     estimator.train(sample_data_no_peak, pd.Series([0]*len(sample_data_no_peak), index=sample_data_no_peak.index))
#     
#     predictions = estimator.predict(sample_data_no_peak)
#     
#     assert isinstance(predictions, pd.Series)
#     assert len(predictions) == len(sample_data_no_peak)
#     assert predictions.sum() == 0 # No peaks should be detected
# 
# def test_predict_requires_columns(base_rule_config, sample_data_with_peak):
#     """
#     Tests that prediction raises an error if required columns are missing.
#     """
#     estimator = RuleBasedPeakEstimator(base_rule_config)
#     estimator.train(sample_data_with_peak, sample_data_with_peak['is_peak'])
#     
#     # Test with missing price column
#     with pytest.raises(ValueError, match="Missing required columns"):
#         estimator.predict(sample_data_with_peak.drop(columns=['price', 'is_peak'])) # Drop is_peak too, as it's a label
#     
#     # Test with missing amount column
#     with pytest.raises(ValueError, match="Missing required columns"):
#         estimator.predict(sample_data_with_peak.drop(columns=['amount', 'is_peak'])) # Drop is_peak too
# 
# def test_save_and_load_model(base_rule_config, temp_test_models_dir, sample_data_with_peak):
#     """
#     Tests saving and loading of the rule-based estimator's configuration.
#     Ensures parameters are preserved and predictions are consistent.
#     """
#     estimator_original = RuleBasedPeakEstimator(base_rule_config)
#     estimator_original.train(sample_data_with_peak, sample_data_with_peak['is_peak'])
#     
#     save_path = temp_test_models_dir # Path to the specific estimator's model dir
#     estimator_original.save_model(str(save_path))
#     
#     # Rule-based estimator saves its config (parameters) in 'config.pkl'
#     assert Path(save_path).joinpath("config.pkl").exists() 
#     
#     estimator_loaded = RuleBasedPeakEstimator({}) # Initialize with dummy config, it will be loaded
#     estimator_loaded.load_model(str(save_path))
#     
#     assert estimator_loaded.is_fitted # is_fitted is an attribute
#     assert estimator_loaded.price_rise_threshold == estimator_original.price_rise_threshold
#     assert estimator_loaded.lookback_window == estimator_original.lookback_window
#     assert estimator_loaded.price_drop_threshold == estimator_original.price_drop_threshold
#     assert estimator_loaded.min_amount_increase == estimator_original.min_amount_increase
#     assert estimator_loaded.check_next_timesteps == estimator_original.check_next_timesteps
# 
#     # Ensure loaded model predicts the same
#     predictions_original = estimator_original.predict(sample_data_with_peak.drop(columns=['is_peak']))
#     predictions_loaded = estimator_loaded.predict(sample_data_with_peak.drop(columns=['is_peak']))
#     pd.testing.assert_series_equal(predictions_original, predictions_loaded, check_names=False)
# 
# def test_load_non_existent_model(base_rule_config, temp_test_models_dir):
#     """
#     Tests that loading a non-existent model raises a FileNotFoundError.
#     """
#     estimator = RuleBasedPeakEstimator(base_rule_config)
#     non_existent_path = temp_test_models_dir / "non_existent_model"
#     with pytest.raises(FileNotFoundError, match="Rule-based estimator config not found"):
#         estimator.load_model(str(non_existent_path))
# 
# def test_multi_sequence_prediction(base_rule_config):
#     """
#     Tests prediction on a multi-index DataFrame (multiple sequences).
#     Ensures the estimator handles sequences independently.
#     """
#     estimator = RuleBasedPeakEstimator(base_rule_config)
#     
#     # Create multi-sequence data
#     # Sequence 0: No peak
#     data0 = { # Renamed to data0 for clarity
#         'date': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00',
#                                 '2023-01-01 03:00:00', '2023-01-01 04:00:00']),
#         'price': [100.0, 101.0, 102.0, 103.0, 104.0],
#         'amount': [1000, 1050, 1100, 1200, 1300],
#         'is_peak': [0, 0, 0, 0, 0]
#     }
#     df0 = pd.DataFrame(data0).set_index('date') # Changed 'timestamp' to 'date'
# 
#     # Sequence 1: Data designed to trigger a peak at index 3 of this sequence
#     data1 = { # Renamed to data1 for clarity
#         'date': pd.to_datetime(['2023-01-01 05:00:00', '2023-01-01 06:00:00', '2023-01-01 07:00:00',
#                                 '2023-01-01 08:00:00', '2023-01-01 09:00:00']),
#         'price': [105.0, 115.0, 125.0, 130.0, 120.0], # Peak at 130, then a sufficient drop to 120
#         'amount': [1400, 1800, 2500, 2800, 1800],    # amount increased sufficiently for min_amount_increase=0.20
#         'is_peak': [0, 0, 0, 1, 0] # Peak at index 3 of this sequence
#     }
#     df1 = pd.DataFrame(data1).set_index('date') # Changed 'timestamp' to 'date'
# 
#     # Combine into a multi-index DataFrame
#     df_multi = pd.concat([df0, df1], keys=[0, 1], names=['sequence_id', 'date']) # Changed 'timestamp' to 'date'
#     
#     # Train the estimator (rules are configured, not learned)
#     # Drop 'is_peak' from X_train as it's a label
#     estimator.train(df_multi.drop(columns=['is_peak']), df_multi['is_peak'])
# 
#     # Drop 'is_peak' from data used for prediction
#     predictions = estimator.predict(df_multi.drop(columns=['is_peak']))
#     
#     assert isinstance(predictions, pd.Series)
#     assert len(predictions) == len(df_multi)
#     
#     # For Sequence 0, all 0s are expected
#     expected_predictions_seq0 = pd.Series([0,0,0,0,0], index=df0.index, dtype=int)
# 
#     # For Sequence 1, expected peak based on the designed data and base_rule_config
#     # price 130 (idx 3 of seq) vs min in [105, 115, 125] (105) -> (130-105)/105 = 0.238 > 0.10 (Passes)
#     # amount 2800 (idx 3 of seq) vs mean in [1400, 1800, 2500] (1900) -> (2800-1900)/1900 = 0.47 > 0.20 (Passes)
#     # price 120 (idx 4 of seq) vs 130 * (1-0.03) = 126.1 -> 120 < 126.1 (Passes, drop within 1 timestep)
#     expected_predictions_seq1 = pd.Series([0,0,0,1,0], index=df1.index, dtype=int)
#     
#     combined_expected_predictions = pd.concat([
#         expected_predictions_seq0,
#         expected_predictions_seq1
#     ], keys=[0,1], names=['sequence_id', 'date']) # Changed 'timestamp' to 'date'
# 
#     pd.testing.assert_series_equal(predictions, combined_expected_predictions, check_names=False)
