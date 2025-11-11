from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
# import json # Removed: not directly used in this class
from loguru import logger

from .base_characteristics import CharacteristicParams

@dataclass
class PumpCharacteristicParams(CharacteristicParams):
    """
    Defines characteristic parameters specific to a pump system.
    Inherits from CharacteristicParams to include general system attributes.
    """
    name: str = "GenericPump"
    description: str = "Characteristics for a general industrial pump."

    # Pump-specific parameters with default values
    max_flow_rate_lps: float = 100.0        # Liters per second (max design flow)
    min_flow_rate_lps: float = 0.0          # Minimum operational flow rate (can be > 0 for some pumps)
    nominal_flow_rate_lps: float = 70.0     # NEW: Optimal operating flow rate
    max_pressure_bar: float = 8.0           # Maximum safe operating pressure
    min_pressure_bar: float = 1.0           # Minimum operational pressure
    nominal_pressure_bar: float = 5.0       # Nominal operating pressure in bar
    power_rating_kw: float = 15.0           # Kilowatts (motor power)
    efficiency_percentage: float = 85.0     # Percentage (0-100) at nominal conditions
    expected_lifespan_hours: int = 20000    # Expected operational lifespan before major overhaul
    maintenance_interval_hours: int = 500   # Recommended interval for routine maintenance

    # Operational ranges for anomaly detection or synthetic data generation
    # These define the 'normal' operating envelope
    flow_rate_normal_range: List[float] = field(default_factory=lambda: [30.0, 90.0])
    pressure_normal_range: List[float] = field(default_factory=lambda: [3.5, 6.5])
    power_normal_range: List[float] = field(default_factory=lambda: [5.0, 13.0])

    # Potential for degradation models or failure modes (conceptual, for future use)
    degradation_factors: Dict[str, float] = field(default_factory=dict) # e.g., {'bearing_wear_rate': 0.001}
    failure_modes: List[str] = field(default_factory=list) # e.g., ['cavitation', 'seal_failure', 'overheating']

    def __post_init__(self):
        """
        Performs basic validation or logging after initialization.
        """
        # Call parent's post_init if it existed, though BaseCharacteristicParams doesn't have one
        
        if not (0 <= self.efficiency_percentage <= 100):
            logger.warning(f"Pump '{self.name}': Efficiency percentage {self.efficiency_percentage} is out of typical range (0-100).")
        if self.min_flow_rate_lps >= self.max_flow_rate_lps:
            logger.error(f"Pump '{self.name}': Min flow rate ({self.min_flow_rate_lps}) should be less than max flow rate ({self.max_flow_rate_lps}).")
        if not (self.min_flow_rate_lps <= self.nominal_flow_rate_lps <= self.max_flow_rate_lps):
            logger.warning(f"Pump '{self.name}': Nominal flow rate ({self.nominal_flow_rate_lps}) is outside min/max flow range.")
        
        # Ensure normal ranges are valid (added missing checks)
        if len(self.flow_rate_normal_range) != 2 or not (self.flow_rate_normal_range[0] <= self.flow_rate_normal_range[1]):
            logger.warning(f"Pump '{self.name}': Invalid flow_rate_normal_range: {self.flow_rate_normal_range}. Resetting to min/max flow rates.")
            self.flow_rate_normal_range = [self.min_flow_rate_lps, self.max_flow_rate_lps]

        if len(self.pressure_normal_range) != 2 or not (self.pressure_normal_range[0] <= self.pressure_normal_range[1]):
            logger.warning(f"Pump '{self.name}': Invalid pressure_normal_range: {self.pressure_normal_range}. Resetting to min/max pressure.")
            self.pressure_normal_range = [self.min_pressure_bar, self.max_pressure_bar] # Assumes min/max pressure are design limits

        if len(self.power_normal_range) != 2 or not (self.power_normal_range[0] <= self.power_normal_range[1]):
            logger.warning(f"Pump '{self.name}': Invalid power_normal_range: {self.power_normal_range}. Resetting to [0, power_rating_kw].")
            self.power_normal_range = [0.0, self.power_rating_kw] # Assuming power can go down to 0, up to rating

    def get_operating_limits(self) -> Dict[str, Any]:
        """Returns a dictionary of key operational limits and nominal values."""
        return {
            "max_flow_rate_lps": self.max_flow_rate_lps,
            "min_flow_rate_lps": self.min_flow_rate_lps,
            "nominal_flow_rate_lps": self.nominal_flow_rate_lps,
            "max_pressure_bar": self.max_pressure_bar,
            "min_pressure_bar": self.min_pressure_bar,
            "nominal_pressure_bar": self.nominal_pressure_bar,
            "power_rating_kw": self.power_rating_kw,
            "efficiency_percentage": self.efficiency_percentage,
            "flow_rate_normal_range": self.flow_rate_normal_range,
            "pressure_normal_range": self.pressure_normal_range,
            "power_normal_range": self.power_normal_range,
        }

    def calculate_power_consumption(self, flow_rate: float, pressure: float) -> float:
        """
        A simplified conceptual method to calculate power consumption.
        In a real scenario, this would involve complex pump curves and hydraulics.
        This function is for illustrative purposes of how characteristics could be used.
        
        Note: This model simplifies the relationship where electrical power consumption
        scales proportionally with the product of flow rate and pressure, relative to
        the pump's nominal operating point and power rating.
        """
        if not (self.min_flow_rate_lps <= flow_rate <= self.max_flow_rate_lps) or \
           not (self.min_pressure_bar <= pressure <= self.max_pressure_bar):
            logger.warning(f"Calculated power for flow {flow_rate} and pressure {pressure} outside nominal pump operating envelope for {self.name}. Returning penalized nominal power.")
            # Return nominal power with a penalty if outside normal range
            return self.power_rating_kw * 1.2 # Exceeding limits might draw more power or indicate issue

        # A very rough approximation assuming electrical power scales with hydraulic power
        # relative to nominal conditions.
        # This simplifies to: Power_actual = Power_rated * (Flow_actual * Pressure_actual) / (Flow_nominal * Pressure_nominal)
        scaled_power = self.power_rating_kw * (flow_rate * pressure) / \
                       (self.nominal_flow_rate_lps * self.nominal_pressure_bar)

        return max(0.1, min(scaled_power, self.power_rating_kw * 1.5)) # Cap at 150% of rating as a crude max