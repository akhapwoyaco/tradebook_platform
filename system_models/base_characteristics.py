from dataclasses import dataclass, field, asdict # Import asdict
from typing import Dict, Any, List, Optional
import json
import inspect # Moved import to the top
from loguru import logger # Import logger for consistent logging

@dataclass
class CharacteristicParams:
    """
    Base class for defining a set of characteristic parameters for any system component.
    This provides a structured way to hold immutable configuration details.
    """
    name: str = "BaseSystemComponent"
    description: str = "General system component characteristics."
    # A dictionary for any additional, flexible parameters not strictly defined
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the dataclass instance to a dictionary.
        Uses dataclasses.asdict for robust conversion, handling nested dataclasses.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a dataclass instance from a dictionary."""
        
        # Get the fields defined in the dataclass's constructor
        sig = inspect.signature(cls)
        constructor_params = {name for name in sig.parameters if name != 'self'}

        init_params = {}
        remaining_params = {}

        # Separate parameters into those that match direct dataclass fields
        # and those that are 'additional'
        for k, v in data.items():
            if k in constructor_params:
                init_params[k] = v
            else:
                remaining_params[k] = v
        
        # If 'additional_params' is a field in the class, ensure it captures all remaining_params.
        # This simplifies the logic to always collect unmapped keys into additional_params.
        if 'additional_params' in constructor_params:
            if 'additional_params' not in init_params:
                # If additional_params was not explicitly provided, initialize it
                init_params['additional_params'] = {}
            # Merge any remaining_params into additional_params
            init_params['additional_params'].update(remaining_params)
        else:
            if remaining_params:
                # If there's no 'additional_params' field but extra data, warn the user
                logger.warning(
                    f"Extra parameters {list(remaining_params.keys())} provided for {cls.__name__} "
                    f"but no 'additional_params' field to store them. These will be ignored."
                )

        return cls(**init_params)

    def __str__(self) -> str:
        """Provides a string representation of the characteristics."""
        return f"{self.name} Characteristics:\n{json.dumps(self.to_dict(), indent=2)}"