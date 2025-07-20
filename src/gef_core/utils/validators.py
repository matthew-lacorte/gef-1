# utils/validators.py

import pydantic

def asdict(model):
    """Returns dict for Pydantic v1/v2 compatibility."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

# Field validator: value must be positive if present (for Pydantic v2)
from pydantic import field_validator

def positive_value(cls, v):
    if v is not None and v < 0:
        raise ValueError("Value must be positive")
    return v

# Example for additional validators (units, sidecar path, etc.)
def valid_units(cls, v):
    # Add your logic here (for example, check against allowed units)
    return v
