"""
Reusable validators and utilities for GEF framework Pydantic models.

Exports:
    - asdict: Version-agnostic model-to-dict converter (Pydantic v1/v2).
    - positive_value: Field validator (value must be positive or None).
    - valid_units: Placeholder for units validation.
    - ... add more as needed
"""

__all__ = [
    "asdict",
    "positive_value",
    "valid_units",
]

from pydantic import field_validator

def asdict(model):
    """
    Return the dictionary representation of a Pydantic model,
    compatible with both Pydantic v1 and v2.

    Args:
        model (BaseModel): The Pydantic model instance.

    Returns:
        dict: Dictionary representation of the model.
    """
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

def positive_value(cls, v):
    """
    Ensure a field's value is positive or None.

    Args:
        cls: The model class (required by Pydantic validator signature).
        v: The value to validate.

    Returns:
        The validated value, if positive or None.

    Raises:
        ValueError: If the value is not None and less than zero.
    """
    if v is not None and v < 0:
        raise ValueError("Value must be positive")
    return v

def valid_units(cls, v):
    """
    (Optional) Placeholder validator for unit validation.
    For now, allows any unit string; extend as needed.

    Args:
        cls: The model class (required by Pydantic validator signature).
        v: The unit string.

    Returns:
        The validated units string.
    """
    # TODO: Add logic to check units against a whitelist or with pint
    return v

# Add further validators or model validators as your project grows.
