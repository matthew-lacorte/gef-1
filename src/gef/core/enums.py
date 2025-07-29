# gef/core/enums.py

from enum import Enum

class ConstantCategory(str, Enum):
    MODEL = "model parameter"
    DERIVED = "derived constant"
    # Add more as needed

class ProvenanceState(str, Enum):
    RAW = "raw"
    VERIFIED = "verified"
    AUDITED = "audited"
    # etc.

__all__ = [
    "ConstantCategory",
    "ProvenanceState",
]
