"""
Foundational constants and model parameters for the GEF core physics framework.

Exports:
    - ConstantInfo: Pydantic model for constant metadata.
    - CONSTANTS: List of ConstantInfo entries.
    - CONSTANTS_DICT: Dict mapping name to ConstantInfo.
    - constants_info: Utility to extract/export all constants and their documentation.
    - b_0, m, c, m_0: Symbolic sympy constants for analytic derivations.

Symbolic constants use sympy for analytic derivations. Numeric values, descriptions, and units are managed via Pydantic for safety and automated documentation.

Auto-generated doc sidecars can be built using `constants_info()`.
"""

__all__ = [
    "ConstantInfo",
    "CONSTANTS",
    "CONSTANTS_DICT",
    "constants_info",
    "b_0", "m", "c", "m_0"
    "__version__",
]

import sympy as sp
import pint
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from gef.core.validators import asdict, positive_value
from gef.core.enums import ConstantCategory

sidecar_path = f"vault/50-sidecars/{__name__}/"

ureg = pint.UnitRegistry()

class ConstantInfo(BaseModel):
    """
    Metadata and value(s) for a single GEF framework constant.
    """
    
    name: str = Field(..., description="Symbolic name of the constant.")
    symbol: Optional[sp.Basic] = Field(None, description="SymPy symbol for analytic calculations.")
    value: Optional[float] = Field(None, description="Default or reference numeric value.")
    units: str = Field("", description="Units of the constant.")
        # q = ureg.Quantity(1, "meter/second")
        # print(q.to_compact())  # e.g., 1 m/s
    description: str = Field("", description="Brief description for docs.")
    sidecar_path: Optional[str] = Field(None, description="Relative path to this constant’s Obsidian .md sidecar file.")
    category: Optional[ConstantCategory] = Field(None, description="Grouping/tag/category for filtering (optional).")

    class Config:
        frozen = True
        arbitrary_types_allowed = True

    @field_validator("value")
    def value_must_be_positive(cls, v):
        return positive_value(cls, v)

    def __str__(self):
        val = f"{self.value}" if self.value is not None else "Symbolic"
        cat = f" [{self.category}]" if self.category else ""
        return f"{self.name} ({self.symbol}): {val} [{self.units}]{cat} — {self.description}"

    @property
    def symbol_name(self):
        return str(self.symbol) if self.symbol else self.name

    @property
    def quantity(self):
        if self.value is not None and self.units:
            return self.value * ureg(self.units)
        return None

# Symbolic declarations
b_0 = sp.Symbol('b_0', real=True, positive=True)
m   = sp.Symbol('m', real=True, positive=True)
c   = sp.Symbol('c', real=True, positive=True)
m_0 = sp.Symbol('m_0', real=True, positive=True)
electron_volt = sp.Symbol('electron_volt', real=True, positive=True)
planck = sp.Symbol('planck', real=True, positive=True)  # Planck constant

CONSTANTS: List[ConstantInfo] = [
    ConstantInfo(
        name="b_0",
        symbol=b_0,
        value=1.0,
        units="dimensionless",
        description="Anisotropy parameter controlling metric deformation. If b₀=1, reduces to isotropic case.",
        category="model parameter",
        sidecar_path="physics/constants/b_0.md"
    ),
    ConstantInfo(
        name="m",
        symbol=m,
        value=1.0,
        units="dimensionless",
        description="Euclidean mass parameter in the foundational GEF model.",
        category="model parameter",
        sidecar_path="physics/constants/m.md"
    ),
    ConstantInfo(
        name="c",
        symbol=c,
        value=None,
        units="speed",
        description="Emergent speed of light derived from the model (not fundamental).",
        category="derived constant",
        sidecar_path="physics/constants/c.md"
    ),
    ConstantInfo(
        name="electron_volt",
        symbol=electron_volt,#J?
        value=1.602_176_634e-19, 
        units="energy",
        description="Emergent electron volt derived from the model (not fundamental).",
        category="derived constant",
        sidecar_path="physics/constants/electron_volt.md"
    ),
    ConstantInfo(
        name="m_0",
        symbol=m_0,
        value=1.602_176_634e-19,
        units="mass",
        description="Emergent rest mass derived from the model (not fundamental).",
        category="derived constant",
        sidecar_path="physics/constants/m_0.md"
    ),
    ConstantInfo(
        name="planck",
        symbol=planck,
        value=6.626_070_15e-34,  # CODATA 2019 exact value
        units="J·s",
        description="Planck constant (h) - fundamental constant of quantum mechanics.",
        category="derived constant",
        sidecar_path="physics/constants/planck.md"
    ),
]

CONSTANTS_DICT: Dict[str, ConstantInfo] = {const.name: const for const in CONSTANTS}

def constants_info(as_dict: bool = False):
    """
    Returns all constants and metadata for documentation/automation.
    Args:
        as_dict (bool): If True, returns a dict keyed by name.
    Returns:
        List[dict] or Dict[str, dict]: Metadata of all constants.
    """
    if as_dict:
        return {const.name: asdict(const) for const in CONSTANTS}
    else:
        return [asdict(const) for const in CONSTANTS]
