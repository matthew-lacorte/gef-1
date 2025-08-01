"""
Foundational constants and model parameters for the GEF core physics framework.

Exports:
    - ConstantInfo: Pydantic model for constant metadata.
    - CONSTANTS: List of ConstantInfo entries.
    - CONSTANTS_DICT: Dict mapping name to ConstantInfo.
    - constants_info: Utility to extract/export all constants and their documentation.
    - b_0, m, c, h_planck, alpha_em, lambda_G, h_hook, m_PlanckParticle, eV: Symbolic sympy constants.
"""

__all__ = [
    "ConstantInfo",
    "CONSTANTS",
    "CONSTANTS_DICT",
    "constants_info",
    "b_0", "m", "c", "h_planck", "alpha_em",
    "lambda_G", "h_hook", "m_PlanckParticle", "eV",
    "__version__",
]

import sympy as sp
import pint
import numpy as np
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

# --- Symbolic Declarations ---
# GEF Model Parameters
b_0 = sp.Symbol('b_0', real=True, positive=True)
m = sp.Symbol('m', real=True, positive=True)
m_PlanckParticle = sp.Symbol('m_PlanckParticle', real=True, positive=True)
lambda_G = sp.Symbol('lambda_G', real=True, positive=True)
h_hook = sp.Symbol('h', real=True, positive=True)

# Fundamental & Conversion Constants
c = sp.Symbol('c', real=True, positive=True)
h_planck = sp.Symbol('h_planck', real=True, positive=True)
eV = sp.Symbol('eV', real=True, positive=True)
alpha_em = sp.Symbol('alpha_em', real=True, positive=True)


# --- Constant Definitions ---
# The single source of truth for all framework constants.
CONSTANTS: List[ConstantInfo] = [
    # --- GEF Model Parameters ---
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
        name="m_PlanckParticle",
        symbol=m_PlanckParticle,
        value=200.0,
        units="MeV",
        description="GEF canonical Planck-particle rest-energy scale.",
        category="model parameter",
        sidecar_path="physics/constants/m_PlanckParticle.md"
    ),
    ConstantInfo(
        name="lambda_G",
        symbol=lambda_G,
        value=157.91 / 137.036, # ~1.1523
        units="dimensionless",
        description="GEF self-coupling constant, derived from the fine-structure constant under the Unity Hypothesis.",
        category="model parameter",
        sidecar_path="physics/constants/lambda_G.md"
    ),
    ConstantInfo(
        name="h_hook",
        symbol=h_hook,
        value=None, # Value to be determined
        units="joule**0.5 * meter", # Based on Lagrangian term analysis
        description="Hook Coupling Constant, defining the energy cost of topological complexity.",
        category="model parameter",
        sidecar_path="physics/constants/h_hook.md"
    ),

    # --- Fundamental Constants ---
    ConstantInfo(
        name="c",
        symbol=c,
        value=299_792_458,
        units="meter/second",
        description="Speed of light in vacuum (exact by definition).",
        category="fundamental",
        sidecar_path="physics/constants/c.md"
    ),
    ConstantInfo(
        name="h_planck",
        symbol=h_planck,
        value=6.626_070_15e-34,
        units="joule*second",
        description="The Planck constant (exact by definition).",
        category="fundamental",
        sidecar_path="physics/constants/planck.md"
    ),
    ConstantInfo(
        name="alpha_em",
        symbol=alpha_em,
        value=1 / 137.035999084, # CODATA 2018
        units="dimensionless",
        description="Fine-structure constant, the coupling constant for the electromagnetic force.",
        category="fundamental",
        sidecar_path="physics/constants/alpha_em.md"
    ),
    ConstantInfo(
        name="GEF_LOOP_FACTOR",
        symbol=None, # Not a symbolic constant
        value=16 * np.pi**2, # ~157.91
        units="dimensionless",
        description="Geometric loop factor (16π²) used in the derivation of induced electromagnetic coupling.",
        category="derived constant",
        sidecar_path="physics/constants/gef_loop_factor.md"
    ),
    
    # --- Conversion Factors ---
    ConstantInfo(
        name="eV",
        symbol=eV,
        value=1.602_176_634e-19,
        units="joule",
        description="Electron volt energy unit in terms of joules (exact by definition).",
        category="conversion",
        sidecar_path="physics/constants/electron_volt.md"
    ),
]

# --- Dictionary and Utility Function ---
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