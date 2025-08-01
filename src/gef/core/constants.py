"""
Foundational constants and model parameters for the GEF core physics framework.

This file serves as the Single Source of Truth (SSoT) for the numerical values
and symbolic representations of all constants used in GEF simulations and derivations.

Version: 1.1.0 - Standardized on hbar, clarified mass relationships.
"""

__version__ = "1.1.0"

__all__ = [
    "ConstantInfo",
    "CONSTANTS",
    "CONSTANTS_DICT",
    "constants_info",
    "kappa", "b_0", "m_euc", "m_0", "c", "hbar", "alpha_em",
    "lambda_G", "g_sq", "P_env", "m_GEF_PP", "eV",
    "__version__",
]

import sympy as sp
import pint
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
# Assuming these are defined in your project's core utilities
# from gef.core.validators import asdict, positive_value
# from gef.core.enums import ConstantCategory

# Placeholder for validators/enums if not yet implemented
def asdict(model): return model.dict()
def positive_value(cls, v):
    if v is not None and v < 0:
        raise ValueError("Value must be non-negative")
    return v
from enum import Enum
class ConstantCategory(str, Enum):
    MODEL_PARAMETER = "model_parameter"
    DERIVED_PARAMETER = "derived_parameter"
    FUNDAMENTAL = "fundamental"
    INTERNAL_FACTOR = "internal_factor"
    CONVERSION = "conversion"
# End Placeholder

ureg = pint.UnitRegistry()

class ConstantInfo(BaseModel):
    """Metadata and value(s) for a single GEF framework constant."""
    
    name: str = Field(..., description="Canonical name of the constant (used as key).")
    symbol: Optional[sp.Basic] = Field(None, description="SymPy symbol for analytic calculations.")
    value: Optional[float] = Field(None, description="Default or reference numeric value.")
    units: str = Field("dimensionless", description="Units of the constant.")
    description: str = Field("", description="Brief description for docs and tooltips.")
    sidecar_path: Optional[str] = Field(None, description="Filename of this constant’s Obsidian .md sidecar file.")
    category: ConstantCategory = Field(..., description="The primary category of the constant.")
    relation: Optional[str] = Field(None, description="Symbolic relationship to other constants.")

    class Config:
        frozen = True
        arbitrary_types_allowed = True

    _validate_value_positive = field_validator("value")(positive_value)

    def __str__(self):
        val_str = f"{self.value}" if self.value is not None else "Symbolic"
        unit_str = f" [{self.units}]" if self.units and self.units != "dimensionless" else ""
        return f"{self.name} ({self.symbol_name}): {val_str}{unit_str} — {self.description}"

    @property
    def symbol_name(self) -> str:
        """Returns the string representation of the SymPy symbol."""
        return str(self.symbol) if self.symbol else self.name

    @property
    def quantity(self) -> Optional[pint.Quantity]:
        """Returns the value as a pint.Quantity object for dimensional analysis."""
        if self.value is not None and self.units and self.units != "dimensionless":
            return self.value * ureg(self.units)
        return None

# --- Symbolic Declarations ---
kappa = sp.Symbol('kappa', real=True, positive=True)
b_0 = sp.Symbol('b_0', real=True, positive=True)
m_euc = sp.Symbol('m_euc', real=True, positive=True)
m_0 = sp.Symbol('m_0', real=True, positive=True)
m_GEF_PP = sp.Symbol('m_GEF_PP', real=True, positive=True)
lambda_G = sp.Symbol('lambda_G', real=True, positive=True)
g_sq = sp.Symbol('g_sq', real=True, positive=True) # Using a simpler variable name
P_env = sp.Symbol('P_env', real=True, positive=True)
c = sp.Symbol('c', real=True, positive=True)
hbar = sp.Symbol('hbar', real=True, positive=True)
eV = sp.Symbol('eV', real=True, positive=True)
alpha_em = sp.Symbol('alpha_em', real=True, positive=True)

# --- Define relationships between symbols ---
RELATIONS = {
    'b_0': sp.sqrt(1 + kappa**2),
    'm_euc': m_0 * c,
    'c': 1 / b_0
}

# --- Define numerical values for base constants ---
_kappa_val = 1e-61
_alpha_em_val = 1 / 137.035999084
_gef_loop_factor = 16 * np.pi**2

# --- The Single Source of Truth for all framework constants. ---
CONSTANTS: List[ConstantInfo] = [
    # === GEF Core Model Parameters ===
    ConstantInfo(
        name="kappa",
        symbol=kappa,
        value=_kappa_val,
        description="Dimensionless cosmic anisotropy flow, the source of the arrow of time.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_kappa.md"
    ),
    ConstantInfo(
        name="lambda_G",
        symbol=lambda_G,
        value=_gef_loop_factor * _alpha_em_val,
        description="GEF self-coupling constant, derived from the fine-structure constant.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_lambda_G.md"
    ),
    ConstantInfo(
        name="m_GEF_PP",
        symbol=m_GEF_PP,
        value=220.0,
        units="MeV",
        description="The rest energy scale of the GEF Planck Particle, the source of kappa.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_m_GEF_PP.md"
    ),
    ConstantInfo(
        name="g_sq",
        symbol=g_sq, # Using the simpler Python variable name for the symbol
        description="Dimensionless internal rigidity constant, unique to each particle type.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_g_sq.md"
    ),
    ConstantInfo(
        name="P_env",
        symbol=P_env,
        description="Dimensionless environmental pressure from the 'Sea of Seas'.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_P_env.md"
    ),

    # === GEF Derived Kinematic Parameters ===
    ConstantInfo(
        name="b_0",
        symbol=b_0,
        value=float(RELATIONS['b_0'].subs(kappa, _kappa_val)), # Directly calculate from kappa
        description="Anisotropy parameter controlling metric deformation.",
        category=ConstantCategory.DERIVED_PARAMETER,
        relation="sqrt(1 + kappa^2)",
        sidecar_path="sidecar_constant_b_0.md"
    ),
    ConstantInfo(
        name="m_euc",
        symbol=m_euc,
        description="Euclidean mass parameter in the GEF action. Has units of Energy.",
        units="MeV", # Clarifying units
        category=ConstantCategory.DERIVED_PARAMETER,
        relation="m_0 * c",
        sidecar_path="sidecar_constant_m_euc.md"
    ),
    ConstantInfo(
        name="m_0",
        symbol=m_0,
        description="Physical rest mass (energy equivalent) of a particle in emergent Minkowski spacetime.",
        units="MeV",
        category=ConstantCategory.DERIVED_PARAMETER,
        sidecar_path="sidecar_constant_m_0.md"
    ),

    # === Fundamental Constants (as measured) ===
    ConstantInfo(
        name="c",
        symbol=c,
        value=299_792_458.0,
        units="m/s",
        description="Speed of light in vacuum (exact by definition).",
        category=ConstantCategory.FUNDAMENTAL,
        sidecar_path="sidecar_constant_c.md"
    ),
    ConstantInfo(
        name="hbar",
        symbol=hbar,
        value=1.054571817e-34,
        units="J*s",
        description="The reduced Planck constant (h/2π). More fundamental in QFT.",
        category=ConstantCategory.FUNDAMENTAL,
        sidecar_path="sidecar_constant_hbar.md"
    ),
    ConstantInfo(
        name="alpha_em",
        symbol=alpha_em,
        value=_alpha_em_val,
        description="Fine-structure constant, the electromagnetic coupling.",
        category=ConstantCategory.FUNDAMENTAL,
        sidecar_path="sidecar_constant_alpha_em.md"
    ),
    
    # === GEF Internal & Conversion Factors ===
    ConstantInfo(
        name="GEF_LOOP_FACTOR",
        value=_gef_loop_factor,
        description="Geometric loop factor (16*pi^2) used in induced EM coupling derivation.",
        category=ConstantCategory.INTERNAL_FACTOR,
        sidecar_path="sidecar_constant_gef_loop_factor.md"
    ),
    ConstantInfo(
        name="eV",
        symbol=eV,
        value=1.602_176_634e-19,
        units="J",
        description="Electron volt energy unit in terms of joules (exact by definition).",
        category=ConstantCategory.CONVERSION,
        sidecar_path="sidecar_constant_electron_volt.md"
    ),
]

# --- Dictionary and Utility Function ---
CONSTANTS_DICT: Dict[str, ConstantInfo] = {const.name: const for const in CONSTANTS}

def constants_info(as_dict: bool = False) -> List[Dict] | Dict[str, Dict]:
    """Returns all constants and metadata for documentation/automation."""
    if as_dict:
        return {const.name: asdict(const) for const in CONSTANTS}
    else:
        return [asdict(const) for const in CONSTANTS]