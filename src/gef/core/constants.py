"""
Foundational constants and model parameters for the GEF core physics framework.

This file serves as the Single Source of Truth (SSoT) for the numerical values
and symbolic representations of all constants used in GEF simulations and derivations.

Exports:
    - ConstantInfo: Pydantic model for constant metadata.
    - CONSTANTS: List of ConstantInfo entries.
    - CONSTANTS_DICT: Dict mapping name to ConstantInfo.
    - constants_info: Utility to export all constants and their documentation.
    - All individual SymPy symbols for direct import.
"""

__version__ = "0.1.0"

__all__ = [
    "ConstantInfo",
    "CONSTANTS",
    "CONSTANTS_DICT",
    "constants_info",
    "b_0", "m_euc", "m_0", "c", "h_planck", "alpha_em",
    "lambda_G", "g_sq", "P_env", "m_GEF_PP", "eV", "kappa",
    "__version__",
]

import sympy as sp
import pint
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from gef.core.validators import asdict, positive_value
from gef.core.enums import ConstantCategory

ureg = pint.UnitRegistry()

class ConstantInfo(BaseModel):
    """
    Metadata and value(s) for a single GEF framework constant.
    """
    
    name: str = Field(..., description="Canonical name of the constant (used as key).")
    symbol: Optional[sp.Basic] = Field(None, description="SymPy symbol for analytic calculations.")
    value: Optional[float] = Field(None, description="Default or reference numeric value.")
    units: str = Field("dimensionless", description="Units of the constant.")
    description: str = Field("", description="Brief description for docs and tooltips.")
    sidecar_path: Optional[str] = Field(None, description="Relative path to this constant’s Obsidian .md sidecar file.")
    category: ConstantCategory = Field(..., description="The primary category of the constant.")

    class Config:
        frozen = True
        arbitrary_types_allowed = True

    # Use a pre-validator to ensure value is positive where applicable
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
# GEF Model Parameters
kappa = sp.Symbol('kappa', real=True, positive=True)
b_0 = sp.Symbol('b_0', real=True, positive=True)
m_euc = sp.Symbol('m_euc', real=True, positive=True) # Renamed for clarity
m_0 = sp.Symbol('m_0', real=True, positive=True)
m_GEF_PP = sp.Symbol('m_GEF_PP', real=True, positive=True)
lambda_G = sp.Symbol('lambda_G', real=True, positive=True)
g_sq = sp.Symbol('g^2', real=True, positive=True)
P_env = sp.Symbol('P_env', real=True, positive=True)

# Fundamental & Conversion Constants
c = sp.Symbol('c', real=True, positive=True)
h_planck = sp.Symbol('hbar', real=True, positive=True) # Using hbar is more standard in QFT
eV = sp.Symbol('eV', real=True, positive=True)
alpha_em = sp.Symbol('alpha_em', real=True, positive=True)


# --- Constant Definitions ---
# The single source of truth for all framework constants.

_gef_loop_factor = 16 * np.pi**2
_alpha_em_val = 1 / 137.035999084

CONSTANTS: List[ConstantInfo] = [
    # === GEF Core Model Parameters ===
    ConstantInfo(
        name="kappa",
        symbol=kappa,
        value=1e-61, # Approximate value from Dark Energy density
        # TODO: kappa Value: The value 1e-61 is correct for κ itself, but the b_0 value np.sqrt(1 + 1e-122) implies that you're using kappa**2 in the b_0 formula. This is correct, but it's worth a comment to make the κ vs κ² relationship explicit in the b_0 description.
        description="Dimensionless cosmic anisotropy flow, the source of the arrow of time.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_kappa.md"
    ),
    ConstantInfo(
        name="lambda_G",
        symbol=lambda_G,
        value=_gef_loop_factor * _alpha_em_val, # ~1.1523
        description="GEF self-coupling constant, derived from the fine-structure constant.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_lambda_G.md"
    ),
    ConstantInfo(
        name="m_GEF_PP",
        symbol=m_GEF_PP,
        value=220.0, # Hypothesis
        units="MeV",
        description="The rest energy scale of the GEF Planck Particle, the source of kappa.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_m_GEF_PP.md"
    ),
    ConstantInfo(
        name="g_sq",
        symbol=g_sq,
        value=None, # Varies by particle type
        description="Dimensionless internal rigidity constant, unique to each particle type.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_g_sq.md"
    ),
    ConstantInfo(
        name="P_env",
        symbol=P_env,
        value=None, # Varies by location
        description="Dimensionless environmental pressure from the 'Sea of Seas'.",
        category=ConstantCategory.MODEL_PARAMETER,
        sidecar_path="sidecar_constant_P_env.md"
    ),

    # === GEF Derived Kinematic Parameters ===
    ConstantInfo(
        name="b_0",
        symbol=b_0,
        value=np.sqrt(1 + 1e-122), # Derived from kappa
        description="Anisotropy parameter controlling metric deformation. b_0 = sqrt(1+kappa^2).",
        category=ConstantCategory.DERIVED_PARAMETER,
        sidecar_path="sidecar_constant_b_0.md"
    ),
    ConstantInfo(
        name="m_euc",
        symbol=m_euc,
        value=None, # Symbolic, m_euc = m_0 * c
        description="Euclidean mass parameter in the foundational GEF action.",
        category=ConstantCategory.DERIVED_PARAMETER,
        sidecar_path="sidecar_constant_m_euc.md"
    ),
    ConstantInfo(
        name="m_0",
        symbol=m_0,
        value=None, # Symbolic, the physical mass
        description="Physical rest mass of a particle in the emergent Minkowski spacetime.",
        category=ConstantCategory.DERIVED_PARAMETER,
        sidecar_path="sidecar_constant_m_0.md"
    ),

    # === Fundamental Constants (as measured) ===
    ConstantInfo(
        name="c",
        symbol=c,
        value=299_792_458.0,
        units="meter/second",
        description="Speed of light in vacuum (exact by definition).",
        category=ConstantCategory.FUNDAMENTAL,
        sidecar_path="sidecar_constant_c.md"
    ),
    ConstantInfo(
        name="h_planck",
        symbol=h_planck, # TODO: Consider h_bar instead
        value=6.626_070_15e-34,
        units="joule*second",
        description="The Planck constant (exact by definition).",
        category=ConstantCategory.FUNDAMENTAL,
        sidecar_path="sidecar_constant_h_planck.md"
    ),
    ConstantInfo(
        name="alpha_em",
        symbol=alpha_em,
        value=_alpha_em_val, # CODATA 2018
        description="Fine-structure constant, the electromagnetic coupling.",
        category=ConstantCategory.FUNDAMENTAL,
        sidecar_path="sidecar_constant_alpha_em.md"
    ),
    
    # === GEF Internal & Conversion Factors ===
    ConstantInfo(
        name="GEF_LOOP_FACTOR",
        symbol=None, # Not a symbolic constant
        value=_gef_loop_factor,
        description="Geometric loop factor (16*pi^2) used in the derivation of induced EM coupling.",
        category=ConstantCategory.INTERNAL_FACTOR,
        sidecar_path="sidecar_constant_gef_loop_factor.md"
    ),
    ConstantInfo(
        name="eV",
        symbol=eV,
        value=1.602_176_634e-19,
        units="joule",
        description="Electron volt energy unit in terms of joules (exact by definition).",
        category=ConstantCategory.CONVERSION,
        sidecar_path="sidecar_constant_electron_volt.md"
    ),
]

# --- Dictionary and Utility Function ---
CONSTANTS_DICT: Dict[str, ConstantInfo] = {const.name: const for const in CONSTANTS}

def constants_info(as_dict: bool = False) -> List[Dict] | Dict[str, Dict]:
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

# --- Direct Access to Common Values (for convenience) ---
# Allows for `from gef.core.constants import SPEED_OF_LIGHT`
# SPEED_OF_LIGHT = CONSTANTS_DICT['c'].value
# PLANCK_CONSTANT = CONSTANTS_DICT['h_planck'].value