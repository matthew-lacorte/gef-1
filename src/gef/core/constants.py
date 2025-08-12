"""
Foundational constants and model parameters for the GEF core physics framework.

This file serves as the canonical Single Source of Truth (SSoT) for the numerical
values and symbolic representations of all constants used in GEF simulations and
derivations. It is a direct implementation of the GEF Canonical Glossary.

Version: 1.2.0 - Refined model structure by postulating M_Pl as a core input.
                - Enhanced clarity of naming and documentation fields.

Exports:
    - ConstantInfo: Pydantic model for constant metadata.
    - CONSTANTS: The canonical list of all framework constants.
    - CONSTANTS_DICT: A dictionary mapping constant names to their ConstantInfo objects.
    - constants_info: Utility to export all constants and their documentation.
    - All individual SymPy symbols for direct import in analytical work.
"""

__version__ = "1.2.0"

# --- Core Imports ---
import sympy as sp
import pint
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum
from dataclasses import asdict

# --- Placeholder for project-specific utilities ---
def positive_value(cls, v):
    if v is not None and v < 0:
        raise ValueError("Value must be non-negative")
    return v
# --- End Placeholder ---

ureg = pint.UnitRegistry()

# --- Core Data Structures ---

class ConstantCategory(str, Enum):
    """Defines the role of a constant within the GEF framework."""
    MODEL_INPUT = "Model Input"                 # A free parameter or postulated scale of the GEF model.
    DERIVED_PREDICTION = "Derived Prediction"   # A value predicted by the model from its inputs.
    OBSERVED_BENCHMARK = "Observed Benchmark"   # An external, measured value the model must reproduce.
    INTERNAL_FACTOR = "Internal Factor"         # A calculation helper, like the loop factor.
    CONVERSION = "Conversion"                   # A unit conversion factor (e.g., eV to Joules).
    PLACEHOLDER = "Placeholder"                 # A particle-specific or context-dependent value.

class ConstantInfo(BaseModel):
    """Metadata and value(s) for a single GEF framework constant."""
    name: str = Field(..., description="Canonical name of the constant (used as key).")
    symbol: Optional[sp.Basic] = Field(None, description="SymPy symbol for analytic calculations.")
    value: Optional[float] = Field(
        None,
        description="Default or reference numeric value.",
        validate_default=True,
    )
    units: str = Field("dimensionless", description="Units of the constant.")
    description: str = Field("", description="Brief description, aligned with the canonical glossary.")
    category: ConstantCategory = Field(..., description="The primary category/role of the constant.")
    relation: Optional[str] = Field(None, description="Symbolic relationship to other constants (LaTeX format).")

    class Config:
        frozen = True
        arbitrary_types_allowed = True

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value_positive(cls, v):
        return positive_value(cls, v)

# --- Symbolic Declarations ---
# Geometry & Flow
b_0 = sp.Symbol('b₀', real=True, positive=True)
c_emergent = sp.Symbol('c', real=True, positive=True)
kappa_bar = sp.Symbol('κ̄', real=True, positive=True)
epsilon_0 = sp.Symbol('ε₀', real=True, positive=True)

# Fields & Couplings
lambda_A = sp.Symbol('λ_A', real=True, positive=True)
lambda_G = sp.Symbol('λ_G', real=True, positive=True)
h_sq = sp.Symbol('h²', real=True, positive=True)

# Mass, Energy & Scales
M_fund = sp.Symbol('M_fund', real=True, positive=True)
M_Pl = sp.Symbol('M_Pl', real=True, positive=True)
V_DE = sp.Symbol('V_{DE}', real=True, positive=True)
P_factor = sp.Symbol('P', real=True, positive=True)
G_newton = sp.Symbol('G', real=True, positive=True)

# Particle Kinematics
m_euc = sp.Symbol('m', real=True, positive=True)
m_0 = sp.Symbol('m₀', real=True, positive=True)

# Dimensionless Couplings
alpha_em = sp.Symbol('\\alpha_{EM}', real=True, positive=True)
alpha_s = sp.Symbol('\\alpha_s', real=True, positive=True)
alpha_G = sp.Symbol('\\alpha_G', real=True, positive=True)

# Symbols for Observed Benchmarks
c_observed = sp.Symbol('c_{obs}', real=True, positive=True)
hbar = sp.Symbol('\\hbar', real=True, positive=True)
eV = sp.Symbol('eV', real=True, positive=True)

__all__ = [
    "ConstantInfo", "CONSTANTS", "CONSTANTS_DICT", "constants_info", "ConstantCategory",
    "b_0", "c_emergent", "kappa_bar", "epsilon_0", "lambda_A", "lambda_G", "h_sq",
    "M_fund", "M_Pl", "V_DE", "P_factor", "G_newton", "m_euc", "m_0",
    "alpha_em", "alpha_s", "alpha_G", "c_observed", "hbar", "eV",
    "__version__",
]

# --- Numerical Base Values for Calculations ---
_M_fund_val = 220.0  # MeV
_alpha_em_obs_val = 1 / 137.035999084
_gef_loop_factor = 16 * np.pi**2
_M_Pl_val = 1.22091e22  # MeV, from √(ħc/G)
_V_DE_obs_val = 2.39e-9  # MeV, from (ρ_DE)^(1/4)

# === The Single Source of Truth for all Framework Constants ===
CONSTANTS: List[ConstantInfo] = [
    # === 1. Core Model Inputs ===
    # These are the fundamental free parameters and postulated scales of the GEF theory.
    ConstantInfo(
        name="M_fund",
        symbol=M_fund,
        value=_M_fund_val,
        units="MeV",
        description="The Fundamental Mass Scale; the mass of the GEF Planck Particle (for matter).",
        category=ConstantCategory.MODEL_INPUT,
    ),
    ConstantInfo(
        name="M_Pl",
        symbol=M_Pl,
        value=_M_Pl_val,
        units="MeV",
        description="The Planck Mass, postulated as the energy scale of the Φ field vacuum (for substrate).",
        category=ConstantCategory.MODEL_INPUT,
    ),
    ConstantInfo(
        name="kappa_bar",
        symbol=kappa_bar,
        description="The baseline magnitude of the κ-flow in empty space.",
        category=ConstantCategory.MODEL_INPUT,
    ),
    ConstantInfo(
        name="lambda_A",
        symbol=lambda_A,
        description="The dimensionless flow-gauge coupling constant.",
        category=ConstantCategory.MODEL_INPUT,
    ),
    ConstantInfo(
        name="h_sq",
        symbol=h_sq,
        description="The fundamental constant defining the energy scale of the short-range 'Hook' interaction.",
        category=ConstantCategory.MODEL_INPUT,
    ),

    # === 2. Derived Predictions of the GEF Model ===
    # These values are calculated from the inputs and other predictions.
    ConstantInfo(
        name="epsilon_0",
        symbol=epsilon_0,
        description="The dimensionless parameter for isotropic Lorentz violation in the particle sector.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="b_0^2 - 1",
    ),
    ConstantInfo(
        name="b_0",
        symbol=b_0,
        description="The dimensionless measure of the w-dimension's anisotropy.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="\\sqrt{1 + \\epsilon_0}",
    ),
    ConstantInfo(
        name="c_emergent",
        symbol=c_emergent,
        description="The universal speed limit in the emergent (3+1)D spacetime.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="1 / b_0",
    ),
    ConstantInfo(
        name="lambda_G",
        symbol=lambda_G,
        value=_gef_loop_factor * _alpha_em_obs_val, # ~1.152
        description="The predicted value for the substrate's quartic self-coupling constant.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="16\\pi^2 \\alpha_{EM}",
    ),
    ConstantInfo(
        name="alpha_em_predicted_from_lambda_G",
        symbol=alpha_em,
        description="The Fine-Structure Constant as derived from the substrate self-coupling.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="\\lambda_G / (16\\pi^2)",
    ),
    ConstantInfo(
        name="alpha_G",
        symbol=alpha_G,
        value=(_M_fund_val / _M_Pl_val)**2,
        description="The predicted gravitational coupling constant for the fundamental particle.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="(M_{fund} / M_{Pl})^2",
    ),
    ConstantInfo(
        name="P_factor",
        symbol=P_factor,
        value=_V_DE_obs_val / _M_Pl_val,
        description="The dimensionless intersection factor governing cosmological hierarchies.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="V_{DE} / M_{Pl}",
    ),
    ConstantInfo(
        name="G_newton",
        symbol=G_newton,
        description="The effective, emergent Newton's constant.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="Derived from M_{fund} properties",
    ),
    ConstantInfo(
        name="alpha_s",
        symbol=alpha_s,
        description="The Strong Coupling Constant, derived from the geometry of the 'Hook Coupling'.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="Derived from h^2",
    ),

    # === 3. Observed Benchmarks (External Reality) ===
    # These are measured values that the model's predictions must match.
    ConstantInfo(
        name="c_observed",
        symbol=c_observed,
        value=299_792_458.0,
        units="m/s",
        description="The observed speed of light in vacuum. A fundamental benchmark for the theory.",
        category=ConstantCategory.OBSERVED_BENCHMARK,
    ),
    ConstantInfo(
        name="hbar",
        symbol=hbar,
        value=1.054571817e-34,
        units="J*s",
        description="The reduced Planck constant (h/2π), an observed quantum scale.",
        category=ConstantCategory.OBSERVED_BENCHMARK,
    ),
    ConstantInfo(
        name="alpha_em_observed",
        symbol=alpha_em,
        value=_alpha_em_obs_val,
        description="The observed fine-structure constant. A fundamental benchmark for the theory.",
        category=ConstantCategory.OBSERVED_BENCHMARK,
    ),
    ConstantInfo(
        name="V_DE",
        symbol=V_DE,
        value=_V_DE_obs_val,
        units="MeV",
        description="The observed characteristic energy scale of the emergent 3D vacuum (Dark Energy).",
        category=ConstantCategory.OBSERVED_BENCHMARK,
    ),

    # === 4. Placeholders & Context-Dependent Values ===
    ConstantInfo(
        name="m_euc",
        symbol=m_euc,
        description="The Euclidean mass parameter appearing in the 4D Lagrangian.",
        category=ConstantCategory.PLACEHOLDER,
        relation="m_0 c",
    ),
    ConstantInfo(
        name="m_0",
        symbol=m_0,
        description="The physical rest mass of a specific particle in (3+1)D spacetime.",
        category=ConstantCategory.PLACEHOLDER,
    ),
    
    # === 5. Internal Factors & Unit Conversions ===
    ConstantInfo(
        name="GEF_LOOP_FACTOR",
        value=_gef_loop_factor,
        description="Geometric loop factor (16*pi^2) used in induced EM coupling derivation.",
        category=ConstantCategory.INTERNAL_FACTOR,
    ),
    ConstantInfo(
        name="eV",
        symbol=eV,
        value=1.602_176_634e-19,
        units="J",
        description="Electron volt energy unit in terms of Joules (exact by definition).",
        category=ConstantCategory.CONVERSION,
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