"""
Foundational constants and model parameters for the GEF core physics framework.

This file serves as the canonical Single Source of Truth (SSoT) for the numerical
values and symbolic representations of all constants used in GEF simulations and
derivations. It is a direct implementation of the GEF Canonical Glossary.

Version: 1.1.0 - Explicitly separates Model Inputs, Derived Predictions, and Observed Benchmarks.

Exports:
    - ConstantInfo: Pydantic model for constant metadata.
    - CONSTANTS: The canonical list of all framework constants.
    - CONSTANTS_DICT: A dictionary mapping constant names to their ConstantInfo objects.
    - constants_info: Utility to export all constants and their documentation.
    - All individual SymPy symbols for direct import in analytical work.
"""

__version__ = "1.1.0"

# --- Core Imports ---
import sympy as sp
import pint
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum

# --- Placeholder for project-specific utilities ---
# In a real project, these would be imported from a shared utilities module.
def asdict(model): return model.dict()
def positive_value(cls, v):
    if v is not None and v < 0:
        raise ValueError("Value must be non-negative")
    return v
# --- End Placeholder ---

ureg = pint.UnitRegistry()

# --- Core Data Structures ---

class ConstantCategory(str, Enum):
    """Defines the role of a constant within the GEF framework."""
    MODEL_INPUT = "Model Input"                 # A free parameter of the GEF model (e.g., κ∞, M_fund).
    DERIVED_PREDICTION = "Derived Prediction"   # A value predicted by the model from its inputs (e.g., c_emergent).
    OBSERVED_BENCHMARK = "Observed Benchmark"   # An external, measured value the model must reproduce.
    INTERNAL_FACTOR = "Internal Factor"         # A calculation helper, like the loop factor.
    CONVERSION = "Conversion"                   # A unit conversion factor (e.g., eV to Joules).
    PLACEHOLDER = "Placeholder"                 # A particle-specific or context-dependent value.

class ConstantInfo(BaseModel):
    """Metadata and value(s) for a single GEF framework constant."""
    name: str = Field(..., description="Canonical name of the constant (used as key).")
    symbol: Optional[sp.Basic] = Field(None, description="SymPy symbol for analytic calculations.")
    value: Optional[float] = Field(None, description="Default or reference numeric value.")
    units: str = Field("dimensionless", description="Units of the constant.")
    description: str = Field("", description="Brief description for docs, aligned with the canonical glossary.")
    category: ConstantCategory = Field(..., description="The primary category/role of the constant.")
    relation: Optional[str] = Field(None, description="Symbolic relationship to other constants.")

    class Config:
        frozen = True
        arbitrary_types_allowed = True

    _validate_value_positive = field_validator("value", pre=True, always=True)(positive_value)

# --- Symbolic Declarations (from Glossary) ---
# Geometry & Flow
b_0 = sp.Symbol('b₀', real=True, positive=True)
c_emergent = sp.Symbol('c', real=True, positive=True)
kappa_inf = sp.Symbol('κ∞', real=True, positive=True)
epsilon_0 = sp.Symbol('ε₀', real=True, positive=True)

# Fields & Couplings
lambda_A = sp.Symbol('λ_A', real=True, positive=True)
lambda_G = sp.Symbol('λ_G', real=True, positive=True)
h_sq = sp.Symbol('h²', real=True, positive=True)

# Mass, Energy & Scales
M_fund = sp.Symbol('M_fund', real=True, positive=True)
M_Pl = sp.Symbol('M_Pl', real=True, positive=True)
V_DE = sp.Symbol('V_DE', real=True, positive=True)
P_factor = sp.Symbol('P', real=True, positive=True)
G_newton = sp.Symbol('G', real=True, positive=True)

# Particle Kinematics
m_euc = sp.Symbol('m', real=True, positive=True)
m_0 = sp.Symbol('m₀', real=True, positive=True)

# Dimensionless Couplings
alpha_em = sp.Symbol('α_EM', real=True, positive=True)
alpha_s = sp.Symbol('α_s', real=True, positive=True)
alpha_G = sp.Symbol('α_G', real=True, positive=True)

# Symbols for Observed Benchmarks
c_observed = sp.Symbol('c_obs', real=True, positive=True)
hbar = sp.Symbol('hbar', real=True, positive=True)
eV = sp.Symbol('eV', real=True, positive=True)

__all__ = [
    "ConstantInfo", "CONSTANTS", "CONSTANTS_DICT", "constants_info", "ConstantCategory",
    "b_0", "c_emergent", "kappa_inf", "epsilon_0", "lambda_A", "lambda_G", "h_sq",
    "M_fund", "M_Pl", "V_DE", "P_factor", "G_newton", "m_euc", "m_0",
    "alpha_em", "alpha_s", "alpha_G", "c_observed", "hbar", "eV",
    "__version__",
]

# --- Numerical Base Values for Calculations ---
_M_fund_val = 220.0  # MeV
_alpha_em_obs_val = 1 / 137.035999084
_gef_loop_factor = 16 * np.pi**2
_M_Pl_obs_val = 1.22091e22  # MeV, from √(ħc/G)
_V_DE_obs_val = 2.39e-9  # MeV, from (ρ_DE)^(1/4)

# === The Single Source of Truth for all Framework Constants ===
CONSTANTS: List[ConstantInfo] = [
    # === 1. Core Model Inputs ===
    # These are the fundamental free parameters of the GEF theory.
    ConstantInfo(
        name="kappa_inf",
        symbol=kappa_inf,
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
    ConstantInfo(
        name="M_fund",
        symbol=M_fund,
        value=_M_fund_val,
        units="MeV",
        description="The Fundamental Mass Scale; the mass of the GEF Planck Particle.",
        category=ConstantCategory.MODEL_INPUT,
    ),

    # === 2. Derived Predictions of the GEF Model ===
    # These values are calculated from the inputs and other predictions.
    ConstantInfo(
        name="epsilon_0",
        symbol=epsilon_0,
        description="The dimensionless parameter for isotropic Lorentz violation in the particle sector.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="b₀² - 1",
    ),
    ConstantInfo(
        name="b_0",
        symbol=b_0,
        description="The dimensionless measure of the w-dimension's anisotropy.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="sqrt(1 + epsilon_0)",
    ),
    ConstantInfo(
        name="c_emergent",
        symbol=c_emergent,
        description="The universal speed limit in the emergent (3+1)D spacetime.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="1/b₀",
    ),
    ConstantInfo(
        name="lambda_G",
        symbol=lambda_G,
        value=_gef_loop_factor * _alpha_em_obs_val, # ~1.152
        description="The predicted value for the substrate's quartic self-coupling constant.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="16π² * α_EM",
    ),
    ConstantInfo(
        name="G_newton",
        symbol=G_newton,
        description="The effective, emergent Newton's constant.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="Derived from M_fund properties.",
    ),
     ConstantInfo(
        name="P_factor",
        symbol=P_factor,
        value=_V_DE_obs_val / _M_Pl_obs_val,
        description="The dimensionless intersection factor governing cosmological hierarchies.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="V_DE / M_Pl",
    ),
    ConstantInfo(
        name="alpha_G",
        symbol=alpha_G,
        value=(_M_fund_val / _M_Pl_obs_val)**2,
        description="The predicted gravitational coupling constant for the fundamental particle.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="(M_fund / M_Pl)²",
    ),
    ConstantInfo(
        name="alpha_s",
        symbol=alpha_s,
        description="The Strong Coupling Constant, derived from the geometry of the 'Hook Coupling'.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="Derived from h².",
    ),
    ConstantInfo(
        name="alpha_em_predicted",
        symbol=alpha_em,
        description="The Fine-Structure Constant as derived from the substrate self-coupling.",
        category=ConstantCategory.DERIVED_PREDICTION,
        relation="λ_G / 16π²",
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
        symbol=alpha_em, # Uses the same symbol as the prediction, context is key.
        value=_alpha_em_obs_val,
        description="The observed fine-structure constant. A fundamental benchmark for the theory.",
        category=ConstantCategory.OBSERVED_BENCHMARK,
    ),
    ConstantInfo(
        name="M_Pl_observed",
        symbol=M_Pl,
        value=_M_Pl_obs_val,
        units="MeV",
        description="The observed Planck Mass, interpreted as the energy scale of the Φ field vacuum.",
        category=ConstantCategory.OBSERVED_BENCHMARK,
    ),
    ConstantInfo(
        name="V_DE_observed",
        symbol=V_DE,
        value=_V_DE_obs_val,
        units="MeV",
        description="The observed characteristic energy scale of the emergent 3D vacuum.",
        category=ConstantCategory.OBSERVED_BENCHMARK,
    ),

    # === 4. Placeholders & Context-Dependent Values ===
    ConstantInfo(
        name="m_euc",
        symbol=m_euc,
        description="The Euclidean mass parameter appearing in the 4D Lagrangian.",
        category=ConstantCategory.PLACEHOLDER,
        relation="m₀ * c",
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

# --- Dictionary and Utility Function (no changes needed) ---
CONSTANTS_DICT: Dict[str, ConstantInfo] = {const.name: const for const in CONSTANTS}

def constants_info(as_dict: bool = False) -> List[Dict] | Dict[str, Dict]:
    """Returns all constants and metadata for documentation/automation."""
    if as_dict:
        return {const.name: asdict(const) for const in CONSTANTS}
    else:
        return [asdict(const) for const in CONSTANTS]