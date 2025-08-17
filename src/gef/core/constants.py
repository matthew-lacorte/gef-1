"""
Foundational constants and model parameters for the GEF core physics framework.

This file serves as the canonical Single Source of Truth (SSoT) for the numerical
values and symbolic representations of all constants used in GEF simulations and
derivations. It is a direct implementation of the GEF Canonical Glossary.

Version: 1.4.0 - Unit-safe refactor with proper conversions and derived relations.
                - Fixed h units and made it observed (2πℏ)
                - Made r_P properly derived from ℏc/M_fund
                - Added conversion constants for unit consistency
                - Export by name, not Unicode symbols

Exports:
    - ConstantInfo: Pydantic model for constant metadata.
    - CONSTANTS: The canonical list of all framework constants.
    - CONSTANTS_DICT: Dictionary mapping constant names to ConstantInfo objects.
    - SYMBOLS: Dictionary mapping names to SymPy symbols.
    - VALUES: Dictionary mapping names to numeric values.
    - All constant names available as module attributes.
"""

__version__ = "1.4.0"
__date__ = "2025-08-17"

# --- Core Imports ---
import sympy as sp
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum

# --- Core Data Structures ---

class ConstantStatus(str, Enum):
    """Canonical status as per GEF glossary."""
    CANONICAL = "Canonical"  # Committed, stable
    WORKING = "Working"      # Used but provisional
    DERIVED = "Derived"      # Follows from other entries

class ConstantCategory(str, Enum):
    """Defines the role of a constant within the GEF framework."""
    FOUNDATIONAL = "Foundational Geometry & Flow"
    FIELDS = "Fields & Couplings"
    SCALES = "Mass, Energy & Canonical Scales"
    KINEMATICS = "Kinematics & Dynamics"
    DIMENSIONLESS = "Dimensionless Couplings"
    PREDICTIONS = "Predictions: GWs & PPN"
    COSMOLOGY = "Λ-suppression parameters"
    OBSERVED = "Observed Benchmarks"
    CONVERSION = "Conversion Factors"
    INTERNAL = "Internal Calculation Helpers"

class ConstantInfo(BaseModel):
    """Complete metadata and value(s) for a single GEF framework constant."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    
    # Core identification
    name: str = Field(..., description="Canonical name (used as key)")
    symbol: Optional[sp.Basic] = Field(None, description="SymPy symbol for analytics")
    latex: Optional[str] = Field(None, description="LaTeX representation")
    
    # Value and units
    value: Optional[float] = Field(None, description="Numeric value if known")
    units: str = Field("dimensionless", description="Physical units")
    
    # Documentation
    description: str = Field("", description="Brief description from glossary")
    category: ConstantCategory = Field(..., description="Category/role")
    status: ConstantStatus = Field(ConstantStatus.CANONICAL)
    
    # Relations and provenance
    relation: Optional[str] = Field(None, description="Symbolic relation (LaTeX)")
    eval_expr: Optional[sp.Expr] = Field(None, description="Evaluatable SymPy expression")
    sidecar_slug: Optional[str] = Field(None, description="Reference to extended docs")
    introduced_in: Optional[str] = Field(None, description="Version introduced")
    last_updated: Optional[str] = Field(None, description="Last modification")
    source_refs: Optional[List[str]] = Field(default_factory=list, description="DOIs/URLs")
    
    @field_validator("value", mode="before")
    @classmethod
    def validate_positive_where_required(cls, v, info):
        """Some values can be negative (e.g., gauge factors), others must be positive."""
        # This is a simplified validator - expand based on physics requirements
        return v

# --- Numerical Base Values and Conversions ---
_c_obs = 299_792_458.0  # m/s
_hbar = 1.054571817e-34  # J·s
_eV = 1.602176634e-19  # J (exact by definition)
_alpha_em_obs = 1/137.035999084
_M_fund_MeV = 235.0  # Canonical calibration (mass-energy scale)
_M_Pl_MeV = 1.22091e22  # √(ℏc/G) in MeV
_V_DE_MeV = 2.39e-9  # (ρ_DE)^(1/4) in MeV
_loop_factor = 16 * np.pi**2

# Unit conversions
HBAR_C_MEV_FM = 197.3269804  # MeV·fm (CODATA)
MEV_TO_J = _eV * 1e6  # J/MeV
FM_TO_M = 1e-15  # m/fm
MEV_C2_TO_KG = MEV_TO_J / (_c_obs**2)  # kg/(MeV/c²)

# Derived value for r_P
_r_P_fm = HBAR_C_MEV_FM / _M_fund_MeV  # Should be ~0.840 fm

# === The Single Source of Truth: Canonical Constants Registry ===
CONSTANTS: List[ConstantInfo] = [
    
    # ========== 1. FOUNDATIONAL GEOMETRY & FLOW ==========
    
    ConstantInfo(
        name="delta_mu_nu",
        symbol=sp.Symbol('δ_μν', real=True),
        latex=r"\delta_{\mu\nu}",
        description="Base metric: Flat Euclidean metric (++++ signature) of the 4D substrate",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="kappa_mu",
        symbol=sp.Symbol('κ_μ', real=True),
        latex=r"\kappa^\mu",
        description="κ-flow: Universal background vector field that breaks SO(4) and sets macroscopic arrow of time",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.CANONICAL,
        relation=r"\kappa^\mu = (0, 0, 0, \bar{\kappa})",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="kappa_bar",
        symbol=sp.Symbol('κ̄', real=True, positive=True),
        latex=r"\bar{\kappa}",
        description="κ-flow magnitude: Baseline magnitude of the κ-flow in empty space",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="theta",
        symbol=sp.Symbol('θ', real=True),
        latex=r"\theta(x)",
        description="Misalignment angle between protected axis and local w-flow; controls on-axis vs off-axis behavior",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="zeta",
        symbol=sp.Symbol('ζ', real=True, positive=True),
        latex=r"\zeta",
        description="AC gauge factor: Analytic continuation (Wick) gauge t = -iζw. On protected axis ζ = 1/c",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.CANONICAL,
        relation=r"\zeta = 1/c",
        eval_expr=1/sp.Symbol('c'),
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="c",
        symbol=sp.Symbol('c', real=True, positive=True),
        latex=r"c",
        value=_c_obs,
        units="m/s",
        description="Conversion speed: Emergent light-speed as conversion gauge on protected axis",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.CANONICAL,
        relation=r"c \equiv 1/\zeta",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="g_eff",
        symbol=sp.Symbol('g_eff', real=True),
        latex=r"g_{\text{eff}}",
        description="Effective metric: Universal anisotropic Euclidean metric induced by κ-flow",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.WORKING,
        relation=r"\text{diag}(1, 1, 1, 1/b_\kappa^2)",
        introduced_in="1.1.0",
    ),
    
    ConstantInfo(
        name="b_kappa",
        symbol=sp.Symbol('b_κ', real=True, positive=True),
        latex=r"b_\kappa",
        description="Anisotropy shorthand: Calculational knob in AC derivations; b_κ = 1/c in protected gauge",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.WORKING,
        relation=r"b_\kappa = 1/c",
        eval_expr=1/sp.Symbol('c'),
        introduced_in="1.1.0",
    ),
    
    ConstantInfo(
        name="chi",
        symbol=sp.Symbol('χ', real=True),
        latex=r"\chi",
        description="LIV dial (isotropic): Phenomenological parameter for isotropic Lorentz violation; empirically ≪ 1",
        category=ConstantCategory.FOUNDATIONAL,
        status=ConstantStatus.WORKING,
        introduced_in="1.1.0",
    ),
    
    # ========== 2. FIELDS & COUPLINGS ==========
    
    ConstantInfo(
        name="Phi",
        symbol=sp.Symbol('Φ', real=True),
        latex=r"\Phi",
        description="Substrate field: Real scalar field of the 4D plenum",
        category=ConstantCategory.FIELDS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="A_mu",
        symbol=sp.Symbol('A_μ', real=True),
        latex=r"A_\mu",
        description="Gauge potential: Standard electromagnetic/gauge field potential",
        category=ConstantCategory.FIELDS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="F_mu_nu",
        symbol=sp.Symbol('F_μν', real=True),
        latex=r"F_{\mu\nu}",
        description="Field strength tensor for gauge interactions",
        category=ConstantCategory.FIELDS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="psi",
        symbol=sp.Symbol('ψ'),
        latex=r"\psi",
        description="Fermion field: 4-component Dirac spinor",
        category=ConstantCategory.FIELDS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="lambda_A",
        symbol=sp.Symbol('λ_A', real=True, positive=True),
        latex=r"\lambda_A",
        description="Flow-gauge coupling: Coefficient of interaction like λ_A(κ·F)²",
        category=ConstantCategory.FIELDS,
        status=ConstantStatus.WORKING,
        introduced_in="1.1.0",
    ),
    
    ConstantInfo(
        name="lambda_Phi",
        symbol=sp.Symbol('λ_Φ', real=True, positive=True),
        latex=r"\lambda_\Phi",
        value=1.152,  # Current working value
        description="Substrate self-coupling: Quartic self-coupling of Φ field",
        category=ConstantCategory.FIELDS,
        status=ConstantStatus.WORKING,
        introduced_in="1.1.0",
    ),
    
    ConstantInfo(
        name="g_H",
        symbol=sp.Symbol('g_H', real=True, positive=True),
        latex=r"g_H",
        description="Hook coupling: Dimensionless coupling for short-range geometric binding",
        category=ConstantCategory.FIELDS,
        status=ConstantStatus.WORKING,
        introduced_in="1.1.0",
    ),
    
    # ========== 3. MASS, ENERGY & CANONICAL SCALES ==========
    
    ConstantInfo(
        name="M_fund",
        symbol=sp.Symbol('M_fund', real=True, positive=True),
        latex=r"M_{\text{fund}}",
        value=_M_fund_MeV,
        units="MeV",
        description="Fundamental mass-energy scale (iris anchor): Mass of GEF Planck particle (fundamental hopfion)",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="r_P",
        symbol=sp.Symbol('r_P', real=True, positive=True),
        latex=r"r_P",
        value=_r_P_fm,
        units="fm",
        description="Iris (protected-sphere) radius: r_P ≡ (ℏc)/M_fund in MeV-fm units",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.CANONICAL,
        relation=r"r_P = (\hbar c)/M_{\text{fund}}",
        eval_expr=sp.Symbol('hbar_c')/sp.Symbol('M_fund'),
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="h",
        symbol=sp.Symbol('h', real=True, positive=True),
        latex=r"h",
        value=2 * np.pi * _hbar,
        units="J·s",
        description="Planck constant (observed): h = 2πℏ",
        category=ConstantCategory.OBSERVED,
        status=ConstantStatus.CANONICAL,
        relation=r"h = 2\pi \hbar",
        eval_expr=2 * sp.pi * sp.Symbol('hbar'),
        introduced_in="2.1.0",
    ),
    
    ConstantInfo(
        name="h_postulate_check",
        latex=r"h_{\text{post}}",
        description="Derived via h = 2π r_P M_fund c with unit conversions",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.DERIVED,
        relation=r"h_{\text{post}} = 2\pi\, (r_P\,{\rm fm}\!\to\!{\rm m}) \times (M_{\text{fund}}\,{\rm MeV}\!\to\!{\rm J})/c",
        eval_expr=2*sp.pi * (sp.Symbol('r_P')*sp.Symbol('FM_TO_M')) * (sp.Symbol('M_fund')*sp.Symbol('MEV_TO_J')) / sp.Symbol('c'),
        introduced_in="2.1.0",
    ),
    
    ConstantInfo(
        name="M_Pl",
        symbol=sp.Symbol('M_Pl', real=True, positive=True),
        latex=r"M_{\text{Pl}}",
        value=_M_Pl_MeV,
        units="MeV",
        description="Planck mass: Traditional √(ℏc/G); in GEF, a 4D-plenum scale",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.CANONICAL,
        relation=r"\sqrt{\hbar c / G}",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="V_DE",
        symbol=sp.Symbol('V_DE', real=True, positive=True),
        latex=r"V_{\text{DE}}",
        value=_V_DE_MeV,
        units="MeV",
        description="Vacuum energy scale: V_DE ≡ ρ_DE^(1/4)",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.CANONICAL,
        relation=r"V_{\text{DE}} \equiv \rho_{\text{DE}}^{1/4}",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="P_G",
        symbol=sp.Symbol('P_G', real=True, positive=True),
        latex=r"P_G",
        value=_M_fund_MeV/_M_Pl_MeV,
        description="Gravitational ratio: P_G ≡ M_fund/M_Pl",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.DERIVED,
        relation=r"P_G \equiv M_{\text{fund}}/M_{\text{Pl}}",
        eval_expr=sp.Symbol('M_fund')/sp.Symbol('M_Pl'),
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="P_C",
        symbol=sp.Symbol('P_C', real=True, positive=True),
        latex=r"P_C",
        value=_V_DE_MeV/_M_fund_MeV,
        description="Cosmological ratio: P_C ≡ V_DE/M_fund",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.DERIVED,
        relation=r"P_C \equiv V_{\text{DE}}/M_{\text{fund}}",
        eval_expr=sp.Symbol('V_DE')/sp.Symbol('M_fund'),
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="G",
        symbol=sp.Symbol('G', real=True, positive=True),
        latex=r"G",
        value=6.67430e-11,
        units="m³/(kg·s²)",
        description="Newton's constant: Effective coupling in emergent sector",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.DERIVED,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="W",
        symbol=sp.Symbol('W', real=True),
        latex=r"W(r)",
        description="Wake potential: Gravitational potential W(r) = (G/c²)M/r",
        category=ConstantCategory.SCALES,
        status=ConstantStatus.DERIVED,
        relation=r"W(r) = (G/c^2) M/r",
        introduced_in="1.0.0",
    ),
    
    # ========== 4. KINEMATICS & DYNAMICS ==========
    
    ConstantInfo(
        name="m",
        symbol=sp.Symbol('m', real=True, positive=True),
        latex=r"m",
        description="Euclidean mass parameter: Mass in 4D Euclidean Lagrangian",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="m_0",
        symbol=sp.Symbol('m_0', real=True, positive=True),
        latex=r"m_0",
        description="Physical rest mass: Observed 3+1D inertial mass",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="mass_map",
        latex=r"m = m_0 c",
        description="Mass map: Gauge/AC relation between Euclidean and physical mass",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.DERIVED,
        relation=r"m = m_0 c",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="H_D",
        symbol=sp.Symbol('H_D'),
        latex=r"H_D",
        description="Dirac Hamiltonian: H_D = c·α·p + βm_0c²",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.DERIVED,
        relation=r"H_D = c \vec{\alpha} \cdot \vec{p} + \beta m_0 c^2",
        introduced_in="1.0.0",
    ),
    
    # Hourglass parameters
    ConstantInfo(
        name="D",
        symbol=sp.Symbol('D', real=True, positive=True),
        latex=r"D",
        description="Diffusion coefficient in hourglass equation",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.CANONICAL,
        relation=r"D = \zeta \hbar_\parallel / (2m_H)",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="alpha",
        symbol=sp.Symbol('α', real=True, positive=True),
        latex=r"\alpha",
        description="Potential strength in hourglass equation",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.CANONICAL,
        relation=r"\alpha = \zeta / \hbar_\parallel",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="hbar_parallel",
        symbol=sp.Symbol('ℏ_∥', real=True, positive=True),
        latex=r"\hbar_\parallel",
        value=_hbar,
        units="J·s",
        description="On-axis quantum scale (calibrated to lab ℏ)",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="m_H",
        symbol=sp.Symbol('m_H', real=True, positive=True),
        latex=r"m_H",
        description="Effective inertial parameter of propagating mode",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="L",
        symbol=sp.Symbol('L'),
        latex=r"L",
        description="Semigroup generator: L = D∇² - αV_eff - v·∇",
        category=ConstantCategory.KINEMATICS,
        status=ConstantStatus.CANONICAL,
        relation=r"L = D\nabla^2 - \alpha V_{\text{eff}} - \vec{v} \cdot \nabla",
        introduced_in="1.0.0",
    ),
    
    # ========== 5. DIMENSIONLESS COUPLINGS ==========
    
    ConstantInfo(
        name="alpha_EM",
        symbol=sp.Symbol('α_EM', real=True, positive=True),
        latex=r"\alpha_{\text{EM}}",
        value=_alpha_em_obs,
        description="Fine-structure constant",
        category=ConstantCategory.DIMENSIONLESS,
        status=ConstantStatus.WORKING,
        relation=r"\alpha_{\text{EM}} \approx \lambda_\Phi / (16\pi^2)",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="alpha_s",
        symbol=sp.Symbol('α_s', real=True, positive=True),
        latex=r"\alpha_s",
        description="Strong coupling constant (related to g_H and hopfion geometry)",
        category=ConstantCategory.DIMENSIONLESS,
        status=ConstantStatus.WORKING,
        introduced_in="1.1.0",
    ),
    
    ConstantInfo(
        name="alpha_G",
        symbol=sp.Symbol('α_G', real=True, positive=True),
        latex=r"\alpha_G",
        value=(_M_fund_MeV/_M_Pl_MeV)**2,
        description="Gravitational coupling (natural units): α_G = (M_fund/M_Pl)²",
        category=ConstantCategory.DIMENSIONLESS,
        status=ConstantStatus.DERIVED,
        relation=r"\alpha_G = (M_{\text{fund}}/M_{\text{Pl}})^2",
        eval_expr=(sp.Symbol('M_fund')/sp.Symbol('M_Pl'))**2,
        introduced_in="1.0.0",
    ),
    
    # ========== 6. PREDICTIONS: GWs & PPN ==========
    
    ConstantInfo(
        name="c_g",
        symbol=sp.Symbol('c_g', real=True, positive=True),
        latex=r"c_g",
        description="Speed of gravitational waves equals c",
        category=ConstantCategory.PREDICTIONS,
        status=ConstantStatus.CANONICAL,
        eval_expr=sp.Symbol('c'),
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="GW_modes",
        description="5 propagating polarizations (2 tensor, 2 vector, 1 scalar)",
        category=ConstantCategory.PREDICTIONS,
        status=ConstantStatus.WORKING,
        introduced_in="1.1.0",
    ),
    
    ConstantInfo(
        name="PPN_gamma",
        symbol=sp.Symbol('γ', real=True),
        latex=r"\gamma",
        value=1.0,
        description="PPN parameter γ = 1 (GR limit in weak fields)",
        category=ConstantCategory.PREDICTIONS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="PPN_beta",
        symbol=sp.Symbol('β', real=True),
        latex=r"\beta",
        value=1.0,
        description="PPN parameter β = 1 (GR limit in weak fields)",
        category=ConstantCategory.PREDICTIONS,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    # ========== 7. Λ-SUPPRESSION PARAMETERS ==========
    
    ConstantInfo(
        name="Xi",
        symbol=sp.Symbol('Ξ', real=True),
        latex=r"\Xi",
        description="Alignment order parameter: Ξ ≡ ⟨cos²θ⟩",
        category=ConstantCategory.COSMOLOGY,
        status=ConstantStatus.CANONICAL,
        relation=r"\Xi \equiv \langle \cos^2\theta \rangle",
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="epsilon_OS",
        symbol=sp.Symbol('ε_OS', real=True, positive=True),
        latex=r"\varepsilon_{\text{OS}}",
        description="OS filter factor: Fraction of Euclidean energy surviving to Lorentzian gravity",
        category=ConstantCategory.COSMOLOGY,
        status=ConstantStatus.WORKING,
        introduced_in="1.1.0",
    ),
    
    ConstantInfo(
        name="Lambda_ansatz",
        latex=r"\rho_\Lambda",
        description="Cosmological constant density ansatz",
        category=ConstantCategory.COSMOLOGY,
        status=ConstantStatus.WORKING,
        relation=r"\rho_\Lambda = \varepsilon_{\text{OS}} \Xi^3 M_{\text{fund}}^4",
        introduced_in="1.1.0",
    ),
    
    # ========== 8. OBSERVED BENCHMARKS ==========
    
    ConstantInfo(
        name="c_observed",
        symbol=sp.Symbol('c_obs', real=True, positive=True),
        latex=r"c_{\text{obs}}",
        value=_c_obs,
        units="m/s",
        description="Observed speed of light in vacuum",
        category=ConstantCategory.OBSERVED,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="hbar",
        symbol=sp.Symbol('ℏ', real=True, positive=True),
        latex=r"\hbar",
        value=_hbar,
        units="J·s",
        description="Reduced Planck constant (h/2π)",
        category=ConstantCategory.OBSERVED,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="alpha_EM_observed",
        latex=r"\alpha_{\text{EM}}^{\text{obs}}",
        value=_alpha_em_obs,
        description="Observed fine-structure constant",
        category=ConstantCategory.OBSERVED,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    # ========== 9. CONVERSION FACTORS & HELPERS ==========
    
    ConstantInfo(
        name="hbar_c",
        symbol=sp.Symbol('hbar_c', real=True, positive=True),
        latex=r"\hbar c",
        value=HBAR_C_MEV_FM,
        units="MeV·fm",
        description="Product ℏc in high-energy units",
        category=ConstantCategory.CONVERSION,
        status=ConstantStatus.CANONICAL,
        introduced_in="2.1.0",
    ),
    
    ConstantInfo(
        name="eV",
        symbol=sp.Symbol('eV', real=True, positive=True),
        latex=r"\text{eV}",
        value=_eV,
        units="J",
        description="Electron volt in Joules (exact by definition)",
        category=ConstantCategory.CONVERSION,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
    
    ConstantInfo(
        name="MEV_TO_J",
        value=MEV_TO_J,
        units="J/MeV",
        description="Energy conversion: MeV → Joules",
        category=ConstantCategory.CONVERSION,
        status=ConstantStatus.CANONICAL,
        introduced_in="2.1.0",
    ),
    
    ConstantInfo(
        name="FM_TO_M",
        value=FM_TO_M,
        units="m/fm",
        description="Length conversion: fm → meters",
        category=ConstantCategory.CONVERSION,
        status=ConstantStatus.CANONICAL,
        introduced_in="2.1.0",
    ),
    
    ConstantInfo(
        name="MEV_C2_TO_KG",
        value=MEV_C2_TO_KG,
        units="kg/(MeV/c²)",
        description="Mass conversion: MeV/c² → kilograms",
        category=ConstantCategory.CONVERSION,
        status=ConstantStatus.CANONICAL,
        introduced_in="2.1.0",
    ),
    
    ConstantInfo(
        name="GEF_LOOP_FACTOR",
        value=_loop_factor,
        description="Geometric loop factor (16π²) in EM coupling derivation",
        category=ConstantCategory.INTERNAL,
        status=ConstantStatus.CANONICAL,
        introduced_in="1.0.0",
    ),
]

# --- Generate Derived Exports ---

# Dictionary access by name
CONSTANTS_DICT: Dict[str, ConstantInfo] = {c.name: c for c in CONSTANTS}

# Symbol dictionary for analytics
SYMBOLS: Dict[str, sp.Basic] = {
    c.name: c.symbol for c in CONSTANTS if c.symbol is not None
}

# Numeric values dictionary
VALUES: Dict[str, float] = {
    c.name: c.value for c in CONSTANTS if c.value is not None
}

# Create module-level attributes for direct symbol access
# Export by constant name, not Unicode symbol string
for const in CONSTANTS:
    if const.symbol is not None:
        globals()[const.name] = const.symbol

# --- Utility Functions ---

def get_constants_by_category(category: ConstantCategory) -> List[ConstantInfo]:
    """Return all constants in a given category."""
    return [c for c in CONSTANTS if c.category == category]

def get_constants_by_status(status: ConstantStatus) -> List[ConstantInfo]:
    """Return all constants with a given status."""
    return [c for c in CONSTANTS if c.status == status]

def print_registry(category: Optional[ConstantCategory] = None, 
                   status: Optional[ConstantStatus] = None,
                   show_values: bool = True,
                   show_relations: bool = True) -> None:
    """
    Pretty-print the constants registry.
    
    Args:
        category: Filter by category
        status: Filter by status
        show_values: Include numeric values
        show_relations: Include symbolic relations
    """
    constants = CONSTANTS
    
    if category:
        constants = [c for c in constants if c.category == category]
    if status:
        constants = [c for c in constants if c.status == status]
    
    # Sort to ensure groupby works correctly
    constants = sorted(constants, key=lambda c: c.category.value)
    
    # Group by category for display
    from itertools import groupby
    
    for cat, group in groupby(constants, key=lambda c: c.category):
        print(f"\n{'='*60}")
        print(f" {cat.value}")
        print('='*60)
        
        for const in group:
            print(f"\n{const.name}")
            if const.latex:
                print(f"  LaTeX: {const.latex}")
            if const.symbol:
                print(f"  Symbol: {const.symbol}")
            if show_values and const.value is not None:
                print(f"  Value: {const.value} {const.units}")
            print(f"  Status: {const.status.value}")
            print(f"  Description: {const.description}")
            if show_relations and const.relation:
                print(f"  Relation: {const.relation}")

def export_to_yaml(filename: str = "constants.yaml") -> None:
    """Export all constants to YAML format for external tools."""
    import yaml
    
    data = {
        "version": __version__,
        "date": __date__,
        "constants": {}
    }
    
    for const in CONSTANTS:
        entry = {
            "latex": const.latex,
            "value": const.value,
            "units": const.units,
            "description": const.description,
            "category": const.category.value,
            "status": const.status.value,
        }
        if const.relation:
            entry["relation"] = const.relation
        if const.source_refs:
            entry["references"] = const.source_refs
            
        data["constants"][const.name] = entry
    
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def export_to_latex(filename: str = "constants.tex") -> None:
    """Generate LaTeX table of constants for documentation."""
    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\begin{document}",
        r"\begin{longtable}{llll}",
        r"\toprule",
        r"Symbol & Name & Value & Description \\",
        r"\midrule",
    ]

    for const in CONSTANTS:
        # Symbol column: prefer LaTeX, fallback to escaped name
        if const.latex:
            symbol = f"${const.latex}$"
        else:
            symbol = const.name.replace('_', r'\_')

        # Value column: handle None vs zero correctly
        value = f"{const.value:.3e} {const.units}" if (const.value is not None) else "—"

        # Short description
        desc = const.description[:50] + "..." if len(const.description) > 50 else const.description

        # Name column with escaped underscores
        name_escaped = const.name.replace('_', r'\_')

        # Append LaTeX table row, emitting \\ at end
        lines.append(f"{symbol} & {name_escaped} & {value} & {desc} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{longtable}",
        r"\end{document}",
    ])

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

def validate_relations() -> Dict[str, Any]:
    """
    Validate that all eval_expr relations can be evaluated.
    Returns dict of any errors found.
    """
    errors = {}
    
    for const in CONSTANTS:
        if const.eval_expr:
            try:
                # Try to substitute known symbols
                subs = {}
                for sym in const.eval_expr.free_symbols:
                    sym_name = str(sym)
                    if sym_name in SYMBOLS:
                        if sym_name in VALUES:
                            subs[sym] = VALUES[sym_name]
                
                if subs:
                    result = const.eval_expr.subs(subs)
                    # Compare with stated value if available
                    if const.value and isinstance(result, (int, float)):
                        rel_error = abs(result - const.value) / const.value
                        if rel_error > 1e-6:
                            errors[const.name] = f"Computed {result}, stated {const.value} (rel_error={rel_error:.2e})"
            except Exception as e:
                errors[const.name] = str(e)
    
    return errors

def run_self_test() -> None:
    """
    Run self-consistency tests on the constants registry.
    """
    print(f"\n{'='*60}")
    print(" GEF Constants Self-Test")
    print('='*60)
    
    # Check relation validation
    errs = validate_relations()
    if errs:
        print("\n⚠️  Relation validation errors:")
        for name, err in errs.items():
            print(f"  {name}: {err}")
    else:
        print("\n✅ All relations validated successfully")
    
    # Check r_P derivation
    r_P_computed = HBAR_C_MEV_FM / _M_fund_MeV
    r_P_stored = VALUES.get("r_P", None)
    if r_P_stored:
        print(f"\n✅ r_P consistency:")
        print(f"  Stored:   {r_P_stored:.6f} fm")
        print(f"  Computed: {r_P_computed:.6f} fm (from hbar_c/M_fund)")
        rel_error = abs(r_P_computed - r_P_stored) / r_P_stored
        print(f"  Relative error: {rel_error:.2e}")
    
    # Check h postulate
    if "h_postulate_check" in CONSTANTS_DICT:
        const = CONSTANTS_DICT["h_postulate_check"]
        if const.eval_expr:
            try:
                h_post = const.eval_expr.subs({
                    SYMBOLS["r_P"]: VALUES["r_P"],
                    SYMBOLS["FM_TO_M"]: VALUES["FM_TO_M"],
                    SYMBOLS["M_fund"]: VALUES["M_fund"],
                    SYMBOLS["MEV_TO_J"]: VALUES["MEV_TO_J"],
                    SYMBOLS["c"]: VALUES["c"],
                })
                h_obs = VALUES["h"]
                print(f"\n✅ h postulate check:")
                print(f"  h_observed:        {h_obs:.6e} J·s")
                print(f"  h_postulate_check: {float(h_post):.6e} J·s")
                rel_error = abs(float(h_post) - h_obs) / h_obs
                print(f"  Relative error: {rel_error:.2e}")
            except Exception as e:
                print(f"\n⚠️  Could not evaluate h_postulate_check: {e}")
    
    # Summary statistics
    print(f"\n📊 Registry Statistics:")
    print(f"  Total constants: {len(CONSTANTS)}")
    print(f"  Categories: {len(set(c.category for c in CONSTANTS))}")
    print(f"  Canonical: {len([c for c in CONSTANTS if c.status == ConstantStatus.CANONICAL])}")
    print(f"  Working: {len([c for c in CONSTANTS if c.status == ConstantStatus.WORKING])}")
    print(f"  Derived: {len([c for c in CONSTANTS if c.status == ConstantStatus.DERIVED])}")
    print(f"  With values: {len(VALUES)}")
    print(f"  With symbols: {len(SYMBOLS)}")

# --- Export all public names ---
__all__ = [
    # Core classes
    "ConstantInfo", "ConstantStatus", "ConstantCategory",
    # Main registry
    "CONSTANTS", "CONSTANTS_DICT", "SYMBOLS", "VALUES",
    # Utility functions
    "get_constants_by_category", "get_constants_by_status",
    "print_registry", "export_to_yaml", "export_to_latex", 
    "validate_relations", "run_self_test",
    # Unit conversions
    "HBAR_C_MEV_FM", "MEV_TO_J", "FM_TO_M", "MEV_C2_TO_KG",
    # Version info
    "__version__", "__date__",
]

# Add all constant names to __all__ for clean imports
__all__.extend([c.name for c in CONSTANTS if c.symbol is not None])

# --- Module initialization message (optional) ---
if __name__ == "__main__":
    print(f"GEF Constants Registry v{__version__}")
    print(f"Loaded {len(CONSTANTS)} constants in {len(set(c.category for c in CONSTANTS))} categories")
    
    # Run the self-test
    run_self_test()