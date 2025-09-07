import pytest

# Target the canonical constants module
from gef.core import constants as constants
from gef.core.constants import (
    CONSTANTS,
    CONSTANTS_DICT,
    ConstantStatus,
    get_constants_by_status,
    validate_relations,
    VALUES,
)


def test_constants_registry_is_not_empty():
    """Sanity check that the registry exists and has entries."""
    assert CONSTANTS is not None
    assert len(CONSTANTS) > 0


def test_constants_dict_no_duplicates():
    """Ensure there are no duplicate names and dict is consistent with the list."""
    names = [c.name for c in CONSTANTS]
    assert len(names) == len(set(names)), "Duplicate constant names found"

    # Dictionary keys should match the set of names
    assert set(CONSTANTS_DICT.keys()) >= set(names)


def test_get_constants_by_status():
    """Test the utility function for filtering constants by status."""
    expected_status = ConstantStatus.WORKING
    working_constants = get_constants_by_status(expected_status)

    assert isinstance(working_constants, list)
    assert all(c.status == expected_status for c in working_constants)


def test_r_P_derivation_is_self_consistent():
    """r_P should equal (hbar_c / M_fund) in MeV·fm / MeV → fm units."""
    hbar_c = VALUES["hbar_c"]   # MeV·fm
    M_fund = VALUES["M_fund"]   # MeV
    stored_r_P = VALUES["r_P"]  # fm

    calculated_r_P = hbar_c / M_fund
    assert stored_r_P == pytest.approx(calculated_r_P, rel=1e-12)


def test_h_postulate_is_self_consistent():
    """Derived h (2π r_P M_fund / c with unit conversions) should match observed h."""
    h_post_const = CONSTANTS_DICT["h_postulate_check"]
    expr = h_post_const.eval_expr

    subs_values = {
        "r_P": VALUES["r_P"],
        "FM_TO_M": VALUES["FM_TO_M"],
        "M_fund": VALUES["M_fund"],
        "MEV_TO_J": VALUES["MEV_TO_J"],
        "c": VALUES["c"],
    }

    calculated_h = expr.subs(subs_values)
    calculated_h_float = float(calculated_h)

    observed_h = VALUES["h"]
    assert observed_h == pytest.approx(calculated_h_float, rel=1e-6)


def test_validate_relations_finds_no_errors():
    """Self-validation should return an empty dict when all relations are consistent."""
    errors = validate_relations()
    assert not errors, f"Found relation validation errors: {errors}"