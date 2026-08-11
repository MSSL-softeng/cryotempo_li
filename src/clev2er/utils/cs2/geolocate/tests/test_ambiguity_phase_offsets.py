"""pytest of clev2er.utils.cs2.geolocate.geolocate_sin.ambiguity_phase_offsets

Baseline-F evolution 3: search multiple 2pi phase ambiguities rather than the
single sign-selected alternative used by baselines B to F010.
"""

import numpy as np
import pytest

from clev2er.utils.cs2.geolocate.geolocate_sin import (
    ambiguity_phase_offsets,
    phase_to_angle,
)

TWO_PI = 2.0 * np.pi


def test_legacy_single_alternative_positive_phase() -> None:
    """num_ambiguities=0 must reproduce the baselines B to F010 behaviour"""
    offsets = ambiguity_phase_offsets(1.0, 0)
    assert offsets == [-TWO_PI], "positive phase used phase - 2pi"


def test_legacy_single_alternative_negative_phase() -> None:
    """the legacy alternative is selected by the sign of the phase"""
    assert ambiguity_phase_offsets(-1.0, 0) == [TWO_PI]
    # phase of exactly zero took the else branch
    assert ambiguity_phase_offsets(0.0, 0) == [TWO_PI]


def test_legacy_alternative_always_crosses_nadir() -> None:
    """the legacy alternative always has the opposite angle sign

    This is the limitation the evolution addresses: the band further out on the
    same side of nadir was never a candidate.
    """
    for phase in (0.5, 1.5, 3.0, -0.5, -1.5, -3.0):
        angle = phase_to_angle(phase)
        (offset,) = ambiguity_phase_offsets(phase, 0)
        alt_angle = phase_to_angle(phase + offset)
        assert np.sign(alt_angle) == -np.sign(angle), (
            f"phase {phase}: legacy alternative should cross nadir, "
            f"angle {angle:.5f} -> {alt_angle:.5f}"
        )


def test_symmetric_search_includes_both_neighbours() -> None:
    """N=1 must test both adjacent bands, not just the one crossing nadir"""
    offsets = ambiguity_phase_offsets(1.0, 1)
    assert offsets == [-TWO_PI, TWO_PI], "N=1 gives k=-1 and k=+1"

    # and unlike the legacy case, the result must not depend on the phase sign
    assert ambiguity_phase_offsets(1.0, 1) == ambiguity_phase_offsets(-1.0, 1)


def test_search_range_scales_with_n() -> None:
    """N bands either side, always excluding the zero offset (the primary)"""
    for num in (1, 2, 3):
        offsets = ambiguity_phase_offsets(0.5, num)
        assert len(offsets) == 2 * num, f"N={num} should give 2N alternatives"
        assert 0.0 not in offsets, "the zero offset is the primary solution"
        expected = [k * TWO_PI for k in range(-num, num + 1) if k != 0]
        assert offsets == expected


def test_one_ambiguity_is_about_1_1_degrees() -> None:
    """sanity check the geometry the search relies on

    If this changes, the DEM may no longer discriminate between candidates.
    """
    step = abs(phase_to_angle(TWO_PI) - phase_to_angle(0.0))
    assert np.degrees(step) == pytest.approx(
        1.1138, abs=1e-3
    ), f"one 2pi should be ~1.1138 deg, got {np.degrees(step):.4f}"
    # and the unambiguous band is half of that either side of nadir
    assert np.degrees(abs(phase_to_angle(np.pi))) == pytest.approx(np.degrees(step) / 2.0, abs=1e-6)
