"""
 pytest unit tests for : cpom/altimetry/level2/cs2/retrackers/cs2_sin_max_coherence_retracker:
 retrack_cs2_sin_max_coherence_retracker()
"""

import os

import numpy as np
import pytest

from clev2er.utils.cs2.retrackers.cs2_sin_max_coherence_retracker import (
    refine_coherence_peak,
    retrack_cs2_sin_max_coherence,
)

# pylint: disable=R0801
# pylint: disable=too-many-arguments

# Set Markers which apply to whole file
pytestmark = [pytest.mark.sin, pytest.mark.mc]


# Set common inputs used in tests
@pytest.fixture
def lrm_file():
    """fixture

    Returns:
        str: path of LRM L1b file
    """
    return (
        os.environ["CLEV2ER_BASE_DIR"]
        + "/testdata/cs2/l1bfiles/"
        + "CS_OFFL_SIR_LRM_1B_20190504T122726_20190504T123244_D001.nc"
    )


@pytest.fixture
def sin_file():
    """fixture

    Returns:
        str: path of SIN L1b file
    """
    return (
        os.environ["CLEV2ER_BASE_DIR"] + "/testdata/cs2/l1bfiles/"
        "CS_OFFL_SIR_SIN_1B_20190504T122546_20190504T122726_D001.nc"
    )


# ---------------------------------------------------------------------------------------------
# SARin Retracker Test
#   - tests that the function runs with a sample SIN file
#   - tests that the number of retracker failures returned == 0 as expected fot this SIN file
# ---------------------------------------------------------------------------------------------

# Test the MC retracker runs without error with a sample SIN file on all waveforms


def test_retrack_cs2_sin_max_coherence(sin_file):  # pylint: disable=W0621
    """test of retrack_cs2_sin_max_coherence

    Args:
        sin_file (str): path
    """
    # Run the Retracker
    (
        _,  # dr_bin_tcog,
        _,  # dr_meters_tcog,
        _,  # leading_edge_start,
        _,  # leading_edge_stop,
        _,  # pwr_at_rtrk_point,
        coherence_at_rtrk_point,
        n_retracker_failures,
        _,  # retrack_flags,
    ) = retrack_cs2_sin_max_coherence(
        sin_file  # ,plot_flag=True \
        # ,debug_flag=True
    )

    assert n_retracker_failures == 0

    # Check coherence at retracking point is valid for all successfully retracked waveforms
    valid_coherence = coherence_at_rtrk_point[np.isfinite(coherence_at_rtrk_point)]
    assert valid_coherence.size > 0
    assert np.all(valid_coherence >= 0.0)


# ---------------------------------------------------------------------------------------------
# Sub-bin location of the coherence maximum (F018 evolution)
# ---------------------------------------------------------------------------------------------


def test_refine_coherence_peak_none_is_whole_bin():
    """default method must reproduce the whole-bin argmax exactly (E001 behaviour)"""
    coherence = np.array([0.1, 0.3, 0.9, 0.4, 0.2])
    assert refine_coherence_peak(coherence, 2) == 2.0
    assert refine_coherence_peak(coherence, 2, method="none") == 2.0


def test_refine_coherence_peak_parabolic_symmetric():
    """a symmetric peak has its maximum on the sample, so must not be moved"""
    coherence = np.array([0.1, 0.5, 0.9, 0.5, 0.1])
    assert refine_coherence_peak(coherence, 2, method="parabolic") == pytest.approx(2.0)


def test_refine_coherence_peak_parabolic_shifts_towards_higher_neighbour():
    """an asymmetric peak must move towards the higher of the two neighbours"""
    # left neighbour higher -> peak moves left
    left_heavy = np.array([0.1, 0.8, 0.9, 0.2, 0.1])
    assert refine_coherence_peak(left_heavy, 2, method="parabolic") < 2.0
    # right neighbour higher -> peak moves right
    right_heavy = np.array([0.1, 0.2, 0.9, 0.8, 0.1])
    assert refine_coherence_peak(right_heavy, 2, method="parabolic") > 2.0


def test_refine_coherence_peak_parabolic_recovers_known_offset():
    """sampling a known parabola must recover its vertex"""
    true_peak = 2.3
    bins = np.arange(5.0)
    coherence = 0.9 - 0.1 * (bins - true_peak) ** 2
    assert refine_coherence_peak(coherence, 2, method="parabolic") == pytest.approx(true_peak)


def test_refine_coherence_peak_parabolic_bounded_to_half_bin():
    """the parabolic estimator must never move the point by more than half a bin

    Includes triples where the centre is NOT the maximum, which occur in practice
    because the MC argmax is taken within the leading edge search window and a
    neighbouring bin outside that window can be higher.
    """
    rng = np.random.default_rng(42)
    for _ in range(2000):
        coherence = rng.random(3)
        offset = refine_coherence_peak(coherence, 1, method="parabolic") - 1.0
        assert abs(offset) <= 0.5


def test_refine_coherence_peak_parabolic_declines_when_not_a_maximum():
    """where the samples do not bracket a maximum the whole bin must be kept

    Refining here would extrapolate outside the sampled bins, and a near-zero
    curvature makes the vertex formula diverge. Pinning such cases to the +/-0.5
    boundary instead would relocate the retracking point rather than refine it.
    """
    # centre lower than a neighbour: on a rising run, not a peak
    assert refine_coherence_peak(np.array([0.2, 0.5, 0.9]), 1, method="parabolic") == 1.0
    # flat: no curvature, formula would divide by zero
    assert refine_coherence_peak(np.array([0.5, 0.5, 0.5]), 1, method="parabolic") == 1.0
    # near-zero curvature with an asymmetric pair: vertex diverges
    assert (
        refine_coherence_peak(np.array([0.9, 0.9 + 1e-13, 0.9 + 2e-13]), 1, method="parabolic")
        == 1.0
    )


def test_refine_coherence_peak_at_array_edge_is_not_refined():
    """a peak with no neighbour on one side has no sub-bin information"""
    coherence = np.array([0.9, 0.5, 0.3, 0.1])
    for method in ("parabolic", "spline"):
        assert refine_coherence_peak(coherence, 0, method=method) == 0.0
        assert refine_coherence_peak(coherence, coherence.size - 1, method=method) == 3.0


def test_refine_coherence_peak_spline_recovers_known_offset():
    """the spline must also find the vertex of a smooth peak"""
    true_peak = 5.3
    bins = np.arange(11.0)
    coherence = 0.9 - 0.01 * (bins - true_peak) ** 2
    assert refine_coherence_peak(coherence, 5, method="spline") == pytest.approx(
        true_peak, abs=0.02
    )


def test_refine_coherence_peak_spline_is_not_bounded():
    """unlike the parabolic fit, the spline can move the point beyond half a bin

    This is the reason 'parabolic' is the F018 default: on a flat, asymmetric top the
    spline overshoot relocates the retracking point rather than refining it.
    """
    coherence = np.array([0.1, 0.2, 0.3, 0.88, 0.9, 0.89, 0.89, 0.5, 0.2])
    offset = refine_coherence_peak(coherence, 4, method="spline") - 4.0
    assert abs(offset) > 0.5


def test_refine_coherence_peak_rejects_unknown_method():
    """an unrecognised method must be an error, not a silent no-op"""
    with pytest.raises(ValueError):
        refine_coherence_peak(np.array([0.1, 0.9, 0.1]), 1, method="oversample")


def test_retrack_subbin_method_none_matches_default(sin_file):  # pylint: disable=W0621
    """the E001-equivalent path must be bit-identical to the shipped default"""
    dr_bin_default = retrack_cs2_sin_max_coherence(sin_file)[0]
    dr_bin_none = retrack_cs2_sin_max_coherence(sin_file, coherence_subbin_method="none")[0]
    np.testing.assert_array_equal(dr_bin_default, dr_bin_none)
    # and the default must be whole-bin, ie no fractional part anywhere
    finite = dr_bin_default[np.isfinite(dr_bin_default)]
    assert finite.size > 0
    np.testing.assert_array_equal(finite, np.rint(finite))


def test_retrack_subbin_parabolic_is_subbin_and_bounded(sin_file):  # pylint: disable=W0621
    """parabolic refinement must produce fractional bins within half a bin of E001"""
    dr_bin_none = retrack_cs2_sin_max_coherence(sin_file, coherence_subbin_method="none")[0]
    dr_bin_parabolic = retrack_cs2_sin_max_coherence(sin_file, coherence_subbin_method="parabolic")[
        0
    ]

    finite = np.isfinite(dr_bin_none) & np.isfinite(dr_bin_parabolic)
    assert finite.sum() > 0
    # the same waveforms must succeed or fail either way
    np.testing.assert_array_equal(np.isfinite(dr_bin_none), np.isfinite(dr_bin_parabolic))

    shift = dr_bin_parabolic[finite] - dr_bin_none[finite]
    assert np.all(np.abs(shift) <= 0.5)
    # the refinement must actually do something on a real file
    assert np.any(np.abs(shift) > 0.0)


def test_retrack_subbin_does_not_change_power_or_coherence(sin_file):  # pylint: disable=W0621
    """only the retracking point is refined; backscatter inputs must be untouched"""
    _, _, _, _, pwr_none, coh_none, fails_none, _ = retrack_cs2_sin_max_coherence(
        sin_file, coherence_subbin_method="none"
    )
    _, _, _, _, pwr_parab, coh_parab, fails_parab, _ = retrack_cs2_sin_max_coherence(
        sin_file, coherence_subbin_method="parabolic"
    )
    np.testing.assert_array_equal(pwr_none, pwr_parab)
    np.testing.assert_array_equal(coh_none, coh_parab)
    assert fails_none == fails_parab


# Test the MC retracker runs without error with a sample LRM file on all waveforms
#   - expected to fail as not a SIN file
@pytest.mark.xfail()
def test_retrack_cs2_sin_max_coherence_with_lrm(lrm_file):  # pylint: disable=W0621
    """_summary_

    Args:
        lrm_file (str): path
    """
    # Run the Retracker
    (
        _,  # dr_bin_tcog,
        _,  # dr_meters_tcog,
        _,  # leading_edge_start,
        _,  # leading_edge_stop,
        _,  # pwr_at_rtrk_point,
        _,  # coherence_at_rtrk_point,
        _,  # n_retracker_failures,
        _,  # retrack_flags,
    ) = retrack_cs2_sin_max_coherence(
        lrm_file  # ,plot_flag=True \
        # ,debug_flag=True
    )


# Test retracking of LRM waveforms which should fail: index [713,714,715,717] where noise floor is
#  exceeded
#   - test returned n_retracker_failures should be 1
#   -  retrack_flags[measurement_index][0] should be 1 to indicate noise floor exceeded


# Test is repeated for each index
@pytest.mark.parametrize(
    "measurement_index, expected_success, expected_retracking_bin, expected_leading_edge_start,"
    " expected_leading_edge_end",
    [(0, 1, 204.0, 199.42, 207.99)],
)
def test_retrack_cs2_sin_max_coherence_at_index(
    sin_file,  # pylint: disable=W0621
    measurement_index,
    expected_success,
    expected_retracking_bin,
    expected_leading_edge_start,
    expected_leading_edge_end,
):
    """test of retrack_cs2_sin_max_coherence at index

    Args:
        sin_file (str): path
        measurement_index (int): _description_
        expected_success (int): _description_
        expected_retracking_bin (float): _description_
        expected_leading_edge_start (float): _description_
        expected_leading_edge_end (float): _description_
    """
    ref_bin_ind_sin = 512

    # Run the Retrackers
    (
        dr_bin_mc,
        _,  # dr_meters_mc,
        leading_edge_start,
        leading_edge_stop,
        _,  # pwr_at_rtrk_point,
        _,  # coherence_at_rtrk_point,
        n_retracker_failures,
        retrack_flags,
    ) = retrack_cs2_sin_max_coherence(
        sin_file, measurement_index=measurement_index, plot_flag=False
    )

    print(
        "dr_bin_mc[measurement_index]+ref_bin_ind_sin= ",
        dr_bin_mc[measurement_index] + ref_bin_ind_sin,
    )
    print(
        "leading_edge_start[measurement_index][0]= ",
        leading_edge_start[measurement_index][0],
    )
    print(
        "leading_edge_stop[measurement_index][0]= ",
        leading_edge_stop[measurement_index][0],
    )

    if expected_success:
        # Test that retracking was successful
        assert n_retracker_failures == 0
        # Check that the retracking point is at the expected bin index
        assert np.isclose(
            dr_bin_mc[measurement_index] + ref_bin_ind_sin,
            expected_retracking_bin,
            0.01,
        )
        # Check the leading edge start is at expected bin, to 2 decimal places
        assert np.isclose(
            leading_edge_start[measurement_index][0],
            expected_leading_edge_start,
            atol=0.01,
        )
        # Check the leading edge stop is at expected bin, to 2 decimal places
        assert np.isclose(
            leading_edge_stop[measurement_index][0],
            expected_leading_edge_end,
            atol=0.01,
        )

    else:
        # Test that retracking failed
        assert n_retracker_failures == 1
        # Test that retracking failed due to noise threshold being exceeded
        assert retrack_flags[measurement_index][0] == 1
        # Retracking bin should be Nan
        assert np.isnan(dr_bin_mc[measurement_index])
        # Leadinge Edge start bin should be Nan
        assert np.isnan(leading_edge_start[measurement_index][0])
