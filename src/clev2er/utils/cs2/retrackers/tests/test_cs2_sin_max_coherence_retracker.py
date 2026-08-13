"""
 pytest unit tests for : cpom/altimetry/level2/cs2/retrackers/cs2_sin_max_coherence_retracker:
 retrack_cs2_sin_max_coherence_retracker()
"""

import os

import numpy as np
import pytest

from clev2er.utils.cs2.retrackers.cs2_sin_max_coherence_retracker import (
    retrack_cs2_sin_max_coherence,
    smooth_coherence,
)
from clev2er.utils.cs2.retrackers.fastsmooth import fastsmooth

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


# ---------------------------------------------------------------------------------------------
# Coherence smoothing method (baseline-F evolution)
# ---------------------------------------------------------------------------------------------


def test_smooth_coherence_default_is_boxcar():
    """the default must be the baselines B to E001 running mean"""
    wf = np.linspace(0.0, 1.0, 64)
    assert np.array_equal(smooth_coherence(wf, 9), fastsmooth(wf, 9))
    assert np.array_equal(smooth_coherence(wf, 9, method="boxcar"), fastsmooth(wf, 9))


def test_smooth_coherence_rejects_bad_arguments():
    """an unknown method, or a window no wider than the polynomial, must raise"""
    wf = np.linspace(0.0, 1.0, 64)
    with pytest.raises(ValueError, match="unknown coherence smoothing method"):
        smooth_coherence(wf, 9, method="gaussian")
    with pytest.raises(ValueError, match="must be wider than the polynomial order"):
        smooth_coherence(wf, 3, method="savgol", poly_order=3)


def test_savgol_preserves_peaks_better_than_boxcar():
    """the motivation for the evolution: savgol should not flatten a peak as boxcar does

    The retracker takes the argmax of the smoothed coherence, so a smoother that
    preserves peak amplitude and position gives a better determined retracking
    point.
    """
    wf = np.full(64, 0.2)
    wf[32] = 1.0  # a sharp isolated peak
    boxcar = smooth_coherence(wf, 9, method="boxcar")
    savgol = smooth_coherence(wf, 9, method="savgol", poly_order=3)
    assert savgol[32] > boxcar[32], "savgol should retain more of the peak amplitude"
    assert int(np.argmax(savgol)) == 32, "savgol should keep the peak in place"


def test_savgol_poly_order_2_and_3_share_interior_coefficients():
    """even/odd Savitzky-Golay order pairs share their INTERIOR coefficients

    Orders 2 and 3 are an identical filter away from the array ends, so choosing
    3 over 2 has no effect on the retracking point: the leading edge is never
    within half a window of bin 0. They differ only in the first and last
    half-window samples, where savgol_filter does a separate boundary fit.
    Order 4 is a genuinely different filter throughout.
    """
    rng = np.random.default_rng(0)
    wf = rng.uniform(0.0, 1.0, 256)
    half = 9 // 2

    order2 = smooth_coherence(wf, 9, method="savgol", poly_order=2)
    order3 = smooth_coherence(wf, 9, method="savgol", poly_order=3)
    order4 = smooth_coherence(wf, 9, method="savgol", poly_order=4)

    assert np.array_equal(
        order2[half:-half], order3[half:-half]
    ), "orders 2 and 3 must be the same filter in the interior"
    assert not np.allclose(order2, order3), "they differ at the boundary fit"
    assert not np.allclose(
        order3[half:-half], order4[half:-half]
    ), "order 4 must differ in the interior too"


def test_retracker_smoothing_method_changes_retracking(sin_file):  # pylint: disable=W0621
    """switching the smoother moves retracking points but must not fail waveforms

    Args:
        sin_file (str): path
    """
    boxcar = retrack_cs2_sin_max_coherence(sin_file, coherence_smoothing_method="boxcar")
    savgol = retrack_cs2_sin_max_coherence(sin_file, coherence_smoothing_method="savgol")

    assert savgol[6] == boxcar[6], "the smoother should not change the failure count"
    bb = np.asarray(boxcar[0], dtype=float)
    sb = np.asarray(savgol[0], dtype=float)
    assert np.array_equal(
        np.isfinite(bb), np.isfinite(sb)
    ), "the same waveforms should retrack successfully either way"
    both = np.isfinite(bb)
    assert not np.array_equal(bb[both], sb[both]), "the retracking points should differ"
