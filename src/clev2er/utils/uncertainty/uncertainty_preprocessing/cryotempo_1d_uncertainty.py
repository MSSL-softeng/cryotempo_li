#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Purpose: Convert (cs2-is2)dh results to 1d slope table

Outputs  :
 - 1d uncertainty table

Author: Alan Muir (MSSL/UCL)
Date: 2022
Copyright: UCL/MSSL/CPOM. Not to be used outside CPOM/MSSL without permission of author

History:
"""

# ------------------------------------------------------------------------------------------------
# Package Imports
# ------------------------------------------------------------------------------------------------

import argparse  # for command line arguments
import sys  # system functions

import numpy as np  # numerical operations
from astropy.stats import median_absolute_deviation
from matplotlib import pyplot as plt
from matplotlib import use as use_headless
from scipy import stats

from clev2er.utils.slopes.slopes import Slopes

# run headless if outputs are image files, rather than display
use_headless("Agg")


def main():
    """main function for tool"""
    # ---------------------------------------------------------------------------------------------
    # Process Command Line Arguments
    # ---------------------------------------------------------------------------------------------

    # initiate the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--ais", "-a", help="process for AIS", action="store_const", const=1)
    parser.add_argument("--gis", "-g", help="process for GIS", action="store_const", const=1)

    parser.add_argument(
        "--dh_file",
        "-f",
        help="input p2p diff file in npz format",
        required=True,
    )
    parser.add_argument(
        "--outdir",
        "-o",
        help="output directory for uncertainty tables and plots",
        required=True,
    )

    # read arguments from the command line
    args = parser.parse_args()

    # Check we have some command line args
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    # ---------------------------------------------------------------------------------------------
    # Test  Command Line Arguments
    # ---------------------------------------------------------------------------------------------

    if not args.ais and not args.gis:
        sys.exit("--ais OR --gis must be used ")

    # ---------------------------------------------------------------------------------------------
    # Configurable Processing Parameters
    # ---------------------------------------------------------------------------------------------

    output_dir = args.outdir

    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    #  Load dh data for this year and region for gt2lgt2r
    # ---------------------------------------------------------------------------------------------

    if args.ais:
        area_id = "antarctica_is"
    else:
        area_id = "greenland_is"

    print("Reading cs2-is2 dh data...")
    dh_data = np.load(args.dh_file, allow_pickle=True)

    lats = dh_data.get("lats")
    lons = dh_data.get("lons")
    dh = dh_data.get("dh")
    print("Number of dh records ", dh.size)

    # -------------------------------------------------------------------------------
    #  Load Slope data for Greenland or Antarctica
    # -------------------------------------------------------------------------------

    if args.gis:
        thisslope = Slopes("awi_grn_2013_1km_slopes")
    else:
        thisslope = Slopes("cpom_ant_2018_1km_slopes")

    # -------------------------------------------------------------------------------
    #  Interpolate slope grid at crossover locations
    # -------------------------------------------------------------------------------

    print("interpolating slope values..")
    slopes = thisslope.interp_slopes(lats, lons, xy_is_latlon=True)

    # -------------------------------------------------------------------------------
    #  Create slope table
    # -------------------------------------------------------------------------------

    min_slope = 0.0
    max_slope = 2.0
    number_of_bins = 20
    bin_width = (max_slope - min_slope) / number_of_bins
    slope_bands = np.arange(min_slope, max_slope + bin_width, bin_width)
    # Create the slope bins (empty arrays for each slope band)
    dh_bins = [[] for i in range(number_of_bins)]

    print("bin_width ", bin_width)
    print("len(dh_bins) ", len(dh_bins))

    # -------------------------------------------------------------------------------
    # bin the dh values in to slope bands
    # -------------------------------------------------------------------------------

    print(f"Binning dh values in to {number_of_bins} band of slope..")
    for i, val in enumerate(dh):
        if np.isnan(slopes[i]):
            continue
        if slopes[i] < min_slope:
            continue
        if slopes[i] >= max_slope:
            continue
        index = int(slopes[i] / bin_width)
        dh_bins[index].append(val)

    # --------------------------------------------------------------------------------
    # Create arrays to store stats
    # --------------------------------------------------------------------------------

    n_vals_raw = np.full(number_of_bins, np.nan)
    n_vals = np.full(number_of_bins, np.nan)

    mean_dh_vals_raw = np.full(number_of_bins, np.nan)
    mean_dh_vals = np.full(number_of_bins, np.nan)
    mean_abs_dh_vals_raw = np.full(number_of_bins, np.nan)
    mean_abs_dh_vals = np.full(number_of_bins, np.nan)

    sigma = np.full(number_of_bins, np.nan)
    sigma_raw = np.full(number_of_bins, np.nan)

    mad_raw = np.full(number_of_bins, np.nan)
    mad = np.full(number_of_bins, np.nan)

    median_dh_raw = np.full(number_of_bins, np.nan)
    median_dh = np.full(number_of_bins, np.nan)

    median_abs_dh_raw = np.full(number_of_bins, np.nan)
    median_abs_dh = np.full(number_of_bins, np.nan)

    rms_dh_raw = np.full(number_of_bins, np.nan)
    rms_dh = np.full(number_of_bins, np.nan)

    abs_rms_dh_raw = np.full(number_of_bins, np.nan)
    abs_rms_dh = np.full(number_of_bins, np.nan)

    # --------------------------------------------------------------------------------
    # Calculate stats for each bin
    # --------------------------------------------------------------------------------

    for i in range(len(dh_bins)):  # pylint: disable=consider-using-enumerate
        # Calculate mean and sigma of dh values in bin
        mean_abs_dh_vals_raw[i] = np.nanmean(np.abs(dh_bins[i]))
        median_abs_dh_raw[i] = np.nanmedian(np.abs(dh_bins[i]))

        mean_dh_vals_raw[i] = np.nanmean(dh_bins[i])
        sigma_raw[i] = np.std(dh_bins[i], ddof=1)
        n_vals_raw[i] = len(dh_bins[i])
        mad_raw[i] = median_absolute_deviation(dh_bins[i])
        median_dh_raw[i] = np.nanmedian(dh_bins[i])
        rms_dh_raw[i] = np.sqrt(np.mean(np.asarray(dh_bins[i]) ** 2))
        abs_rms_dh_raw[i] = np.sqrt(np.mean(np.asarray(np.abs(dh_bins[i])) ** 2))

        # Remove 3sigma outliers
        dh_vals_in_bin = np.asarray(dh_bins[i])
        z = np.abs(stats.zscore(dh_vals_in_bin))
        idx_good = np.where(z < 3)[0]
        if idx_good.size > 0:
            dh_bins[i] = dh_vals_in_bin[idx_good].tolist()

        # re-calculate stats after sigma filter
        mean_abs_dh_vals[i] = np.nanmean(np.abs(dh_bins[i]))
        median_abs_dh[i] = np.nanmedian(np.abs(dh_bins[i]))
        mean_dh_vals[i] = np.nanmean(dh_bins[i])
        sigma[i] = np.std(dh_bins[i], ddof=1)
        n_vals[i] = len(dh_bins[i])
        mad[i] = median_absolute_deviation(dh_bins[i])
        median_dh[i] = np.nanmedian(dh_bins[i])
        rms_dh[i] = np.sqrt(np.mean(np.asarray(dh_bins[i]) ** 2))
        abs_rms_dh[i] = np.sqrt(np.mean(np.asarray(np.abs(dh_bins[i])) ** 2))

    # ----------------------------------------
    # Save Stat Slope table for use in Step-4
    # ----------------------------------------

    # Using unfiltered MAD as slope stat for uncertainty table
    uncertainty_table = median_abs_dh_raw  #  was mad_raw in phase 1 with xovers

    # calculate weighted linear fit to MAD

    # linearfit=np.polynomial.polynomial.Polynomial.fit(slope_bands[0:20], uncertainty_table[0:20],
    # 1, w=n_vals_raw[0:20])
    # linearfit=np.polynomial.polynomial.Polynomial.fit(slope_bands[0:20],
    # uncertainty_table[0:20], 1)

    # uncertainty_table=linearfit(slope_bands[0:-1])

    # Stage4 reads:
    # /raid6/cryo-tempo/land_ice/step3/data/uncertainty_tables/antarctica_uncertainty_from_is2.npz
    # /raid6/cryo-tempo/land_ice/step3/data/uncertainty_tables/greenland_uncertainty_from_is2.npz

    outfile = output_dir + "/antarctica_1d_uncertainty_from_is2_d001.npz"
    if args.gis:
        outfile = output_dir + "/greenland_1d_uncertainty_from_is2_d001.npz"

    np.savez(
        outfile,
        uncertainty_table=uncertainty_table,
        min_slope=min_slope,
        max_slope=max_slope,
        number_of_bins=number_of_bins,
        slope_bands=slope_bands,
        mean_abs_dh_vals=mean_abs_dh_vals,
        mean_abs_dh_vals_raw=mean_abs_dh_vals_raw,
        median_dh=median_dh,
        median_dh_raw=median_dh_raw,
        sigma=sigma,
        sigma_raw=sigma_raw,
        mad_raw=mad_raw,
        area_id=area_id,
        n_vals=n_vals,
        n_vals_raw=n_vals_raw,
        rms_dh=rms_dh,
        rms_dh_raw=rms_dh_raw,
        abs_rms_dh=abs_rms_dh,
        abs_rms_dh_raw=abs_rms_dh_raw,
    )
    print("Saved uncertainty table to: ", outfile)

    # --------------------------------------------------------------------------------
    # Create plots of stats (in headless mode)
    # --------------------------------------------------------------------------------

    # -----------------------------------
    # Mean abs(dH) vs slope band
    # -----------------------------------
    outfile = f"{output_dir}/mean_abs_dh_vs_slope_{area_id}.png"

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(
        slope_bands[0:-1],
        mean_abs_dh_vals,
        color="b",
        marker=".",
        label="Mean |dH| after 3sigma filter",
    )
    ax.plot(
        slope_bands[0:-1],
        mean_abs_dh_vals_raw,
        color="lightblue",
        marker="*",
        linestyle="dashed",
        label="Mean |dH| ",
    )
    plt.xlabel("Slope (degs)")
    plt.ylabel("Mean |dH| (m)")
    plt.title(area_id)
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(slope_bands[0:-1], n_vals, color="lightgreen", marker=".", label="Number of dH")
    ax2.set_ylabel("Number of dH Measurements", color="lightgreen", fontsize=14)
    ax2.fill_between(slope_bands[0:-1], n_vals, color="lightgreen", alpha=0.1)
    fig.tight_layout(pad=3.0)
    plt.savefig(outfile, bbox_inches="tight")
    print("Saved plot to: ", outfile)

    # -----------------------------------
    # Sigma vs slope band
    # -----------------------------------

    outfile = f"{output_dir}/sigma_vs_slope_{area_id}.png"

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(
        slope_bands[0:-1],
        sigma,
        color="b",
        marker=".",
        label="Sigma of dH after 3sigma filter",
    )
    ax.plot(
        slope_bands[0:-1],
        sigma_raw,
        color="lightblue",
        marker="*",
        linestyle="dashed",
        label="Sigma of dH ",
    )
    ax.plot(
        slope_bands[0:-1],
        mad_raw,
        color="pink",
        marker="*",
        linestyle="dashed",
        label="MAD of dH ",
    )
    plt.xlabel("Slope (degs)")
    plt.ylabel("Sigma (m)")
    plt.title(area_id)
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(slope_bands[0:-1], n_vals, color="lightgreen", marker=".", label="Number of dH")
    ax2.set_ylabel("Number of dH Measurements", color="lightgreen", fontsize=14)
    ax2.fill_between(slope_bands[0:-1], n_vals, color="lightgreen", alpha=0.1)
    fig.tight_layout(pad=3.0)
    plt.savefig(outfile, bbox_inches="tight")
    print("Saved plot to: ", outfile)

    # calculate different fits for testing purposes

    c = np.polynomial.polynomial.Polynomial.fit(slope_bands[0:-1], mad_raw, 1, w=n_vals_raw)
    x = slope_bands[0:-1]
    newy = c(x)

    # c2 = np.polynomial.polynomial.Polynomial.fit(slope_bands[0:-1], mad_raw, 2, w=n_vals_raw)

    # c3 = np.polynomial.polynomial.Polynomial.fit(slope_bands[0:-1], mad_raw, 2)

    c4 = np.polynomial.polynomial.Polynomial.fit(slope_bands[0:-1], mad_raw, 4)
    newy4 = c4(x)

    # -----------------------------------
    # MAD vs slope band
    # -----------------------------------

    outfile = f"{output_dir}/mad_vs_slope_{area_id}.png"

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(
        slope_bands[0:-1],
        mad_raw,
        color="black",
        marker="*",
        linestyle="dashed",
        label="MAD of dH ",
    )

    ax.plot(
        slope_bands[0:-1],
        newy,
        color="red",
        marker=".",
        linestyle="dashed",
        label="linear fit",
    )
    # ax.plot(slope_bands[0:-1],newy2,color='orange',marker='*',linestyle='dashed',
    #       label='poly fit')
    # ax.plot(slope_bands[0:-1],newy3,color='yellow',marker='*',linestyle='dashed',
    #       label='poly fit')
    ax.plot(
        slope_bands[0:-1],
        newy4,
        color="brown",
        marker=".",
        linestyle="dashed",
        label="poly fit4",
    )

    plt.xlabel("Slope (degs)")
    plt.ylabel("MAD of dH (m)")
    plt.title(area_id)
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(slope_bands[0:-1], n_vals, color="lightgreen", marker=".", label="Number of dH")
    ax2.set_ylabel("Number of dH Measurements", color="lightgreen", fontsize=14)
    ax2.fill_between(slope_bands[0:-1], n_vals, color="lightgreen", alpha=0.1)
    fig.tight_layout(pad=3.0)
    plt.savefig(outfile, bbox_inches="tight")
    print("Saved plot to: ", outfile)

    # -----------------------------------
    # uncertainty vs slope band
    # -----------------------------------

    outfile = f"{output_dir}/uncertainty_vs_slope_{area_id}.png"

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.plot(
        slope_bands[0:-1],
        uncertainty_table,
        color="black",
        marker="*",
        linestyle="dashed",
        label="Uncertainty (m) ",
        zorder=1,
    )

    plt.xlabel("Slope (degs)")
    plt.ylabel("Uncertainty (m)")
    plt.title(area_id)
    ax.legend(loc="upper center")
    ax2 = ax.twinx()
    ax2.plot(slope_bands[0:-1], n_vals, color="lightgreen", marker=".", label="Number of dH")
    ax2.set_ylabel("Number of dH Measurements", color="lightgreen", fontsize=14)
    ax2.fill_between(slope_bands[0:-1], n_vals, color="lightgreen", alpha=0.1)
    fig.tight_layout(pad=3.0)
    plt.savefig(outfile, bbox_inches="tight")
    print("Saved plot to: ", outfile)


if __name__ == "__main__":
    main()
