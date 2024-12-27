"""clev2er.algorithms.templates.alg_uncertainty"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=E0611

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.roughness.roughness import Roughness
from clev2er.utils.slopes.slopes import Slopes
from clev2er.utils.uncertainty.calc_uncertainty import calc_uncertainty

# -------------------------------------------------

# pylint config
# Similar lines in 2 files, pylint: disable=R0801
# Too many return statements, pylint: disable=R0911
# pylint: disable=too-many-instance-attributes


def get_binned_values(
    slope_values: np.ndarray,
    roughness_values: np.ndarray,
    binned_table: pd.DataFrame,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
) -> np.ndarray:
    """Retrieve the median absolute elevation difference for arrays of slope and roughness values.

    Args:
        slope_values (np.ndarray): Array of slope values for which to retrieve median differences.
        roughness_values (np.ndarray): Array of roughness values for which to retrieve
                                       median differences.
        binned_table (pd.DataFrame): A pivot table of binned median absolute elevation differences.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.

    Returns:
        np.ndarray: An array of median absolute elevation differences corresponding to the
                    input slope and roughness pairs.
    """
    # Convert slope_values and roughness_values to numpy arrays
    slope_values = np.asarray(slope_values)
    roughness_values = np.asarray(roughness_values)

    # Find the slope bin indices for the array of slope_values
    slope_bin_indices = np.digitize(slope_values, slope_bins) - 1
    slope_bin_indices = np.clip(
        slope_bin_indices, 0, len(slope_bins) - 2
    )  # Ensure indices are within range

    # Find the roughness bin indices for the array of roughness_values
    roughness_bin_indices = np.digitize(roughness_values, roughness_bins) - 1
    roughness_bin_indices = np.clip(
        roughness_bin_indices, 0, len(roughness_bins) - 2
    )  # Ensure indices are within range

    # Convert bin labels to row and column indices in the DataFrame
    row_indices = [binned_table.index.get_loc(slope_bins[idx]) for idx in slope_bin_indices]
    col_indices = [
        binned_table.columns.get_loc(roughness_bins[idx]) for idx in roughness_bin_indices
    ]

    # Retrieve the values using numpy indexing on the DataFrame values
    values = binned_table.values[row_indices, col_indices]

    return values


class Algorithm(BaseAlgorithm):
    """**Algorithm to retrieve elevation uncertainty from (CS2-IS2) derived uncertainty table and
    surface slope at each measurement**

    **Contribution to shared_dict**
        -shared_dict["uncertainty"] : (np.ndarray) uncertainty at each track location

    CLEV2ER Algorithm: inherits from BaseAlgorithm

    BaseAlgorithm __init__(config,thislog)
        Args:
            config: Dict[str, Any]: chain configuration dictionary
            thislog: logging.Logger | None: initial logger instance to use or
                                            None (use root logger)
    """

    # Note: __init__() is in BaseAlgorithm. See required parameters above
    # init() below is called by __init__() at a time dependent on whether
    # sequential or multi-processing mode is in operation

    def init(self) -> Tuple[bool, str]:
        """Algorithm initialization

        Add steps in this function that are run once at the beginning of the chain
        (for example loading a DEM or Mask)

        Returns:
            (bool,str) : success or failure, error string

        Raises:
            KeyError : keys not in config
            FileNotFoundError :
            OSError :

        Note: raise and Exception rather than just returning False
        """
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # Add initialization steps here

        self.uncertainty_table_antarctica = ""
        self.uncertainty_table_greenland = ""
        self.ut_table_grn = None
        self.ut_min_slope_grn = 0.0
        self.ut_max_slope_grn = 10.0
        self.ut_number_of_bins_grn = None
        self.ut_table_ant = None
        self.ut_min_slope_ant = 0.0
        self.ut_max_slope_ant = 10.0
        self.ut_number_of_bins_ant = None

        # -------------------------------------------------------------------------
        # Load uncertainty tables
        # -------------------------------------------------------------------------

        # Get uncertainty table files
        if "uncertainty_tables" not in self.config:
            raise KeyError("uncertainty_tables not in config")
        if "base_dir" not in self.config["uncertainty_tables"]:
            raise KeyError("uncertainty_tables.base_dir not in config")

        self.uncertainty_table_antarctica = (
            f"{self.config['uncertainty_tables']['base_dir']}/"
            "ant_2d_uncertainty_table_bilinear_median.pickle"
        )
        self.uncertainty_table_greenland = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "greenland_1d_uncertainty_from_is2_d001.npz"
        )

        if not os.path.isfile(self.uncertainty_table_antarctica):
            raise FileNotFoundError(
                f"Antarctic uncertainty table {self.uncertainty_table_antarctica}" " not found"
            )
        if not os.path.isfile(self.uncertainty_table_greenland):
            raise FileNotFoundError(
                f"Greenland uncertainty table {self.uncertainty_table_greenland}" " not found"
            )

        ut_grn_data = np.load(self.uncertainty_table_greenland, allow_pickle=True)

        keys = ["uncertainty_table", "min_slope", "max_slope", "number_of_bins"]
        for key in keys:
            if key not in ut_grn_data:
                raise KeyError(
                    f"{key} key not in Greenland uncertainty table"
                    f" {self.uncertainty_table_greenland}"
                )

        self.ut_table_grn = ut_grn_data.get("uncertainty_table")
        self.ut_min_slope_grn = ut_grn_data.get("min_slope")
        self.ut_max_slope_grn = ut_grn_data.get("max_slope")
        self.ut_number_of_bins_grn = ut_grn_data.get("number_of_bins")

        self.ut_table_ant = pd.read_pickle(self.uncertainty_table_antarctica)

        # Define slope bins in degrees (0.1 degree steps from 0 to 2 degrees)
        self.ant_slope_bins = np.arange(0, 2.1, 0.1)

        # Define roughness bins in meters (0.1 m steps from 0 to 2 meters)
        self.ant_roughness_bins = np.arange(0, 2.1, 0.1)

        # Test the data
        if not isinstance(self.ut_table_grn, np.ndarray):
            raise ValueError(f"ut_table_grn is not of type np.ndarray: {type(self.ut_table_grn)}")
        if not isinstance(self.ut_table_ant, pd.core.frame.DataFrame):
            raise ValueError(
                f"ut_table_ant is not of type pd.core.frame.DataFrame: {type(self.ut_table_ant)}"
            )

        self.slope_grn = Slopes("awi_grn_2013_1km_slopes")
        if "grn_only" in self.config and self.config["grn_only"]:
            self.slope_ant = None
            self.roughness_ant = None
        else:
            self.slope_ant = Slopes("rema_100m_900ws_slopes_zarr")
            self.roughness_ant = Roughness("rema_100m_900ws_roughness_zarr")

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        """Main algorithm processing function

        Args:
            l1b (Dataset): input l1b file dataset (constant)
            shared_dict (dict): shared_dict data passed between algorithms

        Returns:
            Tuple : (success (bool), failure_reason (str))
            ie
            (False,'error string'), or (True,'')

        **IMPORTANT NOTE:** when logging within the Algorithm.process() function you must use
        the self.log.info(),error(),debug() logger and NOT log.info(), log.error(), log.debug :

        `self.log.error("your message")`

        """

        # This is required to support logging during multi-processing
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # \/    down the chain in the 'shared_dict' dict     \/
        # -------------------------------------------------------------------

        # Calculate uncertainty from POCA (or nadir if POCA failed) parameters:
        # shared_dict["latitudes"], shared_dict["longitudes"]

        if shared_dict["hemisphere"] == "south":
            if self.slope_ant is not None and self.roughness_ant is not None:
                slope_values = self.slope_ant.interp_slopes(
                    shared_dict["latitudes"],
                    shared_dict["longitudes"],
                    xy_is_latlon=True,
                )
                roughness_values = self.roughness_ant.interp_roughness(
                    shared_dict["latitudes"],
                    shared_dict["longitudes"],
                    xy_is_latlon=True,
                )

                uncertainty = get_binned_values(
                    slope_values,
                    roughness_values,
                    self.ut_table_ant,
                    self.ant_slope_bins,
                    self.ant_roughness_bins,
                )

            else:
                uncertainty = None
        else:
            slopes = self.slope_grn.interp_slopes(
                shared_dict["latitudes"],
                shared_dict["longitudes"],
                xy_is_latlon=True,
            )
            uncertainty = calc_uncertainty(
                slopes,
                self.ut_table_grn,  # type: ignore # already checked type in init
                self.ut_min_slope_grn,
                self.ut_max_slope_grn,
            )

        shared_dict["uncertainty"] = uncertainty

        # Return success (True,'')
        return (True, "")


# No finalize() required for this algorithm
