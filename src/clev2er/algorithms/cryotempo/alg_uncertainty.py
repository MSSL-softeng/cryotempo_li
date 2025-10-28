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
    power_values: np.ndarray,
    coherence_values: np.ndarray,
    binned_table: pd.DataFrame,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
) -> list:

    """
    Retrieve the calculated metric of elevation difference for given variable values (ie for given
    slope, roughness and power values).

    Args:
        slope_values (np.ndarray): Array of slope values for which to retrieve calculated differences.
        roughness_values (np.ndarray): Array of roughness values for which to retrieve calculated differences.
        power_values (np.ndarray): Array of power values for which to retrieve calculated differences.
        coherence_values (np.ndarray): Array of coherence values for which to retrieve calculated differences.
        binned_table (pd.DataFrame): A pivot table of binned median absolute elevation differences.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        coherence_bins (np.ndarray): Bins to categorise coherence values.

    Returns:
        list: An array of median absolute elevation differences corresponding to the
                    input values.
    """
    ingested_variables = []

    if slope_values is not None:
        slope_values = np.asarray(slope_values)
        # Find the slope bin indices for the array of slope_values
        slope_bin_indices = np.digitize(slope_values, slope_bins) - 1
        slope_bin_indices = np.clip(
            slope_bin_indices, 0, len(slope_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        slope_indices = [slope_bins[idx] for idx in slope_bin_indices]
        ingested_variables.append(slope_indices)

    if roughness_values is not None:
        roughness_values = np.asarray(roughness_values)
        # Find the roughness bin indices for the array of roughness_values
        roughness_bin_indices = np.digitize(roughness_values, roughness_bins) - 1
        roughness_bin_indices = np.clip(
            roughness_bin_indices, 0, len(roughness_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        roughness_indices = [roughness_bins[idx] for idx in roughness_bin_indices]
        ingested_variables.append(roughness_indices)

    if power_values is not None:
        power_values = np.asarray(power_values)
        # Find the power bin indices for the array of power
        power_bin_indices = np.digitize(power_values, power_bins) - 1
        power_bin_indices = np.clip(
            power_bin_indices, 0, len(power_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        power_indices = [power_bins[idx] for idx in power_bin_indices]
        ingested_variables.append(power_indices)

    if coherence_values is not None:
        coherence_values = np.asarray(coherence_values)
        # Find the coherence bin indices for the array of coherence
        coherence_bin_indices = np.digitize(coherence_values, coherence_bins) - 1
        coherence_bin_indices = np.clip(
            coherence_bin_indices, 0, len(coherence_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        coherence_indices = [coherence_bins[idx] for idx in coherence_bin_indices]
        ingested_variables.append(coherence_indices)

    binned_table_df = binned_table.stack(future_stack=True)
    # print(binned_table_df)

    if len(ingested_variables) == 1:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][0] for i in range(len(ingested_variables[0]))]
    if len(ingested_variables) == 2:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][ingested_variables[1][i]] for i in range(len(ingested_variables[0]))]
    if len(ingested_variables) == 3:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][ingested_variables[1][i]][ingested_variables[2][i]] for i in range(len(ingested_variables[0]))]
    if len(ingested_variables) == 4:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][ingested_variables[1][i]][ingested_variables[2][i]][ingested_variables[3][i]] for i in range(len(ingested_variables[0]))]
    
    # Retrieve the values using numpy indexing on the DataFrame values
    # values = [binned_table_df.loc[row_indices[i]][col1_indices[i]][col2_indices[i]][col3_indices[i]][col4_indices[i]] for i in range(len(row_indices))]

    return values


class Algorithm(BaseAlgorithm):
    """**Algorithm to retrieve elevation uncertainty from (CS2-IS2) derived uncertainty table and
    covariate value(s) at each measurement**

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

        self.uncertainty_table_antarctica_sin = ""
        self.uncertainty_table_antarctica_lrm = ""
        self.uncertainty_table_greenland_sin = ""
        self.uncertainty_table_greenland_lrm = ""
        self.ut_table_grn_sin = None
        self.ut_table_grn_lrm = None
        self.ut_table_ant_sin = None
        self.ut_table_ant_lrm = None

        # -------------------------------------------------------------------------
        # Load uncertainty tables
        # -------------------------------------------------------------------------

        # Get uncertainty table files
        if "uncertainty_tables" not in self.config:
            raise KeyError("uncertainty_tables not in config")
        if "base_dir" not in self.config["uncertainty_tables"]:
            raise KeyError("uncertainty_tables.base_dir not in config")

        self.uncertainty_table_antarctica_sin = (
            f"{self.config['uncertainty_tables']['base_dir']}/"
            "antarctica_icesheets_median_metric_pivot_filled_SIN.pickle"
        )
        self.uncertainty_table_greenland_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "greenland_is_median_metric_pivot_filled_SIN.pickle"
        )

        self.uncertainty_table_antarctica_lrm = (
            f"{self.config['uncertainty_tables']['base_dir']}/"
            "antarctica_icesheets_median_metric_pivot_filled_LRM.pickle"
        )
        self.uncertainty_table_greenland_lrm = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "greenland_is_median_metric_pivot_filled_LRM.pickle"
        )

        if not os.path.isfile(self.uncertainty_table_antarctica_sin):
            raise FileNotFoundError(
                f"Antarctic SIN uncertainty table {self.uncertainty_table_antarctica_sin}" " not found"
            )
        if not os.path.isfile(self.uncertainty_table_greenland_sin):
            raise FileNotFoundError(
                f"Greenland SIN uncertainty table {self.uncertainty_table_greenland_sin}" " not found"
            )
        if not os.path.isfile(self.uncertainty_table_antarctica_lrm):
            raise FileNotFoundError(
                f"Antarctic LRM uncertainty table {self.uncertainty_table_antarctica_lrm}" " not found"
            )
        if not os.path.isfile(self.uncertainty_table_greenland_lrm):
            raise FileNotFoundError(
                f"Greenland LRM uncertainty table {self.uncertainty_table_greenland_lrm}" " not found"
            )

        # Read in SIN lookup tables, and bin distribution
        self.ut_table_grn_sin = pd.read_pickle(self.uncertainty_table_greenland_sin)

        # Define slope bins in degrees 
        self.grn_slope_bins_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_slope_bins_greenland_is_SIN.npz"
        )
        self.grn_slope_bins_sin = np.load(self.grn_slope_bins_sin)['the_slope_bins']

        # Define roughness bins in meters 
        self.grn_roughness_bins_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_roughness_bins_greenland_is_SIN.npz"
        )
        self.grn_roughness_bins_sin = np.load(self.grn_roughness_bins_sin)['the_roughness_bins']

        # Define power bins 
        self.grn_power_bins_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_power_bins_greenland_is_SIN.npz"
        )
        self.grn_power_bins_sin = np.load(self.grn_power_bins_sin)['the_power_bins']

        # Define coherence bins 
        self.grn_coherence_bins_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_coherence_bins_greenland_is_SIN.npz"
        )
        self.grn_coherence_bins_sin = np.load(self.grn_coherence_bins_sin)['the_coherence_bins']

        self.ut_table_ant_sin = pd.read_pickle(self.uncertainty_table_antarctica_sin)

        # Define slope bins in degrees 
        self.ant_slope_bins_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_slope_bins_antarctica_icesheets_SIN.npz"
        )
        self.ant_slope_bins_sin = np.load(self.ant_slope_bins_sin)['the_slope_bins']

        # Define roughness bins in meters 
        self.ant_roughness_bins_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_roughness_bins_antarctica_icesheets_SIN.npz"
        )
        self.ant_roughness_bins_sin = np.load(self.ant_roughness_bins_sin)['the_roughness_bins']

        # Define power bins 
        self.ant_power_bins_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_power_bins_antarctica_icesheets_SIN.npz"
        )
        self.ant_power_bins_sin = np.load(self.ant_power_bins_sin)['the_power_bins']

        # Define coherence bins 
        self.ant_coherence_bins_sin = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_coherence_bins_antarctica_icesheets_SIN.npz"
        )
        self.ant_coherence_bins_sin = np.load(self.ant_coherence_bins_sin)['the_coherence_bins']

        # Read in LRM lookup tables, and bin distribution
        self.ut_table_grn_lrm = pd.read_pickle(self.uncertainty_table_greenland_lrm)

        # Define slope bins in degrees 
        self.grn_slope_bins_lrm = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_slope_bins_greenland_is_LRM.npz"
        )
        self.grn_slope_bins_lrm = np.load(self.grn_slope_bins_lrm)['the_slope_bins']

        # Define roughness bins in meters 
        self.grn_roughness_bins_lrm = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_roughness_bins_greenland_is_LRM.npz"
        )
        self.grn_roughness_bins_lrm = np.load(self.grn_roughness_bins_lrm)['the_roughness_bins']

        # Define power bins 
        self.grn_power_bins_lrm = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_power_bins_greenland_is_LRM.npz"
        )
        self.grn_power_bins_lrm = np.load(self.grn_power_bins_lrm)['the_power_bins']

        self.ut_table_ant_lrm = pd.read_pickle(self.uncertainty_table_antarctica_lrm)

        # Define slope bins in degrees 
        self.ant_slope_bins_lrm = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_slope_bins_antarctica_icesheets_LRM.npz"
        )
        self.ant_slope_bins_lrm = np.load(self.ant_slope_bins_lrm)['the_slope_bins']

        # Define roughness bins in meters 
        self.ant_roughness_bins_lrm = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_roughness_bins_antarctica_icesheets_LRM.npz"
        )
        self.ant_roughness_bins_lrm = np.load(self.ant_roughness_bins_lrm)['the_roughness_bins']

        # Define power bins 
        self.ant_power_bins_lrm = (
            f"{str(self.config['uncertainty_tables']['base_dir'])}/"
            "the_power_bins_antarctica_icesheets_LRM.npz"
        )
        self.ant_power_bins_lrm = np.load(self.ant_power_bins_lrm)['the_power_bins']

        # Test the data
        if not isinstance(self.ut_table_grn_sin, pd.core.frame.DataFrame):
            raise ValueError(f"ut_table_grn is not of type pd.core.frame.DataFrame: {type(self.ut_table_grn_sin)}")
        if not isinstance(self.ut_table_ant_sin, pd.core.frame.DataFrame):
            raise ValueError(
                f"ut_table_ant is not of type pd.core.frame.DataFrame: {type(self.ut_table_ant_sin)}"
            )
        if not isinstance(self.ut_table_grn_lrm, pd.core.frame.DataFrame):
            raise ValueError(f"ut_table_grn is not of type pd.core.frame.DataFrame: {type(self.ut_table_grn_lrm)}")
        if not isinstance(self.ut_table_ant_lrm, pd.core.frame.DataFrame):
            raise ValueError(
                f"ut_table_ant is not of type pd.core.frame.DataFrame: {type(self.ut_table_ant_lrm)}"
            )

        self.slope_grn = Slopes("arcticdem_100m_900ws_slopes_zarr")
        self.roughness_grn = Roughness("rema_100m_900ws_roughness_zarr")
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
                power_values = shared_dict["sig0_20_ku"]
                if shared_dict["instr_mode"] == "SIN":
                    coherence_values = shared_dict["coh_at_rtrk_point"]
                    uncertainty = get_binned_values(
                            slope_values,
                            roughness_values,
                            power_values,
                            coherence_values,
                            self.ut_table_ant_sin,
                            self.ant_slope_bins_sin,
                            self.ant_roughness_bins_sin,
                            self.ant_power_bins_sin,
                            self.ant_coherence_bins_sin
                        )
                else:
                    uncertainty = get_binned_values(
                            slope_values,
                            roughness_values,
                            power_values,
                            None,
                            self.ut_table_ant_lrm,
                            self.ant_slope_bins_lrm,
                            self.ant_roughness_bins_lrm,
                            self.ant_power_bins_lrm,
                            None
                        )

            else:
                uncertainty = None
        else:
            slope_values = self.slope_grn.interp_slopes(
                shared_dict["latitudes"],
                shared_dict["longitudes"],
                xy_is_latlon=True,
            )
            roughness_values = self.roughness_grn.interp_roughness(
                    shared_dict["latitudes"],
                    shared_dict["longitudes"],
                    xy_is_latlon=True,
                )
            power_values = shared_dict["sig0_20_ku"]
            if shared_dict["instr_mode"] == "SIN":
                coherence_values = shared_dict["coh_at_rtrk_point"]
                uncertainty = get_binned_values(
                        slope_values,
                        roughness_values,
                        power_values,
                        coherence_values,
                        self.ut_table_grn_sin,
                        self.grn_slope_bins_sin,
                        self.grn_roughness_bins_sin,
                        self.grn_power_bins_sin,
                        self.grn_coherence_bins_sin
                    )
            else:
                uncertainty = get_binned_values(
                        slope_values,
                        roughness_values,
                        power_values,
                        None,
                        self.ut_table_grn_lrm,
                        self.grn_slope_bins_lrm,
                        self.grn_roughness_bins_lrm,
                        self.grn_power_bins_lrm,
                        None
                    )

        shared_dict["uncertainty"] = uncertainty

        # Return success (True,'')
        return (True, "")


# No finalize() required for this algorithm
