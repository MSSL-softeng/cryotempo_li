""" clev2er.algorithms.cryotempo.alg_cats2023_tide_correction """

# These imports required by Algorithm template
import os
from typing import Tuple

import numpy as np
import pyTMD.compute
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=E0611

from clev2er.algorithms.base.base_alg import BaseAlgorithm

# -------------------------------------------------

# Similar lines in 2 files, pylint: disable=R0801
# Too many return statements, pylint: disable=R0911


class Algorithm(BaseAlgorithm):
    """Algorithm to compute the CATS2008-v2023 (CATS2023) ocean tide elevation at
    each L1b nadir location, for SIN mode files in the southern hemisphere.

    Baseline-F replaces the CATS2008a tide model used in baselines B-E. CATS2023
    provides a finer 2 km grid (from 4 km), coastlines, grounding lines and water
    column thicknesses from BedMachine v3, and an ice shelf flexure model.

    In baselines B-E the CATS2008a tide was pre-computed by a separate tool
    (cpom_software compute_tidal_elevations_from_pytmd.py, run in its own conda
    env) which wrote one tide file per L1b file. From baseline-F the tide is
    computed on the fly here using pyTMD, which removes the pre-processing stage,
    its storage, and the need for a separate environment.

    The tide is evaluated at the L1b nadir locations (lat_20_ku, lon_20_ku),
    matching where the pre-computed CATS2008a files were evaluated. This
    algorithm therefore runs before geolocation, in the same chain position as
    the CATS2008a algorithm it replaces.

    CLEV2ER Algorithm: inherits from BaseAlgorithm

    BaseAlgorithm __init__(config,thislog)
        Args:
            config: Dict[str, Any]: chain configuration dictionary
            thislog: logging.Logger | None: initial logger instance to use or
                                            None (use root logger)

    **Requires from shared dictionary**:

    - `shared_dict["hemisphere"]` : str
    - `shared_dict["instr_mode"]` : str
    - `shared_dict["num_20hz_records"]` : int
    - `shared_dict["floating_ice_locations"]` : np.ndarray
    - `shared_dict["ocean_locations"]` : np.ndarray

    **Outputs to shared dictionary**:

    - `shared_dict["cats_tide"]` : np.ndarray
    - `shared_dict["cats_tide_required"]` : bool, True if CATS tide has been calculated

    Note the shared_dict keys are unchanged from alg_cats2008a_tide_correction so
    that alg_geo_corrections requires no modification.
    """

    # Note: __init__() is in BaseAlgorithm. See required parameters above
    # init() below is called by __init__() at a time dependent on whether
    # sequential or multi-processing mode is in operation

    def init(self) -> Tuple[bool, str]:
        """Algorithm initialization

        Reads the pyTMD tide model settings from the chain config.

        Note that the tide model itself is deliberately *not* loaded here. In
        multi-processing mode run_chain.py spawns one process per L1b file, so
        init() runs per file and there is nothing to be gained by caching the
        model on this object. pyTMD reads what it needs inside process().

        Returns:
            (bool,str) : success or failure, error string

        Raises:
            KeyError : keys not in config
            FileNotFoundError :
            OSError :

        """

        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # Add initialization steps here
        # ------------------------------------------------------------------------------
        # Get the pyTMD tide model settings from the config file: tides.*
        # ------------------------------------------------------------------------------

        if "tides" not in self.config:
            self.log.error("tides section missing from config file")
            return (False, "tides section missing from config file")

        try:
            self.pytmd_tide_models_dir = self.config["tides"]["pytmd_tide_models_dir"]
            self.cats2023_model_name = self.config["tides"]["cats2023_model_name"]
        except KeyError as exc:
            self.log.error("tides.%s missing from config file", exc)
            return (False, f"tides.{exc} missing from config file")

        # Optional tuning parameters, defaulted to the values used by the
        # baselines B-E CATS2008a pre-processor, other than apply_flexure which
        # is new in CATS2023.
        self.cats2023_interp_method = self.config["tides"].get("cats2023_interp_method", "spline")
        self.cats2023_extrapolate = self.config["tides"].get("cats2023_extrapolate", False)
        self.cats2023_cutoff_km = self.config["tides"].get("cats2023_cutoff_km", 10.0)
        self.cats2023_apply_flexure = self.config["tides"].get("cats2023_apply_flexure", True)

        # Check that the tide model directory exists, and contains the model
        if not os.path.isdir(self.pytmd_tide_models_dir):
            self.log.error("%s does not exist", self.pytmd_tide_models_dir)
            return (
                False,
                f"tides.pytmd_tide_models_dir {self.pytmd_tide_models_dir} not found",
            )

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
        # Perform the algorithm processing, store results that need to passed
        # down the chain in the 'shared_dict' dict
        # -------------------------------------------------------------------

        shared_dict["cats_tide_required"] = False

        if shared_dict["hemisphere"] == "north":
            self.log.info("no CATS tide correction required for northern hemisphere")
            return (
                True,  # we don't want to skip file
                "no CATS tide correction required for northern hemisphere",
            )

        if shared_dict["instr_mode"] != "SIN":
            self.log.debug(
                "no CATS tide correction required for %s mode",
                shared_dict["instr_mode"],
            )
            return (
                True,  # we don't want to skip file
                f'No CATS tide correction required for {shared_dict["instr_mode"]} mode',
            )

        if len(shared_dict["floating_ice_locations"]) == 0:
            if len(shared_dict["ocean_locations"]) == 0:
                self.log.info(
                    "no CATS tide correction required as no floating or ocean measurements",
                )
                return (
                    True,  # we don't want to skip file
                    "no CATS tide correction required as not over ocean or floating ice",
                )

        self.log.info("Computing %s tide correction...", self.cats2023_model_name)

        # -------------------------------------------------------------------
        # Read the nadir locations and times from the L1b
        # -------------------------------------------------------------------

        try:
            lats = l1b.variables["lat_20_ku"][:].data.astype(np.float64)
            # pyTMD is given longitude in the 0..360 range, as per the baselines
            # B-E pre-processor
            lons = np.mod(l1b.variables["lon_20_ku"][:].data.astype(np.float64), 360.0)
            # CS2 time_20_ku is TAI seconds since 2000-01-01 00:00:00
            delta_time = l1b.variables["time_20_ku"][:].data.astype(np.float64)
        except (KeyError, AttributeError) as exc:
            self.log.error("Error reading l1b nadir location/time variables : %s", exc)
            return (False, "Error reading l1b nadir location/time variables")

        # -------------------------------------------------------------------
        # Compute the tide elevations
        # -------------------------------------------------------------------

        # The CS2 retracker modules set np.seterr("raise") at import, so harmless
        # underflow in pyTMD's minor constituent inference would otherwise raise a
        # FloatingPointError. Scope the relaxation to this call using np.errstate
        # rather than np.seterr, so we do not alter the setting for other algorithms.
        try:
            with np.errstate(under="ignore"):
                cats_tide = pyTMD.compute.tide_elevations(
                    x=lons,
                    y=lats,
                    delta_time=delta_time,
                    DIRECTORY=self.pytmd_tide_models_dir,
                    MODEL=self.cats2023_model_name,
                    EPSG=4326,  # lon,lat
                    EPOCH=(2000, 1, 1, 0, 0, 0),
                    TIME="TAI",
                    TYPE="drift",  # each measurement has its own timestamp
                    METHOD=self.cats2023_interp_method,
                    EXTRAPOLATE=self.cats2023_extrapolate,
                    CUTOFF=self.cats2023_cutoff_km,
                    APPLY_FLEXURE=self.cats2023_apply_flexure,
                )
        except (OSError, ValueError, KeyError) as exc:
            self.log.error("Error computing %s tide elevations : %s", self.cats2023_model_name, exc)
            return (False, f"Error computing {self.cats2023_model_name} tide elevations")

        # pyTMD returns a masked array. Take the data, with masked values as NaN
        cats_tide = np.ma.filled(cats_tide, np.nan).astype(np.float64).flatten()

        # Check that we have the same number of 20hz records as the L1b file
        if cats_tide.size != shared_dict["num_20hz_records"]:
            self.log.error(
                "CATS2023 tide array length %d should equal num L1b 20Hz records %d",
                cats_tide.size,
                shared_dict["num_20hz_records"],
            )
            return (False, "CATS2023 tide array length mismatch to L1b record size")

        num_valid = int(np.isfinite(cats_tide).sum())
        self.log.info(
            "%s tide computed at %d of %d nadir locations",
            self.cats2023_model_name,
            num_valid,
            cats_tide.size,
        )

        # Locations outside the tide model grid are NaN. Replace with zero, as
        # per the CATS2008a implementation in baselines B-E, so that the summed
        # geo-corrections in alg_geo_corrections are not invalidated by
        # measurements that lie outside the model domain.
        np.nan_to_num(cats_tide, copy=False)

        shared_dict["cats_tide"] = cats_tide
        shared_dict["cats_tide_required"] = True

        # Return success (True,'')
        return (True, "")


# Note no Algorithm.finalize() required for this particular algorithm
