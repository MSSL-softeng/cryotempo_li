"""clev2er.algorithms.cryotempo.alg_uncertainty"""

from pathlib import Path
from typing import Tuple

import numpy as np
from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=E0611

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.raster_maps.raster_map_definitions import raster_map_from_name
from clev2er.utils.uncertainty.luts import MultiDimUncLut

# -------------------------------------------------

# pylint config
# Similar lines in 2 files, pylint: disable=R0801
# Too many return statements, pylint: disable=R0911
# pylint: disable=too-many-instance-attributes

REGIONS = ("antarctica", "greenland")
MODES = ("sin", "lrm")


class Algorithm(BaseAlgorithm):
    """**Algorithm to retrieve elevation uncertainty from empirical 3D/4D uncertainty LUTs**

    Replaces the previous 2D (Antarctica: slope, roughness) and 1D (Greenland: slope)
    uncertainty tables with N-D LUTs of median absolute CryoTEMPO-minus-ICESat-2
    elevation difference, binned by surface slope, surface roughness, power
    (backscatter) and, in SARIn mode, waveform coherence at the retracking point:

    - SIN measurements -> 4D LUT (slope, roughness, power, coherence)
    - LRM measurements -> 3D LUT (slope, roughness, power) : LRM products carry
      no coherence

    Slope and roughness are interpolated at each measurement location from the same
    rastermaps used to train the LUTs (REMA v2 / ArcticDEM v4.1, 100 m, SVD 9x9 -
    as used by the CLEV2ER land-ice chain), so LUT training and runtime lookup
    covariates are consistent by construction. Power is the calibrated backscatter
    (sig0) at each measurement; coherence is the raw (un-smoothed) coherence at the
    retracking point.

    **Required from shared_dict**
        - shared_dict["latitudes"], shared_dict["longitudes"] : measurement locations
          (POCA, or nadir if no POCA available)
        - shared_dict["hemisphere"] : "south" or "north" (selects the region LUT set)
        - shared_dict["instr_mode"] : "SIN" or "LRM" (selects the mode LUT)
        - shared_dict["sig0_20_ku"] : backscatter, the `power` covariate
        - shared_dict["coherence_at_rtrk_point"] : the `coherence` covariate (SIN only)

    **Contribution to shared_dict**
        - shared_dict["uncertainty"] : (np.ndarray) uncertainty at each track location
          (NaN where any covariate is NaN, e.g. off the slope/roughness rastermaps)

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

        Loads, per region (antarctica, greenland):
            - the SIN (4D) and LRM (3D) uncertainty LUT NetCDF files, from
              config["uncertainty_tables"]: base_dir + per region/mode
              filename + covariates
            - the slope and roughness rastermaps, from config["surface_slopes"] and
              config["surface_roughness"]: per region raster_map_name + directory

        If config["grn_only"] is set, Antarctic resources are not loaded (as in the
        previous version of this algorithm).

        Returns:
            (bool,str) : success or failure, error string

        Raises:
            KeyError : keys not in config
            OSError : LUT or rastermap file not found

        Note: raise an Exception rather than just returning False
        """
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        regions = list(REGIONS)
        if "grn_only" in self.config and self.config["grn_only"]:
            regions.remove("antarctica")

        # -------------------------------------------------------------------------
        # Load uncertainty LUTs : config["uncertainty_tables"]
        #   base_dir: directory containing the LUT NetCDF files
        #   <region>: <mode>: {filename, covariates} per region/mode
        # -------------------------------------------------------------------------

        if "uncertainty_tables" not in self.config:
            raise KeyError("uncertainty_tables not in config")
        unc_config = self.config["uncertainty_tables"]
        if "base_dir" not in unc_config:
            raise KeyError("uncertainty_tables.base_dir not in config")
        base_dir = Path(unc_config["base_dir"])

        self.uncertainty_luts: dict = {}
        for region in regions:
            if region not in unc_config:
                raise KeyError(f"uncertainty_tables.{region} not in config")
            self.uncertainty_luts[region] = {}
            for mode in MODES:
                try:
                    mode_config = unc_config[region][mode]
                    lut_path = base_dir / mode_config["filename"]
                    covariates_str = mode_config["covariates"]
                except KeyError as exc:
                    raise KeyError(
                        f"uncertainty_tables.{region}.{mode} missing filename or "
                        f"covariates: {exc}"
                    ) from exc
                covariates = [c.strip() for c in str(covariates_str).split(",") if c.strip()]
                self.log.info(
                    "loading %s %s uncertainty LUT from %s (covariates=%s)",
                    region,
                    mode,
                    lut_path,
                    covariates,
                )
                try:
                    self.uncertainty_luts[region][mode] = MultiDimUncLut(
                        str(lut_path), covariates=covariates, thislog=self.log
                    )
                except (KeyError, OSError, ValueError) as exc:
                    self.log.error("failed to load %s %s uncertainty LUT: %s", region, mode, exc)
                    raise OSError(f"{region} {mode} uncertainty LUT load failed") from exc

        # -------------------------------------------------------------------------
        # Load slope and roughness rastermaps : config["surface_slopes"],
        # config["surface_roughness"] : per region {raster_map_name, directory}
        # -------------------------------------------------------------------------

        self.slopes: dict = {}
        self.roughness: dict = {}
        for resource_key, target, map_type in (
            ("surface_slopes", self.slopes, "slope"),
            ("surface_roughness", self.roughness, "roughness"),
        ):
            if resource_key not in self.config:
                raise KeyError(f"{resource_key} not in config")
            for region in regions:
                try:
                    region_config = self.config[resource_key][region]
                    map_name = region_config["raster_map_name"]
                    directory = Path(region_config["directory"])
                except KeyError as exc:
                    raise KeyError(
                        f"{resource_key}.{region} missing raster_map_name or directory: {exc}"
                    ) from exc
                self.log.info(
                    "loading %s %s rastermap '%s' from %s", region, map_type, map_name, directory
                )
                try:
                    target[region] = raster_map_from_name(
                        map_name,
                        directory=directory,
                        map_type=map_type,  # type: ignore[arg-type]
                    )
                except (KeyError, OSError, ValueError) as exc:
                    self.log.error(
                        "failed to load %s rastermap for region '%s': %s", map_type, region, exc
                    )
                    raise OSError(f"{map_type} rastermap load failed for '{region}'") from exc

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

        # Calculate uncertainty from measurement locations (POCA, or nadir if POCA
        # failed): shared_dict["latitudes"], shared_dict["longitudes"]

        region = "antarctica" if shared_dict["hemisphere"] == "south" else "greenland"
        if region not in self.uncertainty_luts:
            # region resources not loaded (e.g. grn_only run): same behaviour as the
            # previous version of this algorithm
            shared_dict["uncertainty"] = None
            return (True, "")

        instr_mode = shared_dict["instr_mode"]
        if instr_mode == "SIN":
            mode = "sin"
        elif instr_mode == "LRM":
            mode = "lrm"
        else:
            return (False, f"instrument mode {instr_mode} must be LRM or SIN")

        lats = np.asarray(shared_dict["latitudes"], dtype=np.float64)
        lons = np.asarray(shared_dict["longitudes"], dtype=np.float64)
        power = np.asarray(shared_dict["sig0_20_ku"], dtype=np.float64)
        if not lats.size == lons.size == power.size:
            return (
                False,
                f"length mismatch: latitudes {lats.size}, longitudes {lons.size}, "
                f"sig0_20_ku {power.size}",
            )

        covariate_values = {
            "slope": self.slopes[region].interpolate(lats, lons, xy_is_latlon=True),
            "roughness": self.roughness[region].interpolate(lats, lons, xy_is_latlon=True),
            "power": power,
        }
        if mode == "sin":
            coherence = np.asarray(shared_dict["coherence_at_rtrk_point"], dtype=np.float64)
            if coherence.size != lats.size:
                return (
                    False,
                    f"length mismatch: coherence_at_rtrk_point {coherence.size}, "
                    f"latitudes {lats.size}",
                )
            covariate_values["coherence"] = coherence

        lut = self.uncertainty_luts[region][mode]
        try:
            lut_inputs = {name: covariate_values[name] for name in lut.active_covariates}
        except KeyError as exc:
            return (
                False,
                f"uncertainty LUT for {region}/{mode} requires covariate {exc} which is "
                "not available in this mode (check uncertainty_tables config)",
            )

        uncertainty = lut.get_uncertainty(**lut_inputs)

        n_finite = int(np.count_nonzero(np.isfinite(uncertainty)))
        self.log.info(
            "uncertainty (%s, %s): %d of %d finite, median=%.3f m",
            region,
            mode,
            n_finite,
            uncertainty.size,
            float(np.nanmedian(uncertainty)) if n_finite else float("nan"),
        )

        shared_dict["uncertainty"] = uncertainty

        # Return success (True,'')
        return (True, "")


# No finalize() required for this algorithm
