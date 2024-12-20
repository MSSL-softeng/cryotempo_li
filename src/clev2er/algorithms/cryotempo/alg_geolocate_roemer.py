"""clev2er.algorithms.cryotempo.alg_geolocate_roemer"""

from typing import Tuple

import numpy as np
from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=E0611

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.cs2.geolocate.geolocate_roemer import geolocate_roemer
from clev2er.utils.dems.dems import Dem
from clev2er.utils.dhdt_data.dhdt import Dhdt

# -------------------------------------------------

# pylint config
# Similar lines in 2 files, pylint: disable=R0801
# Too many return statements, pylint: disable=R0911
# Too many function args, pylint: disable=E1121


class Algorithm(BaseAlgorithm):
    """**Algorithm to perform LRM geolocation using a Roemer method**.

    CLEV2ER Algorithm: inherits from BaseAlgorithm

    Relocation using 100m or 200m DEMS, using

    antarctic_dem: rema_ant_200m
    greenland_dem: arcticdem_100m_greenland

    BaseAlgorithm __init__(config,thislog)
        Args:
            config: Dict[str, Any]: chain configuration dictionary
            thislog: logging.Logger | None: initial logger instance to use or
                                            None (use root logger)

    **Contribution to shared dictionary**

        - shared_dict["lat_poca_20_ku"] : np.ndarray (POCA latitudes)
        - shared_dict["lon_poca_20_ku"] : np.ndarray (POCA longitudes)
        - shared_dict["height_20_ku"]   : np.ndarray (elevations)
        - shared_dict["latitudes"]   : np.ndarray (final latitudes == POCA or nadir if failed)
        - shared_dict["longitudes"]   : np.ndarray (final lons == POCA or nadir if failed)

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
        # -----------------------------------------------------------------
        #  \/ Place Algorithm initialization steps here \/
        # -----------------------------------------------------------------

        # Check for special case where we create a shared memory
        # version of the DEM's arrays. Note this _init_shared_mem config setting is set by
        # run_chain.py and should not be included in the config files
        init_shared_mem = "_init_shared_mem" in self.config

        # Load DEMs required by this algorithm

        # Load Antarctic DEMs (unless grn_only is set in config)
        if "grn_only" in self.config and self.config["grn_only"]:
            self.dem_ant = None
            self.dem_ant_fine = None
        else:
            self.dem_ant = Dem(
                self.config["lrm_roemer_geolocation"]["antarctic_dem"],
                config=self.config,
                store_in_shared_memory=init_shared_mem,
                thislog=self.log,
            )
            # check if they are the same Dem (in which case we don't need to reload)
            if (
                self.config["lrm_roemer_geolocation"]["antarctic_dem"]
                == self.config["lrm_roemer_geolocation"]["antarctic_dem_fine"]
            ):
                self.dem_ant_fine = self.dem_ant
            else:
                self.dem_ant_fine = Dem(
                    self.config["lrm_roemer_geolocation"]["antarctic_dem_fine"],
                    config=self.config,
                    store_in_shared_memory=init_shared_mem,
                    thislog=self.log,
                )

        # Load Greenland DEMs (unless ais_only is set in config)
        if "ais_only" in self.config and self.config["ais_only"]:
            self.dem_grn = None
            self.dem_grn_fine = None
        else:
            self.dem_grn = Dem(
                self.config["lrm_roemer_geolocation"]["greenland_dem"],
                config=self.config,
                store_in_shared_memory=init_shared_mem,
                thislog=self.log,
            )
            # check if they are the same Dem (in which case we don't need to reload)
            if (
                self.config["lrm_roemer_geolocation"]["greenland_dem"]
                == self.config["lrm_roemer_geolocation"]["greenland_dem_fine"]
            ):
                self.dem_grn_fine = self.dem_grn
            else:
                self.dem_grn_fine = Dem(
                    self.config["lrm_roemer_geolocation"]["greenland_dem_fine"],
                    config=self.config,
                    store_in_shared_memory=init_shared_mem,
                    thislog=self.log,
                )
        # Load dh/dt data set if required
        if self.config["lrm_roemer_geolocation"]["include_dhdt_correction"]:
            self.log.info("Roemer dhdt correction enabled")
            self.dhdt_grn: Dhdt | None = Dhdt(
                self.config["lrm_roemer_geolocation"]["dhdt_grn_name"],
                config=self.config,
            )
            self.dhdt_ant = None  # Not implemented yet!
        else:
            self.dhdt_grn = None
            self.dhdt_ant = None

        # Important Note :
        #     each Dem classes instance must run Dem.clean_up() in Algorithm.finalize()

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

        if shared_dict["instr_mode"] != "LRM":
            self.log.info("algorithm skipped as not LRM file")
            return (True, "algorithm skipped as not LRM file")

        if shared_dict["hemisphere"] == "south":
            thisdem = self.dem_ant
            thisdem_fine = self.dem_ant_fine
            thisdhdt = self.dhdt_ant
        else:
            thisdem = self.dem_grn
            thisdem_fine = self.dem_grn_fine
            thisdhdt = self.dhdt_grn

        # Run the slope correction to calculate POCA lat,lon and height
        # If geolocation fails, lat,lon are set to nadir and height to np.nan
        (
            height_20_ku,
            lat_poca_20_ku,
            lon_poca_20_ku,
            slope_ok,
            relocation_distance,
        ) = geolocate_roemer(
            l1b,
            thisdem,
            thisdem_fine,
            thisdhdt,
            self.config,
            shared_dict["cryotempo_surface_type"],
            shared_dict["geo_corrected_tracker_range"],
            shared_dict["retracker_correction"],
            shared_dict["waveforms_to_include"],
        )

        self.log.info("LRM roemer geolocation completed")
        self.log.info(
            "Roemer slope correction succeded in %.2f %% of measurements",
            np.mean(slope_ok) * 100,
        )

        shared_dict["lat_poca_20_ku"] = lat_poca_20_ku
        np.seterr(under="ignore")  # otherwise next line can fail
        shared_dict["lon_poca_20_ku"] = lon_poca_20_ku % 360.0
        shared_dict["latitudes"] = lat_poca_20_ku
        shared_dict["longitudes"] = lon_poca_20_ku
        shared_dict["height_20_ku"] = height_20_ku
        shared_dict["roemer_slope_ok"] = slope_ok
        shared_dict["relocation_distance"] = relocation_distance

        # Return success (True,'')
        return (True, "")

    def finalize(self, stage: int = 0):
        """Perform final clean up actions for algorithm

        Args:
            stage (int, optional): Can be set to track at what stage the
            finalize() function was called
        """

        self.log.debug("Finalize algorithm %s called at stage %d", self.alg_name, stage)

        # --------------------------------------------------------
        # \/ Add algorithm finalization here \/
        # --------------------------------------------------------

        # Must run Dem.clean_up() for each Dem instance so that any shared memory is
        # unlinked, closed.
        if self.dem_ant is not None:
            self.dem_ant.clean_up()
        if self.dem_grn is not None:
            self.dem_grn.clean_up()
        if self.dem_grn_fine is not None:
            self.dem_grn_fine.clean_up()
        if self.dem_ant_fine is not None:
            self.dem_ant_fine.clean_up()

        # --------------------------------------------------------
