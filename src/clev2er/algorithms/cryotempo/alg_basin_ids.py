""" clev2er.algorithms.templates.alg_basin_ids

# New in Baseline-D

Mouginot (Greenland) replaces Zwally (Greenland)

* basin_id : Zwally (AIS) and Mouginot (GIS)
* basin_id_2: Rignot (both AIS and GIS)

"""

# These imports required by Algorithm template
from typing import Tuple

import numpy as np
from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=E0611

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.masks.masks import Mask

# -------------------------------------------------

# pylint config
# Similar lines in 2 files, pylint: disable=R0801
# Too many return statements, pylint: disable=R0911


class Algorithm(BaseAlgorithm):
    """**Algorithm to do find ice sheet basin id for each location along track**

    BaseAlgorithm: __init__(config,thislog)
        Args:
            config: Dict[str, Any]: chain configuration dictionary
            thislog: logging.Logger | None: initial logger instance to use or
                                            None (use root logger)

    ** Contribution to Shared Dictionary **

        - shared_dict["basin_mask_values_rignot"] : (np.ndarray), basin mask values from Rignot
        - shared_dict["basin_mask_values_zwally"] : (np.ndarray), basin mask values from Zwally

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

        # Check for special case where we create a shared memory
        # version of the DEM's arrays. Note this _init_shared_mem config setting is set by
        # run_chain.py and should not be included in the config files
        init_shared_mem = "_init_shared_mem" in self.config

        # ---------------------------------------------------------------------------------
        # Load Basin Masks required to create 'basin_id' netCDF variable
        #  antarctic_grounded_and_floating_2km_grid_mask
        #  greenland_icesheet_2km_grid_mask
        #  antarctic_icesheet_2km_grid_mask_rignot2016
        # greenland_icesheet_2km_grid_mask_rignot2016
        # ---------------------------------------------------------------------------------

        required_masks = [
            "antarctic_grounded_and_floating_2km_grid_mask",  # Zwally 2012 (AIS)
            "greenland_icesheet_2km_grid_mask_mouginot2019",  # Mouginot 2019 (GIS)
            "antarctic_icesheet_2km_grid_mask_rignot2016",  # Rignot 2016 (AIS)
            "greenland_icesheet_2km_grid_mask_rignot2016",  # Rignot 2016 (GIS)
        ]

        # Check that the config file supports these masks in basin_masks:<mask_name>
        mask_paths = {}
        for mask in required_masks:
            try:
                mask_file = self.config["basin_masks"][mask]
            except KeyError as exc:
                self.log.error(
                    "surface_type_masks:%s not in config file, %s",
                    mask,
                    exc,
                )
                raise KeyError(f"surface_type_masks:{mask} not in config file") from exc
            mask_paths[mask] = mask_file

        if "grn_only" in self.config and self.config["grn_only"]:
            self.zwally_basin_mask_ant = None
        else:
            # Load : antarctic_grounded_and_floating_2km_grid_mask
            #      : source: Zwally 2012, ['unknown','1',..'27']
            self.zwally_basin_mask_ant = Mask(
                "antarctic_grounded_and_floating_2km_grid_mask",
                mask_path=mask_paths["antarctic_grounded_and_floating_2km_grid_mask"],
                store_in_shared_memory=init_shared_mem,
                thislog=self.log,
            )  # source: Zwally 2012, ['unknown','1',..'27']

        # Load: greenland_icesheet_2km_grid_mask
        # source: Zwally 2012, ['None', '1.1', '1.2', '1.3', '1.4', '2.1', '2.2', '3.1', '3.2',
        # '3.3', '4.1', '4.2', '4.3', '5.0', '6.1', '6.2', '7.1', '7.2', '8.1', '8.2']

        self.mouginot_basin_mask_grn = Mask(
            "greenland_icesheet_2km_grid_mask_mouginot2019",
            mask_path=mask_paths["greenland_icesheet_2km_grid_mask_mouginot2019"],
            store_in_shared_memory=init_shared_mem,
            thislog=self.log,
        )

        if "grn_only" in self.config and self.config["grn_only"]:
            self.rignot_basin_mask_ant = None
        else:
            # Load: antarctic_icesheet_2km_grid_mask_rignot2016
            # source: Rignot 2016, values: 0-18 ['Islands','West H-Hp','West F-G',
            # 'East E-Ep','East D-Dp',
            # 'East Cp-D','East B-C','East A-Ap','East Jpp-K','West G-H','East Dp-E','East Ap-B',
            # 'East C-Cp',
            # 'East K-A','West J-Jpp','Peninsula Ipp-J','Peninsula I-Ipp','Peninsula Hp-I',
            # 'West Ep-F']
            self.rignot_basin_mask_ant = Mask(
                "antarctic_icesheet_2km_grid_mask_rignot2016",
                mask_path=mask_paths["antarctic_icesheet_2km_grid_mask_rignot2016"],
                store_in_shared_memory=init_shared_mem,
                thislog=self.log,
            )

        # Load: greenland_icesheet_2km_grid_mask_rignot2016
        # source: Rignot 2016, values: 0,1-56: 0 (unclassified), 1-50 (ice caps), 51 (NW), 52(CW),
        # 53(SW), 54(SE), 55(NE), 56(NO)
        self.rignot_basin_mask_grn = Mask(
            "greenland_icesheet_2km_grid_mask_rignot2016",
            mask_path=mask_paths["greenland_icesheet_2km_grid_mask_rignot2016"],
            store_in_shared_memory=init_shared_mem,
            thislog=self.log,
        )

        # Important Note :
        #     each Mask classes instance must run Mask.clean_up() in Algorithm.finalize()

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        """CLEV2ER Algorithm

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

        if shared_dict["hemisphere"] == "south":
            if self.zwally_basin_mask_ant is None:
                raise ValueError("zwally_basin_mask_ant should not be None")
            # 0..27 : ['unknown','1',..'27']
            mask_values_zwally = self.zwally_basin_mask_ant.grid_mask_values(
                shared_dict["latitudes"], shared_dict["longitudes"], unknown_value=0
            )

            if self.rignot_basin_mask_ant is None:
                raise ValueError("rignot_basin_mask_ant should not be None")

            # 0..18: ['Islands','West H-Hp','West F-G','East E-Ep','East D-Dp','East Cp-D',
            # 'East B-C','East A-Ap','East Jpp-K','West G-H','East Dp-E','East Ap-B','East C-Cp',
            # 'East K-A','West J-Jpp','Peninsula Ipp-J','Peninsula I-Ipp','Peninsula Hp-I',
            # 'West Ep-F']
            # 999 (unknown)
            mask_values_rignot = self.rignot_basin_mask_ant.grid_mask_values(
                shared_dict["latitudes"], shared_dict["longitudes"], unknown_value=999
            )

            # reset the number range so we have 0 (unclassified), 1 (islands)..19(West EP-F)
            for i, m in enumerate(mask_values_rignot):
                if 0 <= m <= 18:
                    mask_values_rignot[i] = m + 1
                else:
                    mask_values_rignot[i] = 0

            shared_dict["basin_mask_values_rignot"] = mask_values_rignot.astype(np.uint)
            shared_dict["basin_mask_values_zwally"] = mask_values_zwally.astype(np.uint)
            shared_dict["basin_mask_values_mouginot"] = None

        else:
            # "Unclassified [0]",
            # "CE [1]", "CW [2]", "NO [3]", "NE [4]", "NW [5]", "SE [6]", "SW [7]"
            mask_values_mouginot = self.mouginot_basin_mask_grn.grid_mask_values(
                shared_dict["latitudes"], shared_dict["longitudes"], unknown_value=0
            )

            mask_values_rignot = self.rignot_basin_mask_grn.grid_mask_values(
                shared_dict["latitudes"], shared_dict["longitudes"], unknown_value=0
            )  # 0..56 : (unclassified), 1-50 (ice caps), 51 (NW), 52(CW), 53(SW), 54(SE),
            #            55(NE), 56(NO)
            # reset the number range so we have 0 (unclassified),
            # 1 (ice caps), 2(NW), 3(CW), 4(SW), 5(SE), 6(NE), 7(NO)
            for i, m in enumerate(mask_values_rignot):
                if 1 <= m <= 50:
                    mask_values_rignot[i] = 1
                elif 51 <= m <= 56:
                    mask_values_rignot[i] = (m - 50) + 1
                else:
                    mask_values_rignot[i] = 0

            shared_dict["basin_mask_values_rignot"] = mask_values_rignot.astype(np.uint)
            shared_dict["basin_mask_values_zwally"] = None
            shared_dict["basin_mask_values_mouginot"] = mask_values_mouginot.astype(np.uint)

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
        # Must run Mask.clean_up() for each Mask instance so that any shared memory is
        # unlinked, closed.

        if self.zwally_basin_mask_ant is not None:
            self.zwally_basin_mask_ant.clean_up()
        if self.mouginot_basin_mask_grn is not None:
            self.mouginot_basin_mask_grn.clean_up()

        if self.rignot_basin_mask_ant is not None:
            self.rignot_basin_mask_ant.clean_up()
        if self.rignot_basin_mask_grn is not None:
            self.rignot_basin_mask_grn.clean_up()

        # --------------------------------------------------------
