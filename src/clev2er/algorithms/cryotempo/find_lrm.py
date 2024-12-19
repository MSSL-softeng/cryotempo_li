
"""find_lrm module
"""


import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List

from netCDF4 import Dataset  # pylint: disable=E0611

from clev2er.algorithms.base.base_finder import BaseFinder

# pylint: disable=R0801
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals

log = logging.getLogger(__name__)


def test_nc_file_in_greenland(file: str):
    """test if L1b file's global attributes indicate it is measuring over
       Greenland

    Args:
        file (str): path of CS2 L1b file

    Returns:
        None|str : None if not over Grn, file if it is
    """
    try:
        with Dataset(file) as nc:
            first_record_lat = nc.first_record_lat / 1e6
            if first_record_lat < 0.0:
                return None
            first_record_lon = nc.first_record_lon / 1e6
            if first_record_lon > 10.0:
                return None
            if first_record_lon < -90.0:
                return None
            return file
    except IOError:
        return None


class FileFinder(BaseFinder):
    """class to find a list of LRM L1b files to process in one or more
       specified months, from

    <base_path>/LRM/<YYYY>/<MM>/CS_*SIR_*.nc

    Args:
        BaseFinder (BaseFinder): base finder class

    In order to find files you can optionally use the following
    which are optionally set by the run_chain.py command line parameters

    Set by command line options:
        self.months  # list of months to find
        self.years   # list of years to find
    Set by config file settings:
        config["l1b_base_dir"]

    """

    # The base class is initialized with:
    # def __init__(self, log: logging.Logger | None = None, config: dict = {}):

    def find_files(self, flat_search=False) -> list[str]:
        """Search for L1b file according to pattern

        Args:
            flat_search (bool) : if True only search in l1b_base_dir, else use pattern
        Returns:
            (str): list of files

        """
        file_list: List[str] = []

        if "sin_only" in self.config:
            return file_list

        if "l1b_baselines" not in self.config:
            raise KeyError("l1b_baselines missing from config")
        l1b_baselines = self.config[
            "l1b_baselines"
        ]  # comma separated list of l1b baseline chars, ie D,E

        if "l1b_base_dir" not in self.config:
            raise KeyError("l1b_base_dir missing from config")
        l1b_base_dir = self.config["l1b_base_dir"]

        if not os.path.isdir(l1b_base_dir):
            raise FileNotFoundError(f"{l1b_base_dir} directory not found")

        if len(self.years) < 1:
            raise ValueError("Empty year list in find_files(). Use .add_year() first")

        if len(self.months) == 0:
            self.months = list(range(1, 13))

        for year in self.years:
            self.log.info("Finding files for year: %d", year)
            for month in self.months:
                self.log.info("Finding files for month: %d", month)
                if flat_search:
                    search_str = (
                        f"{l1b_base_dir}"
                        f"/CS_*SIR_LRM_1B_{year:4d}{month:02d}*[{l1b_baselines}]???.nc"
                    )
                else:
                    search_str = (
                        f"{l1b_base_dir}/LRM/{year:4d}/{month:02d}"
                        f"/CS_*SIR_LRM_1B_{year:4d}{month:02d}*[{l1b_baselines}]???.nc"
                    )
                self.log.info("search string=%s", search_str)
                files = glob.glob(search_str)
                nfiles = len(files)
                if nfiles:
                    file_list.extend(files)
                self.log.info("Number of files found for %.02d/%d: %d", month, year, nfiles)
        self.log.info("Total number of LRM files found: %d", len(file_list))

        if "grn_only" in self.config and self.config["grn_only"]:
            self.log.info("Filtering LRM file list for --grn_only")

            # As filtering the file list can be slow (as there may be ~20000 files/yr)
            # use multi-processing to optionally speed it up by ~ 3x

            if (
                "chain" in self.config
                and "max_processes_for_multiprocessing" in self.config["chain"]
            ):
                num_processes = self.config["chain"]["max_processes_for_multiprocessing"]
            else:
                num_processes = 1

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                results = list(executor.map(test_nc_file_in_greenland, file_list))

            grn_file_list = [result for result in results if result is not None]

            file_list = grn_file_list
            self.log.info(
                "Total number of LRM files found after grn_only filter: %d",
                len(file_list),
            )

        return file_list
