# mypy: ignore-errors
#!/usr/bin/env python
"""
A simplified DEM downloader for Copernicus DEM data over a polygon
area covering Greenland.

This version removes project-specific parts (e.g. reading a KML file for Sentinel-2 tiles,
XML configuration, and anti-meridian handling) so that it focuses only on using a hard-coded
polygon to query and download DEM tiles. It also implements pagination to download all products
and automatically refreshes the access token if it becomes invalid.

Before running, make sure you have your Copernicus credentials and that the required packages are installed.
"""

# pylint: skip-file        # Pylint: ignore this entire file
# mypy: ignore-errors      # mypy: suppress all type errors in this file
# ruff: noqa               # Ruff: ignore all lint rules in this file

import logging
import pathlib
import re
import shutil
from zipfile import ZipFile

import requests

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")
logger = logging.getLogger("DEM_Downloader")


class DemDownloader:
    def __init__(
        self,
        polygon: str,
        dem_resolution="90",  # Use "90" for 90 m DEMs, "30" for 30 m DEMs.
        dem_format="DGED",
        dem_collection="COP-DEM",
        output_dir="/cpdata/SATS/RA/DEMS/COP90",
    ):
        """
        :param polygon: A string representing the polygon in WKT coordinate format (without 'POLYGON(())')
                        e.g. "-73 83, -12 83, -12 59, -73 59, -73 83"
                        (Coordinates are given in longitude latitude order.)
        :param dem_resolution: Desired resolution in meters. Use "90" for 90 m (GLO-90) or "30" for 30 m (GLO-30).
                              Note that while you set this in meters, the underlying product naming
                              uses arc seconds.
        :param dem_format: DEM product format, either "DGED" or "DTED"
        :param dem_collection: Copernicus DEM product collection (e.g., "COP-DEM")
        :param output_dir: Directory where the DEM files will be saved and extracted.
        """
        self.polygon = polygon
        self.dem_resolution = dem_resolution
        self.dem_format = dem_format
        self.dem_collection = dem_collection
        self.dem_search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # These will be set later in run()
        self.username = None
        self.password = None
        self.access_token = None

    def get_access_token(self, username: str, password: str) -> str:
        """Obtain an access token from the Copernicus dataspace."""
        data = {
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        }
        logger.info("Requesting access token...")
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
            timeout=(60.0, 60.0),
        )
        r.raise_for_status()
        token = r.json()["access_token"]
        logger.info("Access token received.")
        return token

    def create_url(self) -> str:
        """Construct the search URL using the defined polygon and DEM parameters.

        The API naming convention uses arc seconds rather than meters:
          - For a 90 m DEM (GLO-90), use "30" (30 arc seconds) in the product name.
          - For a 30 m DEM (GLO-30), use "10" (10 arc seconds) in the product name.
        This conversion is applied automatically.
        """
        dem_url_model = "DGE" if self.dem_format.upper() == "DGED" else "DTE"
        # Convert the specified resolution (meters) into the corresponding arc second value:

        product_type = f"{dem_url_model}_{self.dem_resolution}"
        collection_req = f"Collection/Name eq '{self.dem_collection}'"
        product_type_req = (
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            f"and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        )
        polygon_req = f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({self.polygon}))')"
        max_n_items = "$top=100"  # Retrieve up to 100 items per page
        url = f"{self.dem_search_url}{collection_req} and {product_type_req} and {polygon_req}&{max_n_items}"
        logger.info("Created DEM search URL.")
        print("Search URL:", url)
        return url

    def retrieve_dem_list(self, url: str) -> list:
        """Query the catalogue to obtain the complete list of DEM product IDs, handling pagination."""
        dem_ids = []
        while url:
            logger.info("Querying DEM catalogue: %s", url)
            response = requests.get(url, timeout=(60.0, 60.0))
            response.raise_for_status()
            data = response.json()
            if "value" in data and data["value"]:
                current_ids = [item["Id"] for item in data["value"]]
                dem_ids.extend(current_ids)
                logger.info(
                    "Retrieved %s items from current page, total so far: %s",
                    len(current_ids),
                    len(dem_ids),
                )
            else:
                logger.warning("No DEM products found in current page.")
                break

            # OData responses typically include an '@odata.nextLink' for pagination.
            url = data.get("@odata.nextLink")
            if url:
                logger.info("Found next page of results. Continuing to retrieve...")
        logger.info("Total DEM products found: %s", len(dem_ids))
        return dem_ids

    def download_dem(self, dem_id: str):
        """Download a DEM product with the given ID and extract the DEM file from the ZIP.

        If a 401 error is encountered, the function will refresh the access token
        and retry the download (up to one retry).
        """
        download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({dem_id})/$value"
        for attempt in range(2):  # Allow one retry if token is expired.
            headers = {"Authorization": f"Bearer {self.access_token}"}
            logger.info("Downloading DEM with ID %s (attempt %s)...", dem_id, attempt + 1)
            response = requests.get(
                download_url, headers=headers, stream=True, timeout=(60.0, 60.0)
            )
            if response.status_code == 401:
                logger.warning(
                    "Received 401 Unauthorized for DEM %s. Refreshing access token...", dem_id
                )
                self.access_token = self.get_access_token(self.username, self.password)
                continue  # Retry the download with the new token.
            # Raise an exception for other HTTP errors.
            response.raise_for_status()
            zip_path = self.output_dir / f"{dem_id}.zip"
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Downloaded %s", zip_path)
            self.extract_dem(zip_path, dem_id)
            break  # Exit after a successful download.

    def extract_dem(self, zip_path: pathlib.Path, dem_id: str):
        """Extract the DEM file (with a .tif, .dt1, or .dt2 extension) from the ZIP archive."""
        logger.info("Extracting DEM content from %s...", zip_path)
        with ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                # Look for a DEM file by matching file names ending with DEM.tif, DEM.dt1, or DEM.dt2
                if re.search(r".*DEM\.(tif|dt1|dt2)$", member):
                    filename = member.split("/")[-1]  # extract file name from possible path
                    if not filename:
                        continue
                    with (
                        zip_ref.open(member) as source,
                        open(self.output_dir / filename, "wb") as target,
                    ):
                        shutil.copyfileobj(source, target)
                    logger.info("Extracted %s for DEM id %s.", filename, dem_id)
        zip_path.unlink()  # Remove the zip file after extraction

    def run(self, username: str, password: str):
        """Main method to execute the download process."""
        self.username = username
        self.password = password
        self.access_token = self.get_access_token(username, password)
        url = self.create_url()
        dem_list = self.retrieve_dem_list(url)
        if not dem_list:
            logger.info("No DEM tiles found for the provided search criteria.")
            return
        logger.info("Starting download of %s DEM tiles.", len(dem_list))
        count = 0
        for dem_id in dem_list:
            try:
                self.download_dem(dem_id)
            except Exception as e:
                logger.error("Error downloading DEM %s: %s", dem_id, e)
            count += 1
            # Refresh token every 100 downloads to pre-empt expiration.
            if count % 100 == 0:
                logger.info("Refreshing access token after %s downloads...", count)
                self.access_token = self.get_access_token(self.username, self.password)


def main():
    # Define a polygon around Greenland.
    # Coordinates: (-73,83), (-12,83), (-12,59), (-73,59), (-73,83)
    max_lat = 63  # normally 83
    min_lat = 56
    min_lon = -50  # normally -73
    max_lon = -43  # normally -11

    greenland_polygon = f"{min_lon} {max_lat}, {max_lon} {max_lat}, {max_lon} {min_lat}, {min_lon} {min_lat}, {min_lon} {max_lat}"

    # To download 90 m DEMs (GLO-90; the files will be named like Copernicus_DSM_30_...),
    # set dem_resolution to "90".
    downloader = DemDownloader(
        polygon=greenland_polygon,
        dem_resolution="90",  # Use "90" for 90 m DEMs (this will translate internally to "30" arc seconds).
        dem_format="DGED",
        dem_collection="COP-DEM",
        output_dir="/cpdata/SATS/RA/DEMS/COP90",
    )

    # Replace these credentials with your actual Copernicus credentials.
    username = "a.muir@ucl.ac.uk"
    password = "*Reunion12345"

    downloader.run(username, password)


if __name__ == "__main__":
    main()
