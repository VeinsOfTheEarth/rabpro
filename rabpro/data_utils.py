"""
Data utility functions (data_utils.py)
======================================

"""

import json
import os
import re
import shutil
import tarfile
import zipfile
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import appdirs
import requests
import tqdm
import gdown
from bs4 import BeautifulSoup

from rabpro import utils as ru

_PATH_CONSTANTS = {
    "HydroBasins1": f"HydroBasins{os.sep}level_one",
    "HydroBasins12": f"HydroBasins{os.sep}level_twelve",
    "DEM_fdr": f"MERIT_Hydro{os.sep}MERIT_FDR{os.sep}MERIT_FDR.vrt",
    "DEM_uda": f"MERIT_Hydro{os.sep}MERIT_UDA{os.sep}MERIT_UDA.vrt",
    "DEM_elev_hp": f"MERIT_Hydro{os.sep}MERIT_ELEV_HP{os.sep}MERIT_ELEV_HP.vrt",
    "DEM_width": f"MERIT_Hydro{os.sep}MERIT_WTH{os.sep}MERIT_WTH.vrt",
}

CATALOG_URL = "https://raw.githubusercontent.com/VeinsOfTheEarth/rabpro/main/Data/gee_datasets.json"
CATALOG_URL_USER = "https://raw.githubusercontent.com/VeinsOfTheEarth/rabpro/main/Data/user_gee_datasets.json"

_GEE_CACHE_DAYS = 1

HYDROBASINS_ZIP_ID = "1NLJUEWhJ9A4y47rcGYv_jWF1Tx2nLEO9"

merit_hydro_paths = {
    "elv": f"MERIT_Hydro{os.sep}MERIT_ELEV_HP",
    "dir": f"MERIT_Hydro{os.sep}MERIT_FDR",
    "upa": f"MERIT_Hydro{os.sep}MERIT_UDA",
    "wth": f"MERIT_Hydro{os.sep}MERIT_WTH",
}

hydrobasins_paths = {
    "HydroBasins1": f"HydroBasins{os.sep}level_one",
    "HydroBasins12": f"HydroBasins{os.sep}level_twelve",
}


def _path_generator_util(datapath, configpath):
    if datapath is None:
        try:
            datapath = Path(os.environ["RABPRO_DATA"])
        except:
            datapath = Path(appdirs.user_data_dir("rabpro", "rabpro"))

    if configpath is None:
        try:
            configpath = Path(os.environ["RABPRO_CONFIG"])
        except:
            configpath = Path(appdirs.user_config_dir("rabpro", "rabpro"))

    return datapath, configpath


def create_file_structure(datapath=None, configpath=None):

    datapath, configpath = _path_generator_util(datapath, configpath)

    os.makedirs(configpath, exist_ok=True)
    for key in merit_hydro_paths:
        os.makedirs(os.path.join(datapath, merit_hydro_paths[key]), exist_ok=True)

    for key in hydrobasins_paths:
        os.makedirs(os.path.join(datapath, hydrobasins_paths[key]), exist_ok=True)


def delete_file_structure(datapath=None, configpath=None):
    datapath, configpath = _path_generator_util(datapath, configpath)

    shutil.rmtree(configpath, ignore_errors=True)
    shutil.rmtree(datapath, ignore_errors=True)


def create_datapaths(datapath=None, configpath=None, reset_user_metadata=False):
    datapath, configpath = _path_generator_util(datapath, configpath)

    datapaths = {key: str(datapath / Path(val)) for key, val in _PATH_CONSTANTS.items()}

    datapaths["HydroBasins_root"] = str(datapath / Path("HydroBasins"))
    datapaths["MERIT_root"] = str(datapath / Path("MERIT_Hydro"))
    datapaths["root"] = str(datapath)

    gee_metadata_path = datapath / "gee_datasets.json"
    datapaths["gee_metadata"] = str(gee_metadata_path)

    # User defined GEE datasets
    user_gee_metadata_path = configpath / "user_gee_datasets.json"
    datapaths["user_gee_metadata"] = str(user_gee_metadata_path)
    if reset_user_metadata:
        if os.path.isfile(datapaths["user_gee_metadata"]):
            os.remove(datapaths["user_gee_metadata"])
    if not user_gee_metadata_path.is_file():
        try:
            https_proxy = os.environ["HTTPS_PROXY"]
            response = requests.get(CATALOG_URL_USER, proxies={"https": https_proxy})
        except:
            response = requests.get(CATALOG_URL_USER)

        if response.status_code == 200:
            r = response.json()
            print(datapaths["user_gee_metadata"])
            with open(datapaths["user_gee_metadata"], "w") as f:
                json.dump(r, f, indent=4)

    return datapaths


def does_merit_exist(datapaths):
    """
    Checks if any MERIT tiles are in the MERIT data directories. Also checks
    if the vrts have been built. Returns the number of MERIT layers that have
    data (maximum of 4).
    """
    vrts_exist = 0
    geotiffs_exist = 0
    dem_files = [k for k in datapaths.keys() if "DEM" in k]
    for df in dem_files:
        geotiffs = [
            f
            for f in os.listdir(os.path.dirname(datapaths[df]))
            if f.split(".")[-1] == "tif"
        ]
        if len(geotiffs) > 0:
            geotiffs_exist = geotiffs_exist + 1
        if os.path.isfile(datapaths[df]) is True:
            vrts_exist = vrts_exist + 1

    return geotiffs_exist, vrts_exist


def does_hydrobasins_exist(datapaths):
    """
    Checks if level 1 and level 12 HydroBasins data are available.
    """
    lev1, lev12 = False, False
    if (
        os.path.isfile(
            os.path.join(datapaths["HydroBasins1"], "hybas_all_lev01_v1c.shp")
        )
        is True
    ):
        lev1 = True
    if (
        os.path.isfile(
            os.path.join(datapaths["HydroBasins12"], "hybas_af_lev12_v1c.shp")
        )
        is True
    ):
        lev12 = True
    return lev1, lev12


def download_gee_metadata(datapath=None):
    datapath, _ = _path_generator_util(datapath, None)
    gee_metadata_path = datapath / "gee_datasets.json"

    # Download catalog JSON file
    if gee_metadata_path.is_file():
        mtime = datetime.fromtimestamp(gee_metadata_path.stat().st_mtime)
        delta = datetime.now() - mtime

    if not gee_metadata_path.is_file() or delta > timedelta(days=_GEE_CACHE_DAYS):
        try:
            response = requests.get(CATALOG_URL)
            if response.status_code == 200:
                r = response.json()
                with open(gee_metadata_path, "w") as f:
                    json.dump(r, f, indent=4)
        except:
            print(
                f"{CATALOG_URL} download error. Place manually into {gee_metadata_path}"
            )


def _download_file_from_google_drive(id_file, destination, proxy=None):
    """
    Self-explanatory.
    """

    url = "https://drive.google.com/uc?id={}".format(id_file)
    gdown.download(url, output=destination, proxy=proxy)

    return


def download_hydrobasins(datapath=None, proxy=None):
    """Downloads HydroBASINS to the proper location, then unzips and deletes
    zip file.

    Parameters
    ----------
    datapath : str or Path, optional
        Directory to download and unzip HydroBasins data into; does not include
        filename. If None, will use the rabpro default.
    proxy : str, optional
        Pass a proxy to requests.get, by default None

    Examples
    --------
    .. code-block:: python

        from rabpro import data_utils
        data_utils.hydrobasins()
    """

    if datapath is None:
        datapath, _ = _path_generator_util(None, None)
    else:  # Ensure the user-supplied datapath exists, else create
        if os.path.isdir(datapath) is False:
            os.path.mkdir(datapath)

    datapath = Path(datapath)

    filepath = datapath / "HydroBasins.zip"
    if os.path.isfile(filepath):
        os.remove(filepath)
    print("Downloading HydroBasins zip file (562 MB)...")
    _download_file_from_google_drive(HYDROBASINS_ZIP_ID, str(filepath), proxy=proxy)

    # Check that filesize matches expected
    fsize = os.path.getsize(filepath)
    if fsize != 562761977:
        hb_url = r"https://drive.google.com/file/d/1NLJUEWhJ9A4y47rcGYv_jWF1Tx2nLEO9/view?usp=sharing"
        print(
            "HydroBasins zip file was not successfully downloaded. Check proxy? You may also manually download the HydroBasins file from {} and unzip it to {}".format(
                hb_url, str(datapath)
            )
        )
        os.remove(filepath)
        return

    # Unzip the file
    print("Unzipping HydroBasins.zip...")
    path_hb_dir = datapath / "HydroBasins"
    if os.path.isdir(path_hb_dir):
        shutil.rmtree(path_hb_dir)
    os.mkdir(path_hb_dir)
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(datapath)

    # Delete zip file
    os.remove(filepath)
    print("Done.")

    return


def download_merit_hydro(merit_tile, username, password, proxy=None):
    """Downloads the four required rabpro layers for a given MERIT-Hydro
    30 degree x 30 degree tile. Each tile contains 5 degree x 5 degree geotiffs.
    Data are downloaded and untarred into the location and filestructure rabpro
    expects. Virtual rasters are rebuilt after downloading this data.

    Parameters
    ----------
    merit_tile : str
        The MERIT-Hydro tile to fetch. e.g. "n45w100"
    username : str
        The MERIT-Hydro username (must request from MERIT-Hydro developers).
    password : str
        The MERIT-Hydro password (must request from MERIT-Hydro developers).
    proxy : str, optional
        A proxy to pass to requests Session.

    """

    datapaths = ru.get_datapaths()
    merit_hydro_paths = {
        "elv": "MERIT_ELEV_HP",
        "dir": "MERIT_FDR",
        "upa": "MERIT_UDA",
        "wth": "MERIT_WTH",
    }

    baseurl = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/"

    session = requests.Session()
    if proxy is not None:
        session.proxies.update({"https": proxy})
    else:
        session.proxies = {}
    response = session.get(baseurl)

    soup = BeautifulSoup(response.text, "html.parser")
    # The [2:] gets rid of the "./" in the URL
    urls = [
        x["href"][2:] for x in soup.findAll("a", text=re.compile(merit_tile), href=True)
    ]

    # Download and untar the files
    for urlfile in urls:
        url = baseurl + urlfile
        filename = os.path.basename(urllib.parse.urlparse(url).path)

        if filename[:3] not in merit_hydro_paths:
            continue

        filename = os.path.join(
            datapaths["MERIT_root"], merit_hydro_paths[filename[:3]], filename
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        download_file(url, filename, username, password, proxy)

    # Rebuild virtual rasters to include new geotiffs
    ru.build_virtual_rasters(datapaths, skip_if_exists=False, verbose=True)

    return


def download_tar_file(url, filename, username, password, proxy=None, clean=True):
    if not clean:
        if os.path.isfile(filename):
            return

    print(f"Downloading '{url}' into '{filename}'")

    if proxy is not None:
        r = requests.get(
            url, auth=(username, password), stream=True, proxies={"http": proxy}
        )
    else:
        r = requests.get(url, auth=(username, password), stream=True)

    total_size = int(r.headers.get("content-length", 0))

    if r.status_code != 200:
        print(f"{url} failed with status code {r.status_code}")
        return

    with open(filename, "wb") as f:
        tqdmbar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)
        for chunk in r.iter_content(4 * 1024):
            if chunk:
                tqdmbar.update(len(chunk))
                f.write(chunk)
        tqdmbar.close()

    # Extract TAR archive and remove artifacts
    with tarfile.open(filename) as tf:
        tf.extractall(os.path.dirname(filename))

    tar_dir = filename[:-4]
    files = os.listdir(tar_dir)
    for f in files:
        shutil.move(os.path.join(tar_dir, f), os.path.join(os.path.dirname(tar_dir), f))

    if not clean:
        os.rmdir(tar_dir)
        os.remove(filename)


def download_file(url, filename, username, password, proxy=None):

    # Skip downloading if the file already exists
    if os.path.isfile(filename):
        return

    print(f"Downloading '{url}' into '{filename}'")

    session = requests.Session()
    if proxy is not None:
        session.proxies.update({"https": proxy})
    else:
        session.proxies = {}

    r = session.get(url, auth=(username, password), stream=True)

    total_size = int(r.headers.get("content-length", 0))

    if r.status_code != 200:
        print(f"{url} failed with status code {r.status_code}")
        return

    with open(filename, "wb") as f:
        tqdmbar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)
        for chunk in r.iter_content(4 * 1024):
            if chunk:
                tqdmbar.update(len(chunk))
                f.write(chunk)
        tqdmbar.close()

    # Extract TAR archive and remove artifacts
    with tarfile.open(filename) as tf:
        tf.extractall(os.path.dirname(filename))

    tar_dir = filename[:-4]
    files = os.listdir(tar_dir)
    for f in files:
        shutil.move(os.path.join(tar_dir, f), os.path.join(os.path.dirname(tar_dir), f))

    os.rmdir(tar_dir)
    os.remove(filename)

    return
