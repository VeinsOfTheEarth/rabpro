"""
Data utility functions (data_utils.py)
======================================

"""

import json
import os
import re
import shutil
import tarfile
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import appdirs
import requests
import tqdm
from bs4 import BeautifulSoup

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


def create_datapaths(datapath=None, configpath=None, reset_user_metadata=False):
    datapath, configpath = _path_generator_util(datapath, configpath)

    datapaths = {key: str(datapath / Path(val)) for key, val in _PATH_CONSTANTS.items()}
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


def merit_hydro(target, username, password, proxy=None, clean=True, datapath=None):
    baseurl = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/"

    if proxy is not None:
        response = requests.get(baseurl, proxies={"http": proxy})
    else:
        response = requests.get(baseurl)
    soup = BeautifulSoup(response.text, "html.parser")
    urls = [
        x["href"][2:] for x in soup.findAll("a", text=re.compile(target), href=True)
    ]
    # The [2:] gets rid of the "./" in the URL

    if len(urls) == 0:
        raise ValueError(f"No tile matching '{target}' found.")

    datapath, _ = _path_generator_util(datapath, None)

    for urlfile in urls:
        url = baseurl + urlfile
        filename = os.path.basename(urllib.parse.urlparse(url).path)

        if filename[:3] not in merit_hydro_paths:
            continue

        filename = os.path.join(datapath, merit_hydro_paths[filename[:3]], filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        download_tar_file(url, filename, username, password, proxy, clean)


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


_HYDROBASINS_1_IDS = {
    "all": {
        "dbf": "1duRlrrHTciKn7gM4qogumZ4OhqrB0Ggq",
        "prj": "1fSAUKiFbfYb8-rLqiG1Epo3dMNLBOMHh",
        "qpj": "1ZMCrzYUJuxORxNwkQjL1qvFHODS64WBu",
        "shp": "1ev5Md5d2RwzpTRfpJ6SmCkYPf_7821b2",
        "shx": "15-fa27DPnioY9kDzgKHQdaSxingSGhCJ",
    }
}

_HYDROBASINS_12_IDS = {
    "ar": {
        "dbf": "19tHCft5jIDoSAdS_PWVMXN1kUGEdCCIl",
        "prj": "1WbSigVZ3up7EUsbTGsFjBoi3DsK5r5xg",
        "shp": "1s-L6YEdDN1-mRStamvs4nE9X8GolMv4q",
        "shx": "1LCeZpkczJqV1eyCm_ArJdZCAONwCy48I",
    },
    "as": {
        "dbf": "1ZXqPOvs-LTk_PheF0iRj9KTY9WNosADO",
        "prj": "1Ds4U42MyFz4xioDiBv7erRI3tpn_VmH-",
        "shp": "12PNAOZrWb3U0xggLH0fku1g3MB7IZiGe",
        "shx": "1pikNKG5ZUl90gtHACs3M6PksCEZRjmHR",
    },
}


def _get_file(filename, url, proxy=None, clean=True):

    if not clean:
        if os.path.isfile(filename):
            return

    print(f"Downloading '{url}' into '{filename}'")

    if proxy is not None:
        r = requests.get(url, stream=True, proxies={"https": proxy})
    else:
        r = requests.get(url, stream=True)

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

    return None


def _get_domain(domain, level, proxy=None, clean=True, datapath=None):

    filebase = "hybas_" + domain + "_lev" + level + "_v1c."
    urlbase = "https://drive.google.com/uc?export=download&id="
    datapath, _ = _path_generator_util(datapath, None)
    pathbase = datapath / Path(_PATH_CONSTANTS["HydroBasins" + level.strip("0")])
    os.makedirs(pathbase, exist_ok=True)

    if level == "01":
        id_dict = _HYDROBASINS_1_IDS

    if level == "12":
        id_dict = _HYDROBASINS_12_IDS

    subdict = id_dict[domain]

    for ext in subdict:
        filename = pathbase / Path(filebase + ext)
        url = urlbase + subdict[ext]
        print((filename, url))
        _get_file(filename, url, clean=clean, proxy=proxy)

    return None


def hydrobasins(proxy=None, clean=True, datapath=None):
    """Download HydroBASINS

    Parameters
    ----------
    proxy : str, optional
        Pass a proxy to requests.get, by default None
    clean : bool, optional
        Set False to skip overwrite of existing files, by default True
    datapath : str, optional
        Manually specify a location on the local filesystem, by default None

    from rabpro import data_utils
    data_utils.hydrobasins(clean=False)
    """

    [
        _get_domain(domain, "01", clean=clean, proxy=proxy, datapath=datapath)
        for domain in _HYDROBASINS_1_IDS
    ]

    [
        _get_domain(domain, "12", clean=clean, proxy=proxy, datapath=datapath)
        for domain in _HYDROBASINS_12_IDS
    ]

