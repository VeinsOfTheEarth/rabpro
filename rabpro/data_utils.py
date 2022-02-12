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


def merit_hydro(merit_tile, username, password, proxy=None, clean=True, datapath=None):
    """Download MERIT Hydro

    Parameters
    ----------
    merit_tile : str
        MERIT Hydro tile identifier, e.g. "s30e150". See all possible tiles
        at http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/ under
        "Download"
    username : str
        Register at http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/ to
        access the username.
    password : str        
        Register at http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/ to
        access the password.
    proxy : str, optional
        Pass a proxy to requests.get, by default None
    clean : bool, optional
        Set False to skip overwrite of existing files, by default True
    datapath : str, optional
        Manually specify a location on the local filesystem, by default None


    Raises
    ------
    ValueError
        if a url could not be reached, such as with an invalid tile identifier
    """
    baseurl = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/"

    if proxy is not None:
        response = requests.get(baseurl, proxies={"http": proxy})
    else:
        response = requests.get(baseurl)
    soup = BeautifulSoup(response.text, "html.parser")
    urls = [
        x["href"][2:] for x in soup.findAll("a", text=re.compile(merit_tile), href=True)
    ]
    # The [2:] gets rid of the "./" in the URL

    if len(urls) == 0:
        raise ValueError(f"No tile matching '{merit_tile}' found.")

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

_HYDROBASINS_ZIP = "1NLJUEWhJ9A4y47rcGYv_jWF1Tx2nLEO9"

def download_hydrobasins(proxy=None, datapath=None):
    """Download HydroBASINS

    Parameters
    ----------
    proxy : str, optional
        Pass a proxy to requests.get, by default None
    datapath : str, optional
        Directory to download and unzip HydroBasins data into; does not include
        filename. By default None

    Examples
    --------
    .. code-block:: python

        from rabpro import data_utils
        data_utils.hydrobasins()    
    """

    if datapath is None:
        datapath, _ = _path_generator_util(None, None)
    filename = datapath / 'HydroBasins.zip'
    if os.path.isfile(filename):
        os.remove(filename)
    print('Downloading HydroBasins zip file (562 MB)...')
    _download_file_from_google_drive(_HYDROBASINS_ZIP, filename, proxy=None)
    
    # Check that filesize matches expected
    fsize = os.path.getsize(filename)
    if fsize != 562761977:
        print('Full zip file was not successfully downloaded. Check proxy?')
        os.remove(filename)
        return
    
    # Unzip the file
    print('Unzipping HydroBasins zip file...')
    path_hb_dir = datapath / 'HydroBasins'
    if os.path.isdir(path_hb_dir):
        shutil.rmtree(path_hb_dir)
    os.mkdir(path_hb_dir)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(datapath)
    
    # Delete zip file
    os.remove(filename)
    print('Done.')


def _download_file_from_google_drive(id_file, destination, proxy=None):
    """
    From https://stackoverflow.com/a/39225272/8195528.
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
    
        return None
    
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
    
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id_file }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id_file, 'confirm' : token }
        if proxy is None:
            response = session.get(URL, params = params, stream = True)
        else:
            response = session.get(URL, params = params, stream = True, proxies={"https": proxy})
            
    save_response_content(response, destination)    


# def _get_file(filename, url, proxy=None, clean=True):

#     if not clean:
#         if os.path.isfile(filename):
#             return

#     print(f"Downloading '{url}' into '{filename}'")

#     if proxy is not None:
#         r = requests.get(url, stream=True, proxies={"https": proxy})
#     else:
#         r = requests.get(url, stream=True)

#     total_size = int(r.headers.get("content-length", 0))

#     if r.status_code != 200:
#         print(f"{url} failed with status code {r.status_code}")
#         return

#     with open(filename, "wb") as f:
#         tqdmbar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)
#         for chunk in r.iter_content(4 * 1024):
#             if chunk:
#                 tqdmbar.update(len(chunk))
#                 f.write(chunk)
#         tqdmbar.close()

#     return None


