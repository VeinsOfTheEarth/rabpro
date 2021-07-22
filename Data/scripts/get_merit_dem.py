#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import tarfile
import urllib.parse

import appdirs
import requests
import tqdm
from bs4 import BeautifulSoup

merit_hydro_paths = {
    "elv": f"DEM{os.sep}MERIT_ELEV_HP",
    "dir": f"DEM{os.sep}MERIT_FDR",
    "upa": f"DEM{os.sep}MERIT_UDA",
    "wth": f"DEM{os.sep}MERIT_WTH",
    "dem": f"DEM{os.sep}MERIT103",
}

datapath = appdirs.user_data_dir("rabpro", "jschwenk")


def merit_dem(target, username, password):
    baseurl = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/"
    filename = f"dem_tif_{target}.tar"

    response = requests.get(baseurl)
    soup = BeautifulSoup(response.text, "html.parser")
    url = [x["href"][2:] for x in soup.findAll("a", text=re.compile(filename), href=True)][0]

    url = baseurl + url
    filename = os.path.join(datapath, merit_hydro_paths["dem"], filename)
    print(f"Downloading '{url}' into '{filename}'")
    download_file(url, filename, username, password)
    print()


def merit_hydro(target, username, password):
    baseurl = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/"

    response = requests.get(baseurl)
    soup = BeautifulSoup(response.text, "html.parser")
    urls = [x["href"][2:] for x in soup.findAll("a", text=re.compile(target), href=True)]
    # The [2:] gets rid of the "./" in the URL

    for urlfile in urls:
        url = baseurl + urlfile
        filename = os.path.basename(urllib.parse.urlparse(url).path)

        if filename[:3] not in merit_hydro_paths:
            continue

        filename = os.path.join(datapath, merit_hydro_paths[filename[:3]], filename)

        print(f"Downloading '{url}' into '{filename}'")
        download_file(url, filename, username, password)
        print()


def download_file(url, filename, username, password):
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
    tf = tarfile.open(filename)
    tf.extractall(os.path.dirname(filename))

    tar_dir = filename[:-4]
    files = os.listdir(tar_dir)
    for f in files:
        shutil.move(os.path.join(tar_dir, f), os.path.join(os.path.dirname(tar_dir), f))

    os.rmdir(tar_dir)
    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("target", type=str, help="MERIT tile (e.g. 'n30w090')")

    parser.add_argument("username", type=str, help="MERIT username")

    parser.add_argument("password", type=str, help="MERIT password")

    args = parser.parse_args()
    merit_hydro(args.target, args.username, args.password)
    merit_dem(args.target, args.username, args.password)
