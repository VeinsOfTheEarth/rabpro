#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone

import ee
import requests

CATALOG_URL = "https://earthengine-stac.storage.googleapis.com/catalog/catalog.json"


def parse_date(timestamp):
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )


def parse_url(url, deprecated, verbose):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"{url} returned error status code {response.status_code}")
            return None

        r = response.json()

        # Added by JS to account for an apparently new Catalog structure?
        # The catalog urls no longer point directly to the asset metadata
        # url, but must be unpacked. It is possible that multiple assets be
        # contained in the same Catalog entry, so must loop over all 'child'
        # assets using their direct url.
        direct_url = []
        for link in r["links"]:
            if "rel" in link:
                if link["rel"] == "child":
                    direct_url.append(link["href"])

        assets = []  # for storing assets
        for durl in direct_url:
            response = requests.get(durl)
            if response.status_code != 200:
                print(f"{url} returned error status code {response.status_code}")
                continue

            r = response.json()
            gee_id = r["id"]
            gee_name = r["title"]
            gee_type = r["gee:type"]

            if not deprecated and "deprecated" in r and r["deprecated"]:
                if verbose:
                    print(f"Skipping {gee_id} (deprecated)")
                continue

            # Ignoring anything that's not an Image or ImageCollection
            # i.e. skipping featureCollections
            if gee_type == "image_collection":
                img = ee.ImageCollection(gee_id)
            elif gee_type == "image":
                img = ee.Image(gee_id)
            else:
                if verbose:
                    print(f"Skipping {gee_id} (not an Image or ImageCollection)")
                continue

            # Get date range information
            try:
                start_timestamp, end_timestamp = img.get("date_range").getInfo()
                gee_start = parse_date(start_timestamp)
                gee_end = parse_date(end_timestamp)
            except Exception:
                # Fallback method to compute the date range
                if verbose:
                    print(f"Fallback date method for {gee_id}")

                gee_start = r["extent"]["temporal"]["interval"][0][0].split("T")[0]
                if r["extent"]["temporal"]["interval"][0][1] is not None:
                    gee_end = r["extent"]["temporal"]["interval"][0][1].split("T")[0]
                else:
                    if verbose:
                        print(f"No end date found for {gee_id}. Using today's date.")
                    gee_end = datetime.now().strftime("%Y-%m-%d")

            if gee_type == "image_collection":
                l = img.toList(2)
                if img.toList(2).length().getInfo() > 1:
                    # Checking the length of the generated list is much faster than
                    # checking the length of the dataset
                    img = ee.Image(l.get(1))
                else:
                    img = img.first()

            # Get band, units, and resolution information

            # Account for COPERNICUS/S1_GRD edge case with "sar:bands"
            bandkey = "sar:bands" if "eo:bands" not in r["summaries"] else "eo:bands"
            bands = r["summaries"][bandkey]
            gee_bands = {}
            for band in bands:
                unit = band["gee:unit"] if "gee:unit" in band else None
                gee_bands[band["name"]] = {"units": unit}

            try:
                bandslist = ee.List(list(gee_bands.keys()))
                get_resolution = lambda b: ee.List(
                    [b, img.select([b]).projection().nominalScale()]
                )
                resolutions = bandslist.map(get_resolution).getInfo()
                for band, resolution in resolutions:
                    gee_bands[band]["resolution"] = resolution
            except Exception as e:
                print(e, gee_id)

            asset = {
                "id": gee_id,
                "name": gee_name,
                "start_date": gee_start,
                "end_date": gee_end,
                "type": gee_type,
                "bands": gee_bands,
            }
            assets.append(asset)
        return assets

    except Exception as e:
        print("Error:", e, gee_id)
        return None


def ee_catalog(deprecated, verbose):
    catalog = []
    obj = requests.get(CATALOG_URL).json()

    for assets in obj["links"]:
        try:
            if assets["rel"] == "child":
                asset = parse_url(assets["href"], deprecated, verbose)
                if asset is not None:
                    catalog.extend(asset)

        except Exception as e:
            print(e)

    filepath = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "gee_datasets.json"
    )
    with open(filepath, "w") as f:
        json.dump(catalog, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse GEE Data Catalog metadata")

    parser.add_argument(
        "-d",
        "--deprecated",
        default=False,
        action="store_true",
        help="include deprecated GEE assets",
    )

    parser.add_argument("--verbose", default=False, action="store_true")

    args = parser.parse_args()

    ee.Initialize(project=os.getenv("EARTH_ENGINE_PROJECT"))
    print("Parsing assets...")
    ee_catalog(args.deprecated, args.verbose)
    print("Parsing completed.")
