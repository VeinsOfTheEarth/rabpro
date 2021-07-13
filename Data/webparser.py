import json
from datetime import datetime, timezone

import ee
import requests

CATALOG_URL = "https://earthengine-stac.storage.googleapis.com/catalog/catalog.json"


def parse_date(timestamp):
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def parse_url(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"{url} returned error status code {response.status_code}")
            return None

        r = response.json()

        gee_id = r["id"]
        gee_name = r["title"]
        gee_type = r["gee:type"]

        if "deprecated" in r and r["deprecated"]:
            print(f"Skipping {r['id']} (deprecated)")
            return None

        # TODO Ignoring anything that's not an Image or ImageCollection for now
        if gee_type == "image_collection":
            img = ee.ImageCollection(gee_id)
        elif gee_type == "image":
            img = ee.Image(gee_id)
        else:
            return None

        # Get date range information
        try:
            start_timestamp, end_timestamp = img.get("date_range").getInfo()
            gee_start = parse_date(start_timestamp)
            gee_end = parse_date(end_timestamp)
        except Exception as e:
            # Fallback method to compute the date range
            gee_start = r["extent"]["temporal"]["interval"][0][0].split("T")[0]
            if r["extent"]["temporal"]["interval"][0][1] is not None:
                gee_end = r["extent"]["temporal"]["interval"][0][1].split("T")[0]
            else:
                gee_end = datetime.now().strftime("%Y-%m-%d")

        if gee_type == "image_collection":
            l = img.toList(2)
            if img.toList(2).length().getInfo() > 1:
                img = ee.Image(l.get(1))
            else:
                img = img.first()

        # Get band, units, and resolution information
        bands = r["summaries"]["eo:bands"]
        gee_bands = {}
        for band in bands:
            unit = band["gee:unit"] if "gee:unit" in band else None
            gee_bands[band["name"]] = {"units": unit}

        bandslist = ee.List(list(gee_bands.keys()))
        resolutions = bandslist.map(lambda b : ee.List([b, img.select([b]).projection().nominalScale()])).getInfo()
        for band, resolution in resolutions:
            gee_bands[band]["resolution"] = resolution

        asset = {
            "id": gee_id,
            "name": gee_name,
            "start_date": gee_start,
            "end_date": gee_end,
            "type": gee_type,
            "bands": gee_bands,
        }
        return asset

    except Exception as e:
        print(e, gee_id)
        return None


def ee_catalog():
    catalog = []
    obj = requests.get(CATALOG_URL).json()

    for assets in obj["links"]:
        try:
            if assets["rel"] == "child":
                asset = parse_url(assets["href"])
                if asset is not None:
                    catalog.append(asset)

        except Exception as e:
            print(e)

    # TODO change file path
    with open("gee_catalog.json", "w") as f:
        json.dump(catalog, f, indent=4)


if __name__ == "__main__":
    ee.Initialize()
    print("Parsing assets...")
    ee_catalog()
    print("Parsing completed.")
