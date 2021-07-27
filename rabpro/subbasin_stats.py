"""
Subbasin Statistics (subbasin_stats.py)
=======================================

Computes subbasin statistics using Google Earth Engine.
"""

import json
import os
from datetime import date

import ee
import numpy as np
import pandas as pd

from rabpro import utils as ru


class Dataset:
    """
    Represents one band of a GEE dataset with parameters specifying how to
    compute statistics.

    Attributes
    ----------
    data_id : str
        Google Earth Engine dataset asset id
    band : str
        Google Earth Engine dataset band name
    resolution : int, optional
        Desired resolution in meters of the calculation over the dataset.
        Defaults to native resoltion of dataset.
    start : str, optional
        Desired start date of data in ISO format: YYYY-MM-DD. Defaults to
        dataset start.
    end : str, optional
        Desired end date of data in ISO format: YYYY-MM-DD. Defaults to dataset
        end.
    stats : list, optional
        List of desired stats to compute: min, max, range, mean, count, std,
        sum, and percentiles in the following format: pct1, pct90, pct100, etc.
        Computes ["count", "mean"] by default.
    mask : bool, optional
        Whether or not to mask out water in the dataset using the Global Surface
        Water occurrence band. By default False.

    """

    def __init__(
        self, data_id, band, resolution=None, start=None, end=None, stats=None, mask=False
    ):
        self.data_id = data_id
        self.band = band
        self.resolution = resolution
        self.start = start
        self.end = end
        self.stats = stats if stats is not None else []
        self.mask = mask


def main(sb_inc_gdf, dataset_list, reducer_funcs=None, folder=None, verbose=False, test=False):
    """
    Compute subbasin statistics for each dataset and band specified.

    Attributes
    ----------
    sb_inc_gdf : GeoDataFrame
        Table of subbasin geometries.
    dataset_list : list of Datasets
        List of Dataset objects to compute statistics over.
    reducer_funcs : list of functions, optional
        List of functions to apply to each feature over each dataset. Each
        function should take in an ee.Feature() object. For example, this is how
        the function and header are applied on a feature:
        feature.set(f.__name__, function(feature))
    verbose : bool, optional
        By default False.
    folder : str, option
        Google Drive folder to store results in, by default top-level root.

    """

    # Dictionary for determining which rasters and statistics to compute
    control = _get_controls(dataset_list)
    ee.Initialize()

    # Create water occurence mask
    occ_mask = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select("occurrence").lt(90)

    # Convert GeoDataFrame to ee.Feature objects
    features = []
    for i in range(sb_inc_gdf.shape[0]):
        geom = sb_inc_gdf.iloc[i : i + 1, :]
        jsonDict = json.loads(geom.to_json())
        geojsonDict = jsonDict["features"][0]
        features.append(ee.Feature(geojsonDict))
    featureCollection = ee.FeatureCollection(features)

    # For each raster
    for d in control:
        if d.type == "image":
            imgcol = ee.ImageCollection(ee.Image(d.data_id).select(d.band))
        else:
            imgcol = ee.ImageCollection(d.data_id).select(d.band).filterDate(d.start, d.end)

        if verbose:
            print(f"Computing subbasin stats for {d.data_id}...")

        # Add threshold mask to image using GSW occurrence band
        if d.mask:
            imgcol = imgcol.map(lambda img: img.updateMask(occ_mask))

        # Generate reducer - mean and count always computed
        reducer = ee.Reducer.count().combine(reducer2=ee.Reducer.mean(), sharedInputs=True)

        if ("min" in d.stats and "max" in d.stats) or "range" in d.stats:
            reducer = reducer.combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
        elif "min" in d.stats:
            reducer = reducer.combine(reducer2=ee.Reducer.min(), sharedInputs=True)
        elif "max" in d.stats:
            reducer = reducer.combine(reducer2=ee.Reducer.max(), sharedInputs=True)

        if "stdDev" in d.stats:
            reducer = reducer.combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        if "sum" in d.stats:
            reducer = reducer.combine(reducer2=ee.Reducer.sum(), sharedInputs=True)

        pct_list = [int(pct[3:]) for pct in d.stats if pct[:3] == "pct"]
        if pct_list:
            reducer = reducer.combine(reducer2=ee.Reducer.percentile(pct_list), sharedInputs=True)

        def map_func(img):
            # TODO: change to reduceRegion or simplify geometries
            return img.reduceRegions(
                collection=featureCollection, reducer=reducer, scale=d.resolution
            )

        reducedFC = imgcol.map(map_func)
        table = reducedFC.flatten()

        def range_func(feat):
            # TODO: Change to aggregateArray - more efficient?
            return feat.set("range", feat.getNumber("max").subtract(feat.getNumber("min")))

        # Map across feature collection and use min and max to compute range
        if "range" in d.stats:
            table = table.map(range_func)

        # Apply reducer functions
        if reducer_funcs is not None:
            for f in reducer_funcs:

                def reducer_func(feat):
                    return feat.set(f.__name__, f(feat))

                table = table.map(reducer_func)

        if test:
            data = table.getInfo()

        # TODO: Add selectors to export
        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=f"{d.data_id}__{d.band}".replace("/", "-"),
            folder=folder,
            fileFormat="csv",
        )

        task.start()

        if test:
            return data, task


def _get_controls(datasets):
    """
    Prepare paths and parameters for computing subbasin raster stats. Takes in
    list of user specified datasets.
    """

    # Load raster metadata file
    datapaths = ru.get_datapaths()
    with open(datapaths["gee_metadata"]) as json_file:
        datadict = {d["id"]: d for d in json.load(json_file)}

    if datapaths["user_gee_metadata"] is not None:
        with open(datapaths["user_gee_metadata"]) as json_file:
            user_datadict = {d["id"]: d for d in json.load(json_file)}

        # TODO switch to x | y notation in Python 3.9. Add try/except for this section?
        datadict = {**datadict, **user_datadict}  # merge dictionaries

    control = []
    for d in datasets:
        # TODO Use actual warnings module?
        if d.data_id not in datadict:
            print(f"Warning: invalid data ID provided: {d.data_id}")
            continue

        gee_dataset = datadict[d.data_id]

        if d.band not in gee_dataset["bands"]:
            print(f"Warning: invalid data band provided: {d.data_id}:{d.band}")
            continue

        if d.start is None:
            d.start = gee_dataset["start_date"]
        elif date.fromisoformat(d.start) < date.fromisoformat(gee_dataset["start_date"]):
            print(f"Warning: requested start date earlier than expected for {d.data_id}:{d.band}")

        if d.end is None:
            d.end = gee_dataset["end_date"]
        elif date.fromisoformat(d.end) > date.fromisoformat(gee_dataset["end_date"]):
            print(f"Warning: requested end date later than expected for {d.data_id}:{d.band}")

        d.stats = set(d.stats + ["count", "mean"])

        if "no_data" in gee_dataset["bands"][d.band]:
            d.no_data = gee_dataset["bands"][d.band]["no_data"]

        resolution = None
        if "resolution" in gee_dataset["bands"][d.band]:
            resolution = gee_dataset["bands"][d.band]["resolution"]
        if d.resolution is None:
            d.resolution = resolution
        if d.resolution and resolution and d.resolution < resolution:
            print("Warning: requested resolution is less than the native raster resolution")

        d.type = gee_dataset["type"]

        control.append(d)

    return control
