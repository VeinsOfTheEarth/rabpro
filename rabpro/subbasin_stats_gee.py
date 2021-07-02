import os
import ee
import json
import pandas as pd
import numpy as np
import utils as ru
from datetime import date


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
    resolution : int
        Desired resolution in meters of the calculation over the dataset
    start : str
        Desired start date of data in ISO format: YYYY-MM-DD
    end : str
        Desired end date of data in ISO format: YYYY-MM-DD
    stats : list
        List of desired stats to compute: min, max, range, mean, count, std, 
        sum and percentiles in the following format: pct1, pct90, pct100, etc.
    mask: boolean
        Whether or not to mask out water in the dataset using the Global 
        Surface Water occurrence band

    """

    def __init__(
        self, data_id, band, resolution=None, start=None, end=None, stats=None, mask=False
    ):
        self.data_id = data_id
        self.band = band
        self.resolution = resolution
        self.start = start
        self.end = end
        self.stats = stats if stats is None else []
        self.mask = mask


def main(sb_inc_gdf, dataset_list, verbose=False, folder=None):
    """Compute subbasin statistics for each raster in control file"""
    # For improved speed, rather than computing statistics for each subbasin,
    # fetch values for the subbasins, then compute statistics by combining
    # the values as we move downstream to the next subbasin

    # Dictionary for determining which rasters and statistics to compute
    control = get_controls(dataset_list)
    ee.Initialize()

    # Create water occurence mask
    occ_mask = ee.Image("JRC/GSW1_0/GlobalSurfaceWater").select("occurrence").lt(90)

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
            reducer = reducer.combine(reducer2=ee.Reducer.min(len(r["bands"])), sharedInputs=True)
        elif "max" in d.stats:
            reducer = reducer.combine(reducer2=ee.Reducer.max(len(r["bands"])), sharedInputs=True)

        if "std" in d.stats:
            reducer = reducer.combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        if "sum" in d.stats:
            reducer = reducer.combine(reducer2=ee.Reducer.sum(), sharedInputs=True)

        pct_list = [int(pct[3:]) for pct in d.stats if pct[:3] == "pct"]
        if pct_list:
            reducer = reducer.combine(reducer2=ee.Reducer.percentile(pct_list), sharedInputs=True)

        def map_func(img):
            # TODO: change to reduceRegion or simplify geometries
            return img.reduceRegions(
                collection=featureCollection, reducer=reducer, scale=r["resolution"]
            )

        reducedFC = imgcol.map(map_func)
        table = reducedFC.flatten()

        def range_func(feat):
            # TODO: Change column names later depending on output formatting
            return feat.set("range", feat.get("max") - feat.get("min"))

        # Map across feature collection and use min and max to compute range
        if "range" in d.stats:
            table = table.map(range_func)

        # print(table.getDownloadURL(filetype='csv'))
        # TODO: Add selectors to export and change file name
        task = ee.batch.Export.table.toDrive(
            collection=table, description=f"{d.data_id}_{d.band}", folder=folder, fileFormat="csv"
        )
        task.start()


def get_controls(datasets):
    """
    Prepare paths and parameters for computing subbasin raster stats.
    Takes in list of user specified datasets.
    """

    # Load raster metadata file
    # TODO: Change to json file + user json file
    datapaths = ru.get_datapaths()
    with open(datapaths["metadata"]) as json_file:
        datadict = {d["id"]: d for d in json.load(json_file)}

    control = []
    for d in datasets:
        # TODO: Improve warnings:
        # - Add resolution value check
        # - Use proper warnings module?
        # - dates only support ISO format (YYYY-MM-DD)
        if d.data_id not in datadict:
            print(f"Warning: invalid data ID provided: {d.data_id}")
            continue

        gee_dataset = datadict[d.data_id]

        if d.band not in gee_dataset:
            print(f"Warning: invalid data band provided: {d.data_id}:{d.band}")
            continue

        if d.resolution is None:
            d.resolution = gee_dataset["resolution"]

        if d.start is None or date.fromisoformat(d.start) < date.fromisoformat(
            gee_dataset["start"]
        ):
            d.start = gee_dataset["start"]
            print(f"Warning: overrode start date for {d.data_id}:{d.band}")

        if d.end is None or date.fromisoformat(d.end) > date.fromisoformat(gee_dataset["end"]):
            d.end = gee_dataset["end"]
            print(f"Warning: overrode end date for {d.data_id}:{d.band}")

        d.stats = set(d.stats + ["count", "mean"])
        control.append(d)

    return control
