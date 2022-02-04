"""
Subbasin Statistics (subbasin_stats.py)
=======================================

Computes subbasin statistics using Google Earth Engine.
"""

import json
import pandas as pd
from datetime import date

import ee

from rabpro import utils as ru


class Dataset:
    """
    Represents one band of a GEE dataset with parameters specifying how to
    compute statistics.

    Attributes
    ----------
    data_id : str
        Google Earth Engine dataset asset id
    band : str, optional
        Google Earth Engine dataset band name
    resolution : int, optional
        Desired resolution in meters of the calculation over the dataset.
        Defaults to native resolution of the dataset.
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
    time_stats : list, optional
        List of desired stats to compute through time.
    mask : bool, optional
        Whether or not to mask out water in the dataset using the Global Surface
        Water occurrence band. By default False.

    """

    def __init__(
        self,
        data_id,
        band="None",
        resolution=None,
        start=None,
        end=None,
        stats=None,
        time_stats=None,
        mask=False,
    ):
        self.data_id = data_id
        self.band = band
        self.resolution = resolution
        self.start = start
        self.end = end
        self.stats = stats if stats is not None else []
        self.time_stats = time_stats if time_stats is not None else []
        self.mask = mask


def dataset_to_filename(data_id, band, tag=""):
    if tag == "":
        return f"{data_id}__{band}".replace("/", "-")
    else:
        return f"{data_id}__{band}".replace("/", "-") + "__" + tag


def format_gee(
    url_list,
    tag_list,
    col_drop_list=[],
    col_drop_defaults=["DA", "count", ".geo", "system:index"],
):
    df_list = [pd.read_csv(url) for url in url_list]

    def clean_gee(df, tag, col_drop_list):
        df = ru.drop_column_if_exists(df, col_drop_list + col_drop_defaults)
        df.columns = [tag + "_" + x for x in df.columns]
        return df

    res = [
        clean_gee(df, tag, col_drop_list=col_drop_list)
        for df, tag in zip(df_list, tag_list)
    ]

    return pd.concat(res, axis=1)


def compute(
    dataset_list,
    gee_feature_path=None,
    basins_gdf=None,
    reducer_funcs=None,
    folder=None,
    verbose=False,
    test=False,
    tag="",
):
    """
    Compute subbasin statistics for each dataset and band specified.

    Parameters
    ----------
    dataset_list : list of Datasets
        List of Dataset objects to compute statistics over.
    basins_gdf : GeoDataFrame
        Table of subbasin geometries.
    gee_feature_path : string
        Path to a GEE feature collection
    reducer_funcs : list of functions, optional
        List of functions to apply to each feature over each dataset. Each
        function should take in an ee.Feature() object. For example, this is how
        the function and header are applied on a feature:
        feature.set(f.__name__, function(feature))
    verbose : bool, optional
        By default False.
    folder : str, optional
        Google Drive folder to store results in, by default top-level root.
    tag : str, optional
        A string to append to files created on GDrive
    test : bool, optional
        Return results to the active python session in addition to GDrive

    Examples
    --------
    .. code-block:: python

        import rabpro
        from rabpro.subbasin_stats import Dataset
        import numpy as np
        import geopandas as gpd
        from shapely.geometry import box

        total_bounds = np.array([-85.91331249, 39.42609864, -85.88453019, 39.46429816])
        gdf = gpd.GeoDataFrame({"idx": [1], "geometry": [box(*total_bounds)]}, crs="EPSG:4326")

        # defaults
        data, task = rabpro.subbasin_stats.compute(
            [
                Dataset(
                    "JRC/GSW1_3/MonthlyRecurrence",
                    "monthly_recurrence",
                )
            ],
            basins_gdf = gdf,
            test = True,
        )

        # with time_stats specified
        data, task = rabpro.subbasin_stats.compute(
            [
                Dataset(
                    "JRC/GSW1_3/MonthlyRecurrence",
                    "monthly_recurrence",
                    time_stats = ["median"]
                )
            ],
            basins_gdf = gdf,
            test = True,
        )
    """

    # Dictionary for determining which rasters and statistics to compute
    control = _get_controls(dataset_list)
    ee.Initialize()

    # Create water occurence mask
    occ_mask = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select("occurrence").lt(90)

    # Convert GeoDataFrame to ee.Feature objects
    if basins_gdf is not None:
        features = []
        for i in range(basins_gdf.shape[0]):
            geom = basins_gdf.iloc[i : i + 1, :]
            jsonDict = json.loads(geom.to_json())
            geojsonDict = jsonDict["features"][0]
            features.append(ee.Feature(geojsonDict))
        featureCollection = ee.FeatureCollection(features)
    else:  # gee_feature_path is specified
        featureCollection = ee.FeatureCollection(gee_feature_path)

    # For each raster
    datas, tasks = [], []
    for d in control:
        if d.band is None or d.band == "None":
            if d.type == "image":
                imgcol = ee.ImageCollection(ee.Image(d.data_id))
            else:
                if d.start is not None and d.end is not None:
                    imgcol = ee.ImageCollection(d.data_id).filterDate(d.start, d.end)
                else:
                    imgcol = ee.ImageCollection(d.data_id)
        else:
            if d.type == "image":
                imgcol = ee.ImageCollection(ee.Image(d.data_id).select(d.band))
            else:
                if d.start is not None and d.end is not None:
                    imgcol = (
                        ee.ImageCollection(d.data_id)
                        .select(d.band)
                        .filterDate(d.start, d.end)
                    )
                else:
                    imgcol = ee.ImageCollection(d.data_id).select(d.band)

        if len(d.time_stats) > 0:
            time_reducer = _parse_reducers(base=getattr(ee.Reducer, d.time_stats[0])())
            imgcol = imgcol.reduce(time_reducer)
            imgcol = ee.ImageCollection(imgcol)

        # imgcol = imgcol.map(lambda img: img.clipToCollection(featureCollection))

        if verbose:
            print(f"Computing subbasin stats for {d.data_id}...")

        # Add threshold mask to image using GSW occurrence band
        if d.mask:
            imgcol = imgcol.map(lambda img: img.updateMask(occ_mask))

        # Generate reducer - mean and count always computed
        reducer = _parse_reducers(d.stats)

        def map_func(img):
            # TODO: change to reduceRegion or simplify geometries
            return img.reduceRegions(
                collection=featureCollection, reducer=reducer, scale=d.resolution
            )

        reducedFC = imgcol.map(map_func)
        table = reducedFC.flatten()

        def range_func(feat):
            # TODO: Change to aggregateArray - more efficient?
            return feat.set(
                "range", feat.getNumber("max").subtract(feat.getNumber("min"))
            )

        # Map across feature collection and use min and max to compute range
        if "range" in d.stats:
            table = table.map(range_func)

        # Apply reducer functions
        if reducer_funcs is not None:
            for f in reducer_funcs:

                def reducer_func(feat):
                    return feat.set(f.__name__, f(feat))

                table = table.map(reducer_func)

        # https://geohackweek.github.io/GoogleEarthEngine/04-reducers/
        def remove_geometry(feat):
            return feat.select([".*"], None, False)

        table = table.map(remove_geometry)

        if test:
            data = table.getInfo()

        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=dataset_to_filename(d.data_id, d.band, tag),
            folder=folder,
            fileFormat="csv",
        )

        task.start()

        if test:
            datas.append(data)
            tasks.append(task)
        else:
            datas.append(
                table.getDownloadURL(
                    filetype="csv", filename=dataset_to_filename(d.data_id, d.band, tag)
                )
            )
            tasks.append(task)

    return datas, tasks


def _parse_reducers(stats=None, base=None):
    """Generate reducer - mean and count always computed

    Examples:
    .. code-block:: python

        import ee
        ee.Initialize()
        _parse_reducers(["mean", "max"])
        _parse_reducers(["max"], ee.Reducer.mean())
    """

    if base is None:
        reducer = ee.Reducer.count().combine(
            reducer2=ee.Reducer.mean(), sharedInputs=True
        )
    else:
        reducer = base

    if stats is None:
        return reducer

    if ("min" in stats and "max" in stats) or "range" in stats:
        reducer = reducer.combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
    elif "min" in stats:
        reducer = reducer.combine(reducer2=ee.Reducer.min(), sharedInputs=True)
    elif "max" in stats:
        reducer = reducer.combine(reducer2=ee.Reducer.max(), sharedInputs=True)

    if "stdDev" in stats:
        reducer = reducer.combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
    if "sum" in stats:
        reducer = reducer.combine(reducer2=ee.Reducer.sum(), sharedInputs=True)
    if "freqhist" in stats:
        reducer = reducer.combine(
            reducer2=ee.Reducer.frequencyHistogram(), sharedInputs=True
        )

    pct_list = [int(pct[3:]) for pct in stats if pct[:3] == "pct"]
    if pct_list:
        reducer = reducer.combine(
            reducer2=ee.Reducer.percentile(pct_list), sharedInputs=True
        )

    return reducer


def _get_controls(datasets):
    """
    Prepare paths and parameters for computing subbasin raster stats. Takes in
    list of user specified datasets.
    """

    # Load raster metadata file
    datapaths = ru.get_datapaths(rebuild_vrts=False)
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

        if d.start is not None:
            if date.fromisoformat(d.start) < date.fromisoformat(
                gee_dataset["start_date"]
            ):
                print(
                    f"Warning: requested start date earlier than expected for {d.data_id}:{d.band}"
                )

        if d.end is not None:
            if date.fromisoformat(d.end) > date.fromisoformat(gee_dataset["end_date"]):
                print(
                    f"Warning: requested end date later than expected for {d.data_id}:{d.band}"
                )

        d.stats = set(d.stats + ["count", "mean"])

        if "no_data" in gee_dataset["bands"][d.band]:
            d.no_data = gee_dataset["bands"][d.band]["no_data"]

        resolution = None
        if "resolution" in gee_dataset["bands"][d.band]:
            resolution = gee_dataset["bands"][d.band]["resolution"]
        if d.resolution is None:
            d.resolution = resolution
        if d.resolution and resolution and d.resolution < resolution:
            print(
                "Warning: requested resolution is less than the native raster resolution"
            )

        d.type = gee_dataset["type"]

        control.append(d)

    return control
