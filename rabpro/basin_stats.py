"""
Basin statistics (basin_stats.py)
=======================================

Computes basin statistics using Google Earth Engine.
"""

import json
import re
import warnings
from collections import OrderedDict
from datetime import date

import ee
import numpy as np
import pandas as pd
import requests

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
        Desired start date of data in ISO format: YYYY-MM-DD. Defaults to None which
        GEE interprets as the dataset start.
    end : str, optional
        Desired end date of data in ISO format: YYYY-MM-DD. Defaults to None which GEE
        interprets as the dataset end.
    stats : list, optional
        List of desired stats to compute: min, max, range, mean, count, std,
        sum, and percentiles in the following format: pct1, pct90, pct100, etc.
        Computes ["count", "mean"] by default.
    time_stats : list, optional
        List of desired stats to compute through time.
    mask : bool, optional
        Whether or not to mask out water in the dataset using the Global Surface
        Water occurrence band. By default False.
    mosaic : bool, optional
        If True, the imageCollection in data_id will be mosaiced first. Useful
        when the imageCollection is a set of tiled images. Do not use for
        time-varying imageCollections.
    prepend_label : str, optional
        Text to prepend to the exported statistics file.
    gee_type : str, optional
        Either 'image' or 'imagecollection'.

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
        mosaic=False,
        prepend_label="",
        gee_type=None,
    ):
        self.data_id = data_id
        self.band = band
        self.resolution = resolution
        self.start = start
        self.end = end
        self.stats = stats if stats is not None else []
        self.time_stats = time_stats if time_stats is not None else []
        self.mask = mask
        self.mosaic = mosaic
        self.prepend = prepend_label
        self.type = gee_type

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def dataset_to_filename(prepend, data_id, band=None):
    """
    Examples
    --------
    .. code-block:: python

        import rabpro
        rabpro.basin_stats.dataset_to_filename("ECMWF/ERA5_LAND/MONTHLY", "temperature_2m")

    """
    if prepend == "" or prepend is None:
        if band is None or band == "None":
            res = f"{data_id}"
        else:
            res = f"{data_id}__{band}"
    else:
        if band is None or band == "None":
            res = prepend + "__" + f"{data_id}"
        else:
            res = prepend + "__" + f"{data_id}__{band}"

    res = res.replace("/", "-")
    return res


def _str_to_dict(a_string):
    new_d = re.findall(r"([0-9]*\.?[0-9]+)", a_string)
    res = {new_d[i]: float(new_d[i + 1]) for i in range(0, len(new_d), 2)}
    res = OrderedDict(sorted(res.items()))
    return res


def _format_cols(df, prepend, col_drop_list, col_drop_defaults, col_protect_list):

    col_drop_list = col_drop_list + col_drop_defaults
    df = ru.drop_column_if_exists(df, col_drop_list)

    col_names = df.columns.tolist()
    to_tag = list(
        np.where([x not in col_protect_list for x in [x for x in col_names]])[0]
    )
    if prepend != "":
        for i in to_tag:
            col_names[i] = prepend + "_" + col_names[i]
    df.columns = col_names

    return df


def _read_url(url):
    df = pd.read_csv(url)

    if "histogram" in df.columns:
        histogram = df.histogram
        df = df.drop(columns=["histogram", "mean"])
        histogram = [_str_to_dict(x) for x in histogram]
        histogram = pd.DataFrame(histogram)
        df = pd.concat([df, histogram], axis=1)

    return df


def format_gee(
    df_list,
    prepend_list,
    col_drop_list=[],
    col_protect_list=["id_basin", "id_outlet", "idx", "id", "vote_id"],
    col_drop_defaults=["DA", "count", ".geo", "da_km2"],
):
    """Parameters
    ----------
    df_list : list
        list of DataFrames
    prepend_list : list
        tags to pre-append to corresponding columns, of length url_list
    col_drop_list : list, optional
        custom columns to drop, by default []
    col_protect_list : list, optional
        columns to avoid tagging, by default ["id_basin", "id_outlet", "idx", "id",
        "vote_id"]
    col_drop_defaults : list, optional
        built-in columns to drop, by default ["DA", "count", ".geo", "system:index",
        "da_km2"]

    Returns
    -------
    DataFrame
    """
    res = [
        _format_cols(
            df,
            prepend,
            col_drop_list=col_drop_list,
            col_drop_defaults=col_drop_defaults,
            col_protect_list=col_protect_list,
        )
        for df, prepend in zip(df_list, prepend_list)
    ]

    res = pd.concat(res, axis=1)

    # drop duplicate columns and move to front
    where_duplicated = [x for x in np.where(res.columns.duplicated())[0]]
    if len(where_duplicated) > 0:
        first_column_names = res.columns[where_duplicated][0]
        first_column = res.pop(res.columns[where_duplicated][0]).iloc[:, 0]
        res.insert(0, first_column_names, first_column)

    return res


def fetch_gee(
    url_list,
    prepend_list,
    col_drop_list=[],
    col_protect_list=["id_basin", "id_outlet", "idx", "id", "vote_id"],
    col_drop_defaults=["DA", "count", ".geo", "da_km2"],
):
    """Download and format data from GEE urls

    Parameters
    ----------
    url_list : list
        list of urls returned from compute
    prepend_list : list
        tags to pre-append to corresponding columns, of length url_list
    col_drop_list : list, optional
        custom columns to drop, by default []
    col_protect_list : list, optional
        columns to avoid tagging, by default ["id_basin", "id_outlet", "idx", "id",
        "vote_id"]
    col_drop_defaults : list, optional
        built-in columns to drop, by default ["DA", "count", ".geo", "system:index",
        "da_km2"]

    Returns
    -------
    DataFrame

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import geopandas as gpd
        from shapely.geometry import box

        import rabpro
        from rabpro.basin_stats import Dataset

        total_bounds = np.array([-85.91331249, 39.42609864, -85.88453019, 39.46429816])
        basin = gpd.GeoDataFrame({"idx": [0], "geometry": [box(*total_bounds)]}, crs="EPSG:4326")

        dataset_list = [
            Dataset("ECMWF/ERA5_LAND/MONTHLY", "temperature_2m", stats=["mean"], time_stats = ["median"]),
            Dataset("NASA/GPM_L3/IMERG_MONTHLY_V06", "precipitation", stats=["mean"], time_stats = ["median"]),
        ]

        urls, tasks = rabpro.basin_stats.compute(
            dataset_list, basins_gdf=basin, folder="rabpro"
        )

        tag_list = ["temperature", "precip"]
        data = rabpro.basin_stats.fetch_gee(urls, tag_list, col_drop_list = ["system:index"])

    """

    df_list = [_read_url(url) for url in url_list]

    res = format_gee(
        df_list, prepend_list, col_drop_list, col_protect_list, col_drop_defaults
    )

    return res


def _gdf_to_features(gdf):
    features = []
    for i in range(gdf.shape[0]):
        geom = gdf.iloc[i : i + 1, :]
        jsonDict = json.loads(geom.to_json())
        geojsonDict = jsonDict["features"][0]
        features.append(ee.Feature(geojsonDict))

    return features


def compute(
    dataset_list,
    gee_feature_path=None,
    gee_featureCollection=None,
    basins_gdf=None,
    reducer_funcs=None,
    folder=None,
    filename=None,
    verbose=False,
    test=False,
    validate_dataset_list=True,
):
    """
    Compute subbasin statistics for each dataset and band specified. One of
    gee_feature_path, gee_featureCollection, or basins_gdf must be specified.

    Parameters
    ----------
    dataset_list : list of Datasets
        List of rabpro Dataset objects to compute statistics over.
    basins_gdf : GeoDataFrame
        Table of subbasin geometries.
    gee_feature_path : string
        Path to a GEE feature collection
    gee_featureCollection : ee.FeatureCollection object
        FeatureCollection object.
    reducer_funcs : list of functions, optional
        List of functions to apply to each feature over each dataset. Each
        function should take in an ee.Feature() object. For example, this is how
        the function and header are applied on a feature:
        feature.set(f.__name__, function(feature))
    verbose : bool, optional
        By default False.
    folder : str, optional
        Google Drive folder to store results in, by default top-level root.
    filename : str, optional
        Name of the GEE export file.
    test : bool, optional
        Return results to the active python session in addition to GDrive
    validate_dataset_list: bool, optional
        Validate the dataset_list against a scraped version of the GEE catalog?

    Examples
    --------
    .. code-block:: python

        import rabpro
        import numpy as np
        import geopandas as gpd
        from shapely.geometry import box
        from rabpro.basin_stats import Dataset

        total_bounds = np.array([-85.91331249, 39.42609864, -85.88453019, 39.46429816])
        gdf = gpd.GeoDataFrame({"idx": [1], "geometry": [box(*total_bounds)]}, crs="EPSG:4326")

        # defaults
        data, task = rabpro.basin_stats.compute(
            [
                Dataset(
                    "JRC/GSW1_4/MonthlyRecurrence",
                    "monthly_recurrence",
                )
            ],
            basins_gdf = gdf,
            test = True,
        )

        # with time_stats specified
        data, task = rabpro.basin_stats.compute(
            [
                Dataset(
                    "JRC/GSW1_4/MonthlyRecurrence",
                    "monthly_recurrence",
                    time_stats = ["median"]
                )
            ],
            basins_gdf = gdf,
            test = True,
        )
    """

    # Prepare the featureCollection
    if basins_gdf is not None:
        features = _gdf_to_features(basins_gdf)
        featureCollection = ee.FeatureCollection(features)
    elif gee_featureCollection is not None:
        featureCollection = gee_featureCollection
    elif gee_feature_path is not None:  # gee_feature_path is specified
        featureCollection = ee.FeatureCollection(gee_feature_path)
    else:
        raise KeyError(
            "A featurecollection must be provided by specifying one of gee_feature_path, gee_featureCollection, or basins_gdf."
        )

    # Dictionary for determining which rasters and statistics to compute
    if validate_dataset_list:
        control = _get_controls(dataset_list)
    else:  # override validation, probably need to manually set gee_type and resolution in Dataset call
        control = dataset_list

    ee.Initialize()

    # Create water occurence mask
    occ_mask = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").lt(90)

    # For each raster
    datas, tasks = [], []
    for d in control:
        if d.band in ["None", None]:
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
                # Arbitrarily subject image assets to a time reducer which
                # should have no effect other than avoiding the error # 147
                # d.time_stats = ["median"]
            else:
                if d.start is not None and d.end is not None:
                    imgcol = (
                        ee.ImageCollection(d.data_id)
                        .select(d.band)
                        .filterDate(d.start, d.end)
                    )
                else:
                    imgcol = ee.ImageCollection(d.data_id).select(d.band)

        if (
            d.mosaic == True
        ):  # Don't use 'is' because numpy booleans aren't the same object type, == bypasses this
            imgcol = ee.ImageCollection(imgcol.mosaic())

        if len(d.time_stats) > 0:
            time_reducer = _parse_reducers(base=getattr(ee.Reducer, d.time_stats[0])())
            imgcol = imgcol.reduce(time_reducer)
            imgcol = ee.ImageCollection(imgcol)

        # imgcol = imgcol.map(lambda img: img.clipToCollection(featureCollection))

        if verbose:
            print(f"Submitting basin stats task to GEE for {d.data_id}...")

        # Add threshold mask to image using GSW occurrence band
        if d.mask:
            imgcol = imgcol.map(lambda img: img.updateMask(occ_mask))

        # Generate reducer - mean and count always computed
        reducer = _parse_reducers(d.stats)

        def map_func(img):
            # The .limit() here is due to a GEE bug, see:
            # https://gis.stackexchange.com/questions/407965/null-value-after-reduceregions-in-gee?rq=1
            return img.reduceRegions(
                collection=featureCollection.limit(1000000000),
                reducer=reducer,
                scale=d.resolution,
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

        if filename is None:
            filename = dataset_to_filename(d.prepend, d.data_id, d.band)

        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=filename,
            folder=folder,
            fileFormat="csv",
        )

        task.start()

        if test:
            datas.append(table.getInfo())
            tasks.append(task)
        else:
            datas.append(
                table.getDownloadURL(
                    filetype="csv",
                    filename=filename,
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

    if "stdDev" in stats or "std" in stats:
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


def _validate_dataset(d, datadict):
    if d.data_id not in datadict:
        raise Exception(
            "Unable to validate Dataset list. Does it exist in the GEE catalog?"
        )

    gee_dataset = datadict[d.data_id]

    if d.band not in gee_dataset["bands"]:
        warnings.warn(
            f"Warning: invalid data band provided: {d.data_id}:{d.band}",
            UserWarning,
        )

    if d.start is not None:
        if date.fromisoformat(d.start) < date.fromisoformat(gee_dataset["start_date"]):
            warnings.warn(
                "Warning: requested start date earlier than expected for"
                f" {d.data_id}:{d.band}",
                UserWarning,
            )

    if d.end is not None:
        if date.fromisoformat(d.end) > date.fromisoformat(gee_dataset["end_date"]):
            warnings.warn(
                "Warning: requested end date later than expected for"
                f" {d.data_id}:{d.band}",
                UserWarning,
            )

    d.stats = set(d.stats) | set(["count", "mean"])

    if "no_data" in gee_dataset["bands"][d.band]:
        d.no_data = gee_dataset["bands"][d.band]["no_data"]

    resolution = None
    if "resolution" in gee_dataset["bands"][d.band]:
        resolution = gee_dataset["bands"][d.band]["resolution"]
    if d.resolution is None:
        d.resolution = resolution
    if d.resolution and resolution and d.resolution < resolution:
        warnings.warn(
            "Warning: requested resolution is less than the native raster"
            " resolution",
            UserWarning,
        )

    d.type = gee_dataset["type"]
    return d


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
        d = _validate_dataset(d, datadict)
        control.append(d)

    return control


def image(
    dataset_list,
    gee_feature_path=None,
    basins_gdf=None,
    categorical=[False],
    verbose=False,
):
    """Download a GEE raster as a GeoTiff clipped to a basins GeoDataFrame

    Parameters
    ----------
    dataset_list : list of Datasets
        List of Dataset objects to compute statistics over.
    gee_feature_path : str, optional
        Path to a GEE feature collection, by default None
    basins_gdf : GeoDataFrame, optional
        Table of subbasin geometries, by default None
    categorical : list, optional
        By default [False]
    verbose : bool, optional
        By default False

    Returns
    -------
    list
        of GeoTiff download urls

    Examples
    --------
    .. code-block:: python

        import rabpro
        from rabpro.basin_stats import Dataset

        import numpy as np
        from shapely.geometry import box

        total_bounds = np.array([-85.91331249, 39.2, -85.5, 39.46429816])
        gdf = gpd.GeoDataFrame({"idx": [1], "geometry": [box(*total_bounds)]}, crs="EPSG:4326")

        dataset_list = [
            Dataset("ECMWF/ERA5_LAND/MONTHLY", "temperature_2m", time_stats=["median"])
        ]
        urls, tasks = rabpro.basin_stats.image(dataset_list, basins_gdf=gdf)
        basin_stats._fetch_raster(urls[0])
    """

    # Dictionary for determining which rasters to pull
    control = _get_controls(dataset_list)
    ee.Initialize()

    # Convert GeoDataFrame to ee.Feature objects
    if basins_gdf is not None:
        features = _gdf_to_features(basins_gdf)
        featureCollection = ee.FeatureCollection(features)
    else:  # gee_feature_path is specified
        featureCollection = ee.FeatureCollection(gee_feature_path)

    # ensure categorical is the proper length
    categorical_lengthed = []
    for i in range(0, len(control)):
        if i < len(categorical):
            categorical_lengthed.append(categorical[i])
        else:
            categorical_lengthed.append(False)

    # For each raster
    urls, tasks = [], []
    for d, is_categorical in zip(control, categorical_lengthed):
        # d.band, d.data_id, d.start, d.end
        if not is_categorical:
            if d.type == "image":
                img = ee.Image(d.data_id).select(d.band)
            elif d.mosaic == True:
                if d.band is None:
                    img = ee.ImageCollection(d.data_id).mosaic()
                else:
                    img = ee.ImageCollection(d.data_id).select(d.band).mosaic()
            else:
                if d.band is None:
                    img = ee.ImageCollection(d.data_id)
                else:
                    img = ee.ImageCollection(d.data_id).select(d.band)
                if d.start is not None and d.end is not None:
                    img = img.filterDate(d.start, d.end)
                img = img.reduce(ee.Reducer.mean())
        else:
            img = (
                ee.ImageCollection(d.data_id)
                .select(d.band)
                .limit(1, "system:time_start", False)
                .first()
            )

        if verbose:
            print(f"Submitting image retrieval task to GEE for {d.data_id}...")

        task = ee.batch.Export.image.toDrive(
            image=img,
            scale=30,
            region=featureCollection.geometry(),
            description=dataset_to_filename(d.prepend, d.data_id, d.band),
            crs="EPSG:4326",
        )

        url = img.getDownloadURL(
            {
                "format": "GEO_TIFF",
                "filename": dataset_to_filename(d.prepend, d.data_id, d.band),
                "region": featureCollection.geometry(),
                "scale": 30,
            }
        )
        task.start()

        urls.append(url)
        tasks.append(task)

    return urls, tasks


def _fetch_raster(url, fname="temp.tif", cleanup=True):
    response = requests.get(url)
    with open(fname, "wb") as fd:
        fd.write(response.content)

    return fname
