"""
Utility functions (utils.py)
============================

"""

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import appdirs
import cv2
import geopandas as gpd
import numpy as np
import osgeo
import pandas as pd
import requests
import shapely
from osgeo import gdal, osr, ogr
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from skimage import measure

CATALOG_URL = "https://raw.githubusercontent.com/jonschwenk/rabpro/main/Data/gee_datasets.json"

_DATAPATHS = None

_PATH_CONSTANTS = {
    "HydroBasins1": f"HydroBasins{os.sep}level_one",
    "HydroBasins12": f"HydroBasins{os.sep}level_twelve",
    "DEM": f"DEM{os.sep}MERIT103{os.sep}merit_dem.vrt",
    "DEM_fdr": f"DEM{os.sep}MERIT_FDR{os.sep}MERIT_FDR.vrt",
    "DEM_uda": f"DEM{os.sep}MERIT_UDA{os.sep}MERIT_UDA.vrt",
    "DEM_elev_hp": f"DEM{os.sep}MERIT_ELEV_HP{os.sep}MERIT_ELEV_HP.vrt",
    "DEM_width": f"DEM{os.sep}MERIT_WTH{os.sep}MERIT_WTH.vrt",
}

_GEE_CACHE_DAYS = 1


def get_datapaths():
    """
    Returns a dictionary of paths to all data that RaBPro uses. Also builds
    virtual rasters for MERIT data.

    Returns
    -------
    dict
        contains paths to all data that RaBPro uses
    """

    global _DATAPATHS
    if _DATAPATHS is not None:
        _build_virtual_rasters(_DATAPATHS)
        return _DATAPATHS

    datapath = Path(appdirs.user_data_dir("rabpro", "jschwenk"))
    configpath = Path(appdirs.user_config_dir("rabpro", "jschwenk"))
    datapaths = {key: str(datapath / Path(val)) for key, val in _PATH_CONSTANTS.items()}
    gee_metadata_path = datapath / "gee_datasets.json"
    datapaths["gee_metadata"] = str(gee_metadata_path)

    # Download catalog JSON file
    if gee_metadata_path.is_file():
        mtime = datetime.fromtimestamp(gee_metadata_path.stat().st_mtime)
        delta = datetime.now() - mtime

    if not gee_metadata_path.is_file() or delta > timedelta(days=_GEE_CACHE_DAYS):
        try:
            response = requests.get(CATALOG_URL)
            if response.status_code == 200:
                r = response.json()
                with open(datapaths["gee_metadata"], "w") as f:
                    json.dump(r, f, indent=4)
            else:
                print(
                    f"{CATALOG_URL} returned error status code {response.status_code}. Download manually into {gee_metadata_path}"
                )
        except Exception as e:
            print(e)

    # User defined GEE datasets
    user_gee_metadata_path = configpath / "user_gee_datasets.json"
    datapaths["user_gee_metadata"] = str(user_gee_metadata_path)
    if not user_gee_metadata_path.is_file():
        datapaths["user_gee_metadata"] = None

    _build_virtual_rasters(datapaths)
    _DATAPATHS = datapaths
    return datapaths


def _build_virtual_rasters(datapaths):
    msg_dict = {
        "DEM": "Building virtual raster DEM from MERIT tiles...",
        "DEM_fdr": "Building flow direction virtual raster DEM from MERIT tiles...",
        "DEM_uda": "Building drainage areas virtual raster DEM from MERIT tiles...",
        "DEM_elev_hp": "Building hydrologically-processed elevations virtual raster DEM from MERIT tiles...",
        "DEM_width": "Building width virtual raster from MERIT tiles...",
    }

    # Ensure that DEM virtual rasters are built
    for key in msg_dict:
        if not os.path.isfile(datapaths[key]):
            print(msg_dict[key])
            build_vrt(
                os.path.dirname(os.path.realpath(datapaths[key])), outputfile=datapaths[key],
            )


def get_exportpaths(name, basepath=None, overwrite=False):
    """ Returns a dictionary of paths for exporting RaBPro results. Also creates
    "results" folders when necessary.

    Parameters
    ----------
    name : str
        Name of directory to create within "results" directory.
    basepath : str, optional
        path to put "results" directory. By default None. If None, creates in
        current working directory.
    overwrite : bool, optional
        overwrite "name" directory, by default False

    Returns
    -------
    dict
        contains paths to all output that RaBPro generates
    """

    if basepath is None:
        results = Path(os.getcwd()) / "results"
    else:
        results = Path(basepath)

    # Make a results directory if it doesn't exist
    if not results.exists():
        results.mkdir(parents=True, exist_ok=True)

    namedresults = results / name

    # Make a named results directory if it doesn't exist
    if not namedresults.exists():
        namedresults.mkdir(parents=True, exist_ok=True)
    elif overwrite:
        clear_directory(namedresults)

    # Results path dictionary
    exportpaths = {
        "base": str(results),
        "basenamed": str(namedresults),
        "subbasins": str(namedresults / "subbasins.json"),
        "subbasins_inc": str(namedresults / "subbasins_inc.json"),
        "centerline_results": str(namedresults / "centerline_results.json"),
        "dem_results": str(namedresults / "dem_flowpath.json"),
    }

    return exportpaths


def parse_keys(gdf):
    """ 
    Attempts to interpret the column names of the input dataframe.
    In particular, looks for widths and distances along centerline.

    Parameters
    ----------
    gdf : GeoDataFrame
        table to parse

    Returns
    -------
    dict
        contains column names and corresponding properties
    """
    keys = gdf.keys()
    parsed = {"distance": None, "width": None}
    for k in keys:
        if "distance" in k.lower():
            parsed["distance"] = k
        if "width" in k.lower():
            parsed["width"] = k

    return parsed


def build_vrt(
    tilespath,
    clipper=None,
    extents=None,
    outputfile=None,
    nodataval=None,
    res=None,
    sampling="nearest",
    ftype="tif",
    separate=False,
):
    """ Creates a text file for input to gdalbuildvrt, then builds vrt file with
    same name. If output path is not specified, vrt is given the name of the
    final folder in the path.

    Parameters
    ----------
    tilespath : str
        the path to the file (or folder of files) to be clipped-- if tilespath
        contains an extension (e.g. .tif, .vrt), then that file is used.
        Otherwise, a virtual raster will be built of all the files in the
        provided folder. if filespath contains an extension (e.g. .tif, .vrt),
        filenames of tiffs to be written to vrt. This list can be created by
        tifflist and should be in the same folder
    clipper : str, optional
        path to a georeferenced image, vrt, or shapefile that will be used to
        clip. By default None
    extents : list, optional
        the extents by which to crop the vrt. Extents should be a 4 element
        list: [left, right, top, bottom] in the same projection coordinates as
        the file(s) to be clipped. By default None
    outputfile : str, optional
        path (including filename w/ext) to output the vrt. If none is provided,
        the vrt will be saved in the 'filespath' path. By default None
    nodataval : int, optional
        value to be masked as nodata, by default None
    res : flt, optional
        resolution of the output vrt (applied to both x and y directions), by
        default None
    sampling : str, optional
        resampling scheme (nearest, bilinear, cubic, cubicspline, lanczos,
        average, mode), by default "nearest"
    ftype : str, optional
        "tif" if building from a list of tiffs, or "vrt" if building from a vrt,
        by default "tif"
    separate : bool, optional
        [description], by default False

    Returns
    -------
    str
        path of the built virtual raster

    Raises
    ------
    TypeError
        Unsupported file type passed in 'ftype'
    RuntimeError
        No files found to build raster or raster build fails
    """
    base, folder, file, ext = parse_path(tilespath)

    # Set output names
    if outputfile is None:
        if clipper:
            cliptxt = "_clip"
        else:
            cliptxt = ""
        vrtname = os.path.join(base, folder, folder + cliptxt + ".vrt")
        vrttxtname = os.path.join(base, folder, folder + cliptxt + ".txt")
    else:
        vrtname = os.path.normpath(outputfile)
        vrttxtname = vrtname.replace(".vrt", ".txt")

    # If a folder was given, make a list of all the text files
    if len(file) == 0:

        filelist = []

        if ftype == "tif":
            checktype = ("tif", "tiff")
        elif ftype == "hgt":
            checktype = "hgt"
        elif ftype == "vrt":
            checktype = "vrt"
        elif ftype == "nc":
            checktype = "nc"
        else:
            raise TypeError("Unsupported filetype provided - must be tif, hgt, nc, or vrt.")

        for f in os.listdir(tilespath):
            if f.lower().endswith(checktype):  # ensure we're looking at a tif
                filelist.append(os.path.join(tilespath, f))
    else:
        filelist = [tilespath]

    if len(filelist) < 1:
        print(f"Supplied path for building vrt: {filelist}")
        raise RuntimeError("The path you supplied appears empty.")

    # Clear out .txt and .vrt files if they already exist
    delete_file(vrttxtname)
    delete_file(vrtname)

    with open(vrttxtname, "w") as tempfilelist:
        for f in filelist:
            tempfilelist.writelines("%s\n" % f)

    # Get extents of clipping raster
    if clipper:
        extents = raster_extents(clipper)

    # Build the vrt with input options
    callstring = [
        "gdalbuildvrt",
        "-overwrite",
    ]

    if np.size(extents) == 4:
        stringadd = [
            "-te",
            str(extents[0]),
            str(extents[3]),
            str(extents[1]),
            str(extents[2]),
        ]
        for sa in stringadd:
            callstring.append(sa)

    if nodataval:
        stringadd = ["-srcnodata", str(nodataval)]
        for sa in stringadd:
            callstring.append(sa)

    if res:
        stringadd = ["-resolution", "user", "-tr", str(res), str(res)]
        for sa in stringadd:
            callstring.append(sa)

    if sampling != "nearest":
        stringadd = ["-r", sampling]
        for sa in stringadd:
            callstring.append(sa)

    if separate is True:
        callstring.append("-separate")

    stringadd = ["-input_file_list", vrttxtname, vrtname]
    for sa in stringadd:
        callstring.append(sa)

    # Make the call
    proc = subprocess.Popen(callstring, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    # Check that vrt built successfully
    if len(stderr) > 3:
        raise RuntimeError(f"Virtual raster did not build sucessfully. Error: {stderr}")
    else:
        print(stdout.decode())

    return vrtname


def raster_extents(raster_path):

    # Outputs extents as [xmin, xmax, ymin, ymax]

    # Check if file is shapefile, else treat as raster
    fext = raster_path.split(".")[-1]
    if fext == "shp" or fext == "SHP":
        driver = ogr.GetDriverByName("ESRI Shapefile")
        shapefile = driver.Open(raster_path, 0)  # open as read-only
        layer = shapefile.GetLayer()
        ext = np.array(layer.GetExtent())
        extents = [ext[0], ext[1], ext[3], ext[2]]
    else:
        # Get the clipping raster lat/longs
        rast = gdal.Open(raster_path)
        cgt = rast.GetGeoTransform()
        clip_ULx = cgt[0]
        clip_ULy = cgt[3]
        clip_LRx = cgt[0] + cgt[1] * rast.RasterXSize
        clip_LRy = cgt[3] + cgt[5] * rast.RasterYSize
        extents = [clip_ULx, clip_LRx, clip_ULy, clip_LRy]

    return extents


def parse_path(path):
    """
    Parses a file or folderpath into: base, folder (where folder is the
    outermost subdirectory), filename, and extention. Filename and extension are
    empty if a directory is passed.
    """

    if path[0] != os.sep and platform.system() != "Windows":  # This is for non-windows...
        path = os.sep + path

    # Pull out extension and filename, if exist
    if "." in path:
        extension = "." + path.split(".")[-1]
        temp = path.replace(extension, "")
        filename = temp.split(os.sep)[-1]
        drive, temp = os.path.splitdrive(temp)
        path = os.path.join(*temp.split(os.sep)[:-1])
        path = drive + os.sep + path
    else:
        extension = ""
        filename = ""

    # Pull out most exterior folder
    folder = path.split(os.sep)[-1]

    # Pull out base
    drive, temp = os.path.splitdrive(path)
    base = os.path.join(*temp.split(os.sep)[:-1])
    base = drive + os.sep + base

    return base, folder, filename, extension


def delete_file(file):
    # Deletes a file. Input is file's location on disk (path + filename)
    try:
        os.remove(file)
    except OSError:
        pass


def clear_directory(Path_obj):
    """
    Given a pathlib Path obj, clears all the contents of the directory. Does
    not remove the directory itself.
    """
    for child in Path_obj.glob("*"):
        if not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)


def lonlat_to_xy(lons, lats, gt):

    lats = np.array(lats)
    lons = np.array(lons)

    xs = ((lons - gt[0]) / gt[1]).astype(int)
    ys = ((lats - gt[3]) / gt[5]).astype(int)
    # xs = round((lons - gt[0]) / gt[1])
    # ys = round((lats - gt[3]) / gt[5])

    return np.column_stack((xs, ys))


def xy_to_coords(xs, ys, gt):
    """
    Transforms a set of x and y coordinates to their corresponding coordinates
    within a geotiff image.

    Parameters
    ----------
    xs : numpy.ndarray
        x coordinates to transform
    ys : numpy.ndarray
        y coordinates to transform
    gt : tuple
        6-element tuple gdal GeoTransform. (uL_x, x_res, rotation, ul_y, rotation, y_res).
        Automatically created by gdal's GetGeoTransform() method.

    Returns
    ----------
    cx, cy : tuple of ints
        Column and row indices of the provided coordinates.
    """

    cx = gt[0] + (xs + 0.5) * gt[1]
    cy = gt[3] + (ys + 0.5) * gt[5]

    return cx, cy


def lonlat_plus_distance(lon, lat, dist, bearing=0):
    """
    Returns the lon, lat coordinates of a point that is dist away (dist in km)
    and with a given bearing (in degrees, 0 is North). Uses a Haversine
    approximation.
    """

    bearing = np.radians(bearing)
    R = 6378.1  # Radius of the Earth

    latr = np.radians(lat)  # Current lat point converted to radians
    lonr = np.radians(lon)  # Current long point converted to radians

    lat_m = np.arcsin(
        np.sin(latr) * np.cos(dist / R) + np.cos(latr) * np.sin(dist / R) * np.cos(bearing)
    )

    lon_m = lonr + np.arctan2(
        np.sin(bearing) * np.sin(dist / R) * np.cos(latr),
        np.cos(dist / R) - np.sin(latr) * np.sin(latr),
    )

    lat_m = np.degrees(lat_m)
    lon_m = np.degrees(lon_m)

    return lon_m, lat_m


def union_gdf_polygons(gdf, idcs, buffer=True):
    """
    Given an input geodataframe and a list of indices, return a shapely geometry
    that unions the geometries found at idcs into a single shapely geometry
    object.

    This function also buffers each polygon slightly, then un-buffers the
    unioned polygon by the same amount. This is to avoid errors associated with
    floating-point round-off; see here:
    https://gis.stackexchange.com/questions/277334/shapely-polygon-union-results-in-strange-artifacts-of-tiny-non-overlapping-area

    Parameters
    ----------
    gdf : GeoDataFrame
        Geometries to combine
    idcs : list
        Indicates which geometris in 'gdf' to combine
    buffer : bool, optional
        buffer polygons, by default True

    Returns
    -------
    shapely.geometry object
        Union of passed geometries
    """

    if buffer:
        from shapely.geometry import JOIN_STYLE

        # Buffer distance (tiny)
        eps = 0.0001

    polys = []
    for i in idcs:
        if buffer:
            polys.append(gdf.iloc[i].geometry.buffer(eps, 1, join_style=JOIN_STYLE.mitre))
        #            polys.append(gdf.iloc[i].geometry.buffer(eps))
        else:
            polys.append(gdf.iloc[i].geometry)

    polyout = shapely.ops.unary_union(polys)

    if buffer:
        #        polyout = polyout.buffer(-eps)
        polyout = polyout.buffer(-eps, 1, join_style=JOIN_STYLE.mitre)

    return polyout


def haversine(lats, lons):
    """
    Computes distances between latitude and longitude pairs of points.

    Parameters
    ----------
    lats : numpy array (or interpretable by numpy)
        latitude values
    lons : numpy array (or interpretable by numpy)
        longitude values

    Returns
    -------
    numpy.ndarray
        Distances between each point defined by lats, lons.
    """

    R = 6372.8 * 1000

    dLat = np.radians(np.diff(lats))
    dLon = np.radians(np.diff(lons))

    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])

    a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def validify_polygons(polys):
    """
    Hacky ways to validify a polygon. If can't be validified, returns the
    original.

    Parameters
    ----------
    geom : list
        List of shapely.geometry.Polygon

    Returns
    -------
    geomsv : list
        List of shapely.geometry.Polygon that have been attempted to validify.

    """
    geomsv = []
    for geom in polys:

        if type(geom) is Polygon:
            if geom.buffer(0).is_valid is True:
                geomsv.append(geom.buffer(0))
            else:
                geomsv.append(geom)

        if type(geom) is MultiPolygon:
            geomu = unary_union(geom)
            geomu = geomu.buffer(0)
            if geomu.is_valid is True:
                geomsv.append(geomu)
            else:
                geomsv.append(geom)

    return geomsv
