"""
Utility functions (utils.py)
============================

"""

import copy
import http.client as httplib
import itertools
import os
import platform
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from osgeo import gdal, ogr
from pyproj import Geod
from skimage import measure
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

from rabpro import data_utils as du

_DATAPATHS = None


def has_internet():
    # https://stackoverflow.com/a/29854274/3362993
    conn = httplib.HTTPConnection("www.google.com", timeout=5)
    try:
        conn.request("HEAD", "/")
        return True
    except Exception:
        return False
    finally:
        conn.close()


def envvars_rabpro():
    var_list = ["RABPRO_DATA", "RABPRO_CONFIG"]
    res_dict = {}
    for var in var_list:
        try:
            res_dict[var] = os.environ[var]
        except:
            pass
    return res_dict


def get_datapaths(
    root_path=None, config_path=None, force=False, update_gee_metadata=False
):
    """
    Returns a dictionary of paths to all data that rabpro uses. Also builds
    virtual rasters for MERIT data and downloads latest version of GEE catalog.

    Parameters
    ----------
    root_path: string, optional
        Path to rabpro Data folder that contains the HydroBASINS, MERIT-Hydro,
        and/or gee catalog jsons. Will read from an environment variable
        "RABPRO_DATA". If this variable is not set, uses appdirs to create a
        local data directory. This path is the parent directory for the MERIT
        and HydroBasins data directories.
    config_path: string, optional
        Path to rabpro config folder. Will read from an environment variable
        "RABPRO_CONFIG". If not set, uses appdirs to create local directory.
    force: boolean, optional
        Set True to override datapath caching. Otherwise only fetched once per py
        session.
    update_gee_metadata: boolean, optional
        If True, will attempt to download the latest GEE dataset metadata.

    Returns
    -------
    dict
        contains paths to all data that rabpro uses

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        utils.get_datapaths()
    """

    # This chunk makes sure that folder creation, data downloads, etc. only
    # happen once per py session
    global _DATAPATHS
    if _DATAPATHS is None or force is True:
        # Ensure data directories are established
        du.create_file_structure(datapath=root_path, configpath=config_path)
        datapaths = du.create_datapaths(datapath=root_path, configpath=config_path)
        _DATAPATHS = datapaths

    if has_internet() and update_gee_metadata:
        du.download_gee_metadata()

    return _DATAPATHS


def build_virtual_rasters(datapaths, skip_if_exists=False, verbose=True, **kwargs):
    """
    Builds virtual rasters on the four MERIT-Hydro tilesets.

    Parameters
    ----------
    datapaths: dict
        Contains the paths to the data. Generate with get_datapaths().
    skip_if_exists: bool, optional
        If True, will not rebuild the virtual raster if one already exists.
        The default is False.
    verbose: bool, optional
        If True, will provide updates as virtual rasters are built.
    **kwargs
        Arguments passed to the build_vrt function.

    Warns
    -----
    RuntimeWarning
        Missing data

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        d_paths = utils.get_datapaths()
        utils.build_virtual_rasters(d_paths, extents=[-180.00041666666667, 179.99958333333913, 84.99958333333333, -60.000416666669], verbose=False)
    """

    msg_dict = {
        "DEM_fdr": "Building flow direction virtual raster DEM from MERIT tiles...",
        "DEM_uda": "Building drainage areas virtual raster DEM from MERIT tiles...",
        "DEM_elev_hp": "Building hydrologically-processed elevations virtual raster DEM from MERIT tiles...",
        "DEM_width": "Building width virtual raster from MERIT tiles...",
    }

    missing_dict = {
        "DEM_fdr": "flow directions (FDR)",
        "DEM_uda": "drainage area (UDA)",
        "DEM_elev_hp": "hydrologically adjusted DEM (ELEV_HP)",
        "DEM_width": "width (WTH)",
    }

    missing_merit = []
    for key in msg_dict:
        # Check that MERIT data are available before trying to build VRT
        geotiffs = os.listdir(os.path.dirname(datapaths[key]))
        if len(geotiffs) == 0:
            if verbose:
                print(f"No MERIT data found for {missing_dict[key]}.")
            missing_merit.append(key)
            continue
        else:
            if verbose:
                print(msg_dict[key])
            build_vrt(
                os.path.dirname(os.path.realpath(datapaths[key])),
                outputfile=datapaths[key],
                quiet=~verbose,
                **kwargs,
            )

    if len(missing_merit) > 0:
        warnings.warn(
            "Virtual rasters could not be built for the following MERIT-Hydro tiles"
            f" because no data were available: {missing_merit}. Use rabro.data_utils."
            "download_merit_hydro() to fetch a MERIT tile.",
            RuntimeWarning,
        )

    return


def get_exportpaths(name, basepath=None, overwrite=False):
    """Returns a dictionary of paths for exporting rabpro results. Also creates
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
        contains paths to all output that rabpro generates
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
        _clear_directory(namedresults)

    # Results path dictionary
    exportpaths = {
        "base": str(results),
        "basenamed": str(namedresults),
        "watershed": str(namedresults / "watershed.json"),
        "flowline": str(namedresults / "flowline.json"),
    }

    return exportpaths


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
    quiet=False,
):
    """Creates a text file for input to gdalbuildvrt, then builds vrt file with
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
        See separate argument to gdalbuildvrt, by default False
    quiet : bool, optional
        Set True to print progress, by default False

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
    base, folder, file, ext = _parse_path(tilespath)

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
            raise TypeError(
                "Unsupported filetype provided - must be tif, hgt, nc, or vrt."
            )

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
            tempfilelist.writelines(f"{f}\n")

    # Get extents of clipping raster
    if clipper:
        extents = raster_extents(clipper)

    # Build the vrt with input options
    callstring = ["gdalbuildvrt", "-overwrite"]

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
        print(f"Virtual raster may not have built sucessfully. Error: {stderr}")
    else:
        if not quiet:
            print(stdout.decode())

    return vrtname


def raster_extents(raster_path):
    """Output raster extents as [xmin, xmax, ymin, ymax]

    Parameters
    ----------
    raster_path : str
        Path to file

    Returns
    -------
    list
        [xmin, xmax, ymin, ymax]

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        utils.raster_extents(utils.get_datapaths(rebuild_vrts=False)["DEM_fdr"])
    """

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


def _parse_path(path):
    """
    Parses a file or folderpath into: base, folder (where folder is the
    outermost subdirectory), filename, and extention. Filename and extension are
    empty if a directory is passed.
    """

    # This is for non-windows...
    if path[0] != os.sep and platform.system() != "Windows":
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


def _clear_directory(Path_obj):
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
        6-element tuple gdal GeoTransform. (uL_x, x_res, rotation, ul_y, rotation,
        y_res).
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
        np.sin(latr) * np.cos(dist / R)
        + np.cos(latr) * np.sin(dist / R) * np.cos(bearing)
    )

    lon_m = lonr + np.arctan2(
        np.sin(bearing) * np.sin(dist / R) * np.cos(latr),
        np.cos(dist / R) - np.sin(latr) * np.sin(latr),
    )

    lat_m = np.degrees(lat_m)
    lon_m = np.degrees(lon_m)

    return lon_m, lat_m


def _regionprops(I, props, connectivity=2):
    """
    Finds blobs within a binary image and returns requested properties of
    each blob.
    This function was modeled after matlab's regionprops and is essentially
    a wrapper for skimage's regionprops. Not all of skimage's available blob
    properties are available here, but they can easily be added.
    Taken from RivGraph.im_utils

    Note that 'perimeter' will only return the outer-most perimeter in the
    case of a region that contains holes.

    Parameters
    ----------
    I : np.array
        Binary image containing blobs.
    props : list
        Properties to compute for each blob. Can include 'area', 'coords',
        'perimeter', 'centroid', 'mean', 'perim_len', 'convex_area',
        'eccentricity', 'major_axis_length', 'minor_axis_length',
        'label'.
    connectivity : int, optional
        If 1, 4-connectivity will be used to determine connected blobs. If
        2, 8-connectivity will be used. The default is 2.
    Returns
    -------
    out : dict
        Keys of the dictionary correspond to the requested properties. Values
        for each key are lists of that property, in order such that, e.g., the
        first entry of each property's list corresponds to the same blob.
    Ilabeled : np.array
        Image where each pixel's value corresponds to its blob label. Labels
        can be returned by specifying 'label' as a property.
    """
    # Check that appropriate props are requested
    available_props = [
        "area",
        "coords",
        "perimeter",
        "centroid",
        "mean",
        "perim_len",
        "convex_area",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "equivalent_diameter",
        "label",
    ]
    props_do = [p for p in props if p in available_props]
    cant_do = set(props) - set(props_do)
    if len(cant_do) > 0:
        warnings.warn(
            f"Cannot compute the following properties: {cant_do}", RuntimeWarning
        )

    Ilabeled = measure.label(I, background=0, connectivity=connectivity)
    properties = measure.regionprops(Ilabeled, intensity_image=I)

    out = {}
    # Get the coordinates of each blob in case we need them later
    if "coords" in props_do or "perimeter" in props_do:
        coords = [p.coords for p in properties]

    for prop in props_do:
        if prop == "area":
            out[prop] = np.array([p.area for p in properties])
        elif prop == "coords":
            out[prop] = list(coords)
        elif prop == "centroid":
            out[prop] = np.array([p.centroid for p in properties])
        elif prop == "mean":
            out[prop] = np.array([p.mean_intensity for p in properties])
        elif prop == "perim_len":
            out[prop] = np.array([p.perimeter for p in properties])
        elif prop == "perimeter":
            perim = []
            for blob in coords:
                # Crop to blob to reduce cv2 computation time and save memory
                Ip, cropped = crop_binary_coords(blob)

                # Pad cropped image to avoid edge effects
                Ip = np.pad(Ip, 1, mode="constant")

                # Get the perimeter using contours
                contours_init = measure.find_contours(
                    Ip, fully_connected="high", level=0.99
                )

                # In the cases of holes within the blob, multiple contours
                # will be returned. We take the longest.
                ci_lens = [len(ci) for ci in contours_init]
                contours_init = contours_init[ci_lens.index(max(ci_lens))]

                # Round the contour to get the pixel coordinates
                contours_init = [[round(c[0]), round(c[1])] for c in contours_init]

                # The skimage contour method returns duplicate pixel coordinates
                # at corners which must be removed
                contours = []
                for i in range(len(contours_init) - 1):
                    if contours_init[i] == contours_init[i + 1]:
                        continue
                    else:
                        contours.append(contours_init[i])

                # Adjust the coordinates for padding
                crows, ccols = [], []
                for c in contours:
                    crows.append(c[0] + cropped[1] - 1)
                    ccols.append(c[1] + cropped[0] - 1)
                cont_np = np.transpose(np.array((crows, ccols)))  # format the output
                perim.append(cont_np)
            out[prop] = perim
        elif prop == "convex_area":
            out[prop] = np.array([p.convex_area for p in properties])
        elif prop == "eccentricity":
            out[prop] = np.array([p.eccentricity for p in properties])
        elif prop == "equivalent_diameter":
            out[prop] = np.array([p.equivalent_diameter for p in properties])
        elif prop == "major_axis_length":
            out[prop] = np.array([p.major_axis_length for p in properties])
        elif prop == "minor_axis_length":
            out[prop] = np.array([p.minor_axis_length for p in properties])
        elif prop == "label":
            out[prop] = np.array([p.label for p in properties])
        else:
            warnings.warn(f"{prop} is not a valid property.", UserWarning)

    return out, Ilabeled


def crop_binary_coords(coords):
    """
    Crops an array of (row, col) coordinates (e.g. blob indices) to the smallest
    possible array.
    Taken from RivGraph.im_utils

    Parameters
    ----------
    coords: np.array
        N x 2 array. First column are rows, second are columns of pixel coordinates.

    Returns
    -------
    I: np.array
        Image of the cropped coordinates, plus padding if desired.
    clipped: list
        Number of pixels in [left, top, right, bottom] direction that were
        clipped.  Clipped returns the indices within the original coords image
        that define where I should be positioned within the original image.
    """
    top = np.min(coords[:, 0])
    bottom = np.max(coords[:, 0])
    left = np.min(coords[:, 1])
    right = np.max(coords[:, 1])

    I = np.zeros((bottom - top + 1, right - left + 1))
    I[coords[:, 0] - top, coords[:, 1] - left] = True

    clipped = [left, top, right, bottom]

    return I, clipped


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
            polys.append(
                gdf.iloc[i].geometry.buffer(eps, 1, join_style=JOIN_STYLE.mitre)
            )
        #            polys.append(gdf.iloc[i].geometry.buffer(eps))
        else:
            polys.append(gdf.iloc[i].geometry)

    polyout = unary_union(polys)

    if buffer:
        #        polyout = polyout.buffer(-eps)
        polyout = polyout.buffer(-eps, 1, join_style=JOIN_STYLE.mitre)

    return polyout


def area_4326(pgons_4326):
    """
    Given a list of shapely polygons in EPSG:4326, compute the area in km^2.
    Only returns the area of the perimeter; does not yet account for holes
    in the polygon. Mutlipolygons are supported.
    """
    if type(pgons_4326) is Polygon:
        pgons_4326 = [pgons_4326]

    # specify a named ellipsoid
    geod = Geod(ellps="WGS84")

    areas_km2 = [abs(geod.geometry_area_perimeter(p)[0]) / 1e6 for p in pgons_4326]
    
    return areas_km2


def dist_from_da(da, nwidths=20):
    """
    Returns the along-stream distance of a flowline to resolve for a given
    DA. An empirical formula provided by
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013WR013916
    (equation 15) is used to estimate width.

    Parameters
    ----------
    da : float or numeric
        drainage area in km^2
    nwidths : numeric
        number of channel widths to set distance

    Returns
    -------
    dist: float
        Distance in kilometers that represents nwidths*W_bankfull where
        W_bankfull computed according to Wilkerson et al., 2014.

    """
    logda = np.log(da)
    if da < 4.95:
        width = 2.18 * (da**0.191)
    elif da > 337:
        width = 7.18 * (da**0.183)
    elif logda < 1.6:
        width = 2.18 * (da**0.191)
    elif logda < 5.820:
        width = 1.41 * (da**0.462)
    else:
        width = 7.18 * (da**0.183)

    dist = width * nwidths / 1000

    return dist


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
        Distance in meters between each point defined by lats, lons.
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
    Hacky ways to validify a (multi)polygon. If can't be validified, returns the
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


def build_gee_vector_asset(basins, out_path="basins.zip"):
    """Create zipped shapefile for uploading as a Google Earth Engine vector asset.

    Parameters
    ----------
    basins : GeoDataFrame
    out_path : str, optional
        by default "basins.zip"

    Returns
    -------
    str
        path to zip file

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        import geopandas as gpd
        basins = gpd.read_file("tests/results/merit_test/subbasins.json")
        utils.build_gee_vector_asset(basins)
    """

    path_parts = out_path.split("/")
    out_dir = "/".join(path_parts[0 : len(path_parts) - 1])

    os.makedirs("temp/" + out_dir, exist_ok=True)
    temp_dir = Path("temp/" + out_dir)
    basins.to_file(filename="temp/" + out_dir + "/basins.shp", driver="ESRI Shapefile")

    with zipfile.ZipFile(out_path, "w") as zipf:
        for f in temp_dir.glob("*"):
            zipf.write(f, arcname=f.name)

    shutil.rmtree("temp")
    return out_path


def upload_gee_vector_asset(
    zip_path, gee_user, gcp_bucket, gcp_folder="", gcp_upload=True, gee_upload=True
):
    """Upload a zipped shapefile as a GEE vector asset

    Parameters
    ----------
    zip_path : str
        Path to zipped shapefile.
    gee_user : str
        Google Earth Engine user name.
    gcp_bucket : str
        Google Cloud Platform bucket url (e.g. gs://my_bucket_name).
    gcp_folder : str, optional
        Google Cloud Platform bucket folder, by default ""
    gcp_upload : bool, optional
        Set False to skip GCP uploading, by default True
    gee_upload : bool, optional
        Set False to skip GEE uploading, by default True

    Returns
    -------
    str
        GEE asset path

    Raises
    ------
    RuntimeError
        Throws error if gsutil is not installed or authenticated.

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        utils.upload_gee_vector_asset("test.zip", "my_gee_user", "my_gcp_bucket")
    """
    gee_path = (
        "users/" + gee_user + "/" + os.path.splitext(os.path.basename(zip_path))[0]
    )
    if gcp_folder == "":
        out_path = gcp_bucket + "/" + os.path.basename(zip_path)
    else:
        out_path = gcp_bucket + "/" + gcp_folder + "/" + os.path.basename(zip_path)

    if gcp_upload:
        shell_cmd = "gsutil cp " + zip_path + " " + out_path
        print(shell_cmd)
        try:
            subprocess.call(shell_cmd)
        except:
            raise RuntimeError(
                "Errors here could indicate that gsutil is not installed."
            )

    if gee_upload:
        shell_cmd = (
            "earthengine upload table --force --asset_id " + gee_path + " " + out_path
        )
        print(shell_cmd)
        subprocess.call(shell_cmd)

    return gee_path


def upload_gee_tif_asset(
    tif_path,
    gee_user,
    gcp_bucket,
    title,
    gcp_folder="",
    gee_folder="",
    time_start="",
    epsg="4326",
    description="",
    citation="",
    gcp_upload=True,
    gee_upload=True,
    dry_run=False,
    gee_force=False,
):
    """Upload a GeoTIFF file as a GEE raster asset

    Parameters
    ----------
    tif_path : str
        Path to GeoTIFF file
    gee_user : str
        Google Earth Engine user name
    gcp_bucket : str
        Google Cloud Platform bucket url
    title : str
        GEE asset title
    gcp_folder : str, optional
        Google Cloud Platform bucket folder, by default ""
    gee_folder : str, optional
        Google Earth Engine asset folder, by default ""
    time_start : str, optional
        YYYY-MM-DD, by default ""
    epsg : str, optional
        EPSG CRS code, by default "4326"
    description : str, optional
        GEE asset description text, by default ""
    citation : str, optional
        GEE asset citation text, appended to description, by default ""
    gcp_upload : bool, optional
        Set False to skip GCP uploading, by default True
    gee_upload : bool, optional
        Set False to skip GEE uploading, by default True
    dry_run : bool, optional
        Set True to skip GCP and GEE uploading, by default False
    gee_force : bool, optional
        Set True to overwrite any existing GEE assets, by default False

    Returns
    -------
    NoneType
        None

    Warns
    -----
    RuntimeWarning

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        utils.upload_gee_tif_asset("my.tif", "my_gee_user", "my_gcp_bucket")
    """

    if gee_folder == "":
        gee_path = (
            "users/" + gee_user + "/" + os.path.splitext(os.path.basename(tif_path))[0]
        )
        out_path = gcp_bucket + "/" + gcp_folder + "/" + os.path.basename(tif_path)
    else:
        out_path = gcp_bucket + "/" + gee_folder + "/" + os.path.basename(tif_path)
        gee_path = (
            "users/"
            + gee_user
            + "/"
            + gee_folder
            + "/"
            + os.path.splitext(os.path.basename(tif_path))[0]
        )

    if gcp_upload:
        shell_cmd = "gsutil cp " + tif_path + " " + out_path
        print(shell_cmd)
        if not dry_run:
            subprocess.call(shell_cmd)

    if gee_upload:

        force = ""
        if gee_force:
            force = "--force "

        shell_cmd = (
            "earthengine upload image --time_start={time_start} --asset_id={gee_path}"
            " --crs EPSG:{epsg} {force}{out_path}"
        )

        print(shell_cmd)
        if not dry_run:
            try:
                subprocess.call(shell_cmd)
            except:
                warnings.warn(
                    "Are you on Windows? Try installing this fork of the earthengine"
                    "-api package to enable timestamp handling:\n"
                    "https://github.com/jsta/earthengine-api",
                    RuntimeWarning,
                )
                if sys.platform == "win32" and int(time_start[0:4]) < 1970:
                    raise Exception(
                        "Can't upload GEE assets on Windows with time stamps before"
                        " 1970\nhttps://issuetracker.google.com/issues/191997926"
                    )

        if gee_folder != "":
            gee_path = "users/" + gee_user + "/" + gee_folder
        shell_cmd = (
            f'earthengine asset set -p description="{description} Suggested '
            f'citation(s) {citation}" {gee_path}'
        )
        if not dry_run:
            subprocess.call(shell_cmd)

        shell_cmd = 'earthengine asset set -p title="' + title + '" ' + gee_path
        if not dry_run:
            subprocess.call(shell_cmd)

    return None


def drop_column_if_exists(df, col_name_list):
    for col_name in col_name_list:
        if col_name in df.columns:
            df = df.drop([col_name], axis=1)

    return df


def format_freqhist(feature, name_category):
    feature = copy.deepcopy(feature)
    res_hist = pd.DataFrame(feature["properties"]["histogram"], index=[0])
    res_hist.columns = [name_category + "_" + x for x in res_hist.columns]

    del feature["properties"]["histogram"]
    res = pd.DataFrame(feature["properties"], index=[0])
    res["id"] = feature["id"]

    res = pd.concat([res, res_hist], axis=1)

    return res


def coords_to_merit_tile(lon, lat):
    """Identify MERIT-Hydro "tiles" of interest. See
    http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/

    Parameters
    ----------
    lon : float

    lat : float

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        utils.coords_to_merit_tile(178, -17)
        # > "s30e150"
        utils.coords_to_merit_tile(-118, 32)
        # > "n30w120"
        utils.coords_to_merit_tile(-97.355, 45.8358)
        # > "n30w120"
    """

    if abs(lon) > 180 or abs(lat) > 90:
        raise ValueError("Provided coordinates are invalid.")

    nodata_tiles = ["n00w150", "s60w150", "s60w120"]

    def coords_ns_ew(x, less_than_0, gtequal_0):
        if x < 0:
            res = less_than_0 + str(abs(x))
        else:
            res = gtequal_0 + str(x)

        if x == 0:
            res = "n00"
        return res

    lat_segments = [i for i in range(-60, 61, 30)]
    lon_segments = [i for i in range(-180, 151, 30)]

    grid = list(itertools.product(lat_segments, lon_segments))
    grid = pd.DataFrame(grid, columns=("lat", "lon"))
    grid["lat_format"] = grid["lat"].apply(lambda x: coords_ns_ew(x, "s", "n"))
    grid["lon_format"] = grid["lon"].apply(lambda x: coords_ns_ew(x, "w", "e"))

    def closest_less_than(x, y_list):
        res = list(itertools.compress(y_list, [y < x for y in y_list]))
        return res[len(res) - 1]

    res = grid.loc[
        (grid["lat"] == closest_less_than(lat, lat_segments))
        & (grid["lon"] == closest_less_than(lon, lon_segments))
    ]
    res = res.reset_index()
    res = res["lat_format"].to_string(index=False) + res["lon_format"].to_string(
        index=False
    )

    if res in nodata_tiles:
        raise ValueError("MERIT data does not exist for the provided coordinate.")

    return res
