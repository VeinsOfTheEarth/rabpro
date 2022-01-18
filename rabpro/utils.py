"""
Utility functions (utils.py)
============================

"""

import os
import shutil
import zipfile
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal, ogr
from skimage import measure
import http.client as httplib
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

try:
    import rabpro.data_utils as du
except:
    import data_utils as du

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


def get_datapaths(datapath=None, configpath=None, rebuild_vrts=True, **kwargs):
    """
    Returns a dictionary of paths to all data that RaBPro uses. Also builds
    virtual rasters for MERIT data.

    Parameters
    ----------
    datapath: string, optional
        path to rabpro data folder, will read from an environment variable "RABPRO_DATA", if not set uses appdirs
    configpath: string, optional
        path to rabpro config folder, will read from an environment variable "RABPRO_CONFIG", if not set uses appdirs
    rebuild_vrts: boolean, optional
        rebuild virtual raster files, default is True
    kwargs:
        arguments passed to build_vrt

    Returns
    -------
    dict
        contains paths to all data that RaBPro uses

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        utils.get_datapaths()
        utils.get_datapaths(extents=[-180.00041666666667, 179.99958333333913, 84.99958333333333, -60.000416666669], quiet=True)
    """

    global _DATAPATHS
    if _DATAPATHS is not None:
        _build_virtual_rasters(_DATAPATHS, force_rebuild=rebuild_vrts, **kwargs)
        return _DATAPATHS

    du.create_file_structure(datapath=datapath, configpath=configpath)
    datapaths = du.create_datapaths(datapath=datapath, configpath=configpath)
    if has_internet():
        du.download_gee_metadata()

    _build_virtual_rasters(datapaths, force_rebuild=rebuild_vrts, **kwargs)
    _DATAPATHS = datapaths
    return datapaths


def _build_virtual_rasters(datapaths, force_rebuild=True, quiet=False, **kwargs):
    msg_dict = {
        "DEM_fdr": "Building flow direction virtual raster DEM from MERIT tiles...",
        "DEM_uda": "Building drainage areas virtual raster DEM from MERIT tiles...",
        "DEM_elev_hp": "Building hydrologically-processed elevations virtual raster DEM from MERIT tiles...",
        "DEM_width": "Building width virtual raster from MERIT tiles...",
    }

    # Ensure that DEM virtual rasters are built
    for key in msg_dict:
        if not os.path.isfile(datapaths[key]) or force_rebuild:
            if not quiet:
                print(msg_dict[key])
            build_vrt(
                os.path.dirname(os.path.realpath(datapaths[key])),
                outputfile=datapaths[key],
                quiet=quiet,
                **kwargs,
            )


def get_exportpaths(name, basepath=None, overwrite=False):
    """Returns a dictionary of paths for exporting RaBPro results. Also creates
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


# def parse_keys(gdf): # Used only in centerline, deprecated
#     """
#     Attempts to interpret the column names of the input dataframe.
#     In particular, looks for widths and distances along centerline.

#     Parameters
#     ----------
#     gdf : GeoDataFrame
#         table to parse

#     Returns
#     -------
#     dict
#         contains column names and corresponding properties
#     """
#     keys = gdf.keys()
#     parsed = {"distance": None, "width": None}
#     for k in keys:
#         if "distance" in k.lower():
#             parsed["distance"] = k
#         if "width" in k.lower():
#             parsed["width"] = k

#     return parsed


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
        [description], by default False
    quiet : bool, optional
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
        if not quiet:
            print(stdout.decode())

    return vrtname


def raster_extents(raster_path):
    """Output raster extents as [xmin, xmax, ymin, ymax]

    Example:
        import utils
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


def parse_path(path):
    """
    Parses a file or folderpath into: base, folder (where folder is the
    outermost subdirectory), filename, and extention. Filename and extension are
    empty if a directory is passed.
    """

    if (
        path[0] != os.sep and platform.system() != "Windows"
    ):  # This is for non-windows...
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


def regionprops(I, props, connectivity=2):
    """
    Finds blobs within a binary image and returns requested properties of
    each blob.
    This function was modeled after matlab's regionprops and is essentially
    a wrapper for skimage's regionprops. Not all of skimage's available blob
    properties are available here, but they can easily be added.
    Taken from RivGraph.im_utils

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
        print("Cannot compute the following properties: {}".format(cant_do))

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

                # Convert to cv2-ingestable data type
                Ip = np.array(Ip, dtype="uint8")
                contours, _ = cv2.findContours(Ip, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # IMPORTANT: findContours returns points as (x,y) rather than (row, col)
                contours = contours[0]
                crows = []
                ccols = []
                for c in contours:
                    # must add back the cropped rows and columns, as well as the single-pixel pad
                    crows.append(c[0][1] + cropped[1] - 1)
                    ccols.append(c[0][0] + cropped[0] - 1)
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
            print("{} is not a valid property.".format(prop))

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


def build_gee_vector_asset(basins, out_path="basins"):
    """Create zipped shapefile for uploading as a Google Earth Engine vector asset.

    Args:
        basins ([type]): [description]
        out_path (str, optional): [description]. Defaults to "basins".

    Returns:
        [type]: [description]

    Examples
    --------
    .. code-block:: python

        from rabpro import utils
        import geopandas as gpd
        basins = gpd.read_file("tests/results/merit_test/subbasins.json")
        utils.build_gee_vector_asset(basins)
    """
    os.makedirs("temp", exist_ok=True)
    temp_dir = Path("temp")
    basins.to_file(filename="temp/" + out_path + ".shp", driver="ESRI Shapefile")

    with zipfile.ZipFile(out_path + ".zip", "w") as zipf:
        for f in temp_dir.glob("*"):
            zipf.write(f, arcname=f.name)

    shutil.rmtree("temp")
    return out_path + ".zip"


def upload_gee_vector_asset(
    zip_path, gee_user, gcp_bucket, gee_folder="", gcp_upload=True, gee_upload=True
):
    """[summary]

    Args:
        zip_path ([type]): [description]
        gee_user ([type]): [description]
        gcp_bucket ([type]): [description]
        gee_folder (str, optional): [description]. Defaults to "".
        gcp_upload (bool, optional): [description]. Defaults to True.
        gee_upload (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    
    Examples
    --------
    .. code-block:: python

        upload_gee_vector_asset("test.zip", "my_gee_user", "my_gcp_bucket")
    ```
    """
    gee_path = (
        "users/" + gee_user + "/" + os.path.splitext(os.path.basename(zip_path))[0]
    )
    if gee_folder == "":
        out_path = gcp_bucket + "/" + os.path.basename(zip_path)
    else:
        out_path = gcp_bucket + "/" + gee_folder + "/" + os.path.basename(zip_path)

    if gcp_upload:
        shell_cmd = "gsutil cp " + zip_path + " " + out_path
        print(shell_cmd)
        subprocess.call(shell_cmd)

    if gee_upload:
        shell_cmd = "earthengine upload table --asset_id " + gee_path + " " + out_path
        print(shell_cmd)
        subprocess.call(shell_cmd)

    return gee_path


def upload_gee_tif_asset(
    asset_filename, gcp_bucket, gcp_filename, gee_collection="", time_start="",
):
    if gee_collection != "":
        gee_collection = gee_collection + "/"

    shell_cmd = (
        "earthengine upload image" + " --time_start=" + time_start + " --asset_id=",
        gee_collection
        + asset_filename
        + " --crs EPSG:4326"
        + " --force"
        + " "
        + gcp_bucket
        + "/"
        + gcp_filename,
    )
    print(shell_cmd)
    subprocess.call(shell_cmd)

    return None
