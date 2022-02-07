"""
Elevation Profile Computation (elev_profile.py)
===============================================

"""

import numpy as np
from osgeo import gdal
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import Point

from rabpro import utils as ru
from rabpro import merit_utils as mu


def main(cl_gdf, verbose=False, nrows=50, ncols=50):
    """Compute the elevation profile. The profile is computed such that the provided coordinate is the centerpoint (check if this is true).

    Parameters
    ----------
    cl_gdf : GeoDataFrame
        Desc. Should have a column called 'DA' that stores drainage areas.
        Should be in EPSG 4326 for use of the Haversine formula
    verbose : bool
        Defaults to False.
    nrows : int
        Number of rows in the neighborhood of the point to search. Defaults to 50.
    ncols : int
        Number of columns in the neighborhood of the point to search. Defaults to 50.

    Returns
    -------
    cl_gdf : GeoDataFrame
        The original point layer geometry.
    merit_gdf : GeoDataFrame
        The elevation profile geometry.

    Examples
    --------
    .. code-block:: python

        import rabpro
        coords = (56.22659, -130.87974)
        rpo = rabpro.profiler(coords, name='basic_test')
        cl_gdf, merit_gdf = rabpro.elev_profile.main(rpo.gdf)
    """

    # cl_gdf should have a column called 'DA' that stores drainage areas
    # cl_gdf should be in 4326 for use of the Haversine formula...could add a
    # check and use other methods, but simpler this way.

    # Get data locked and loaded
    dps = ru.get_datapaths(rebuild_vrts=False)
    hdem_obj = gdal.Open(dps["DEM_elev_hp"])
    da_obj = gdal.Open(dps["DEM_uda"])
    fdr_obj = gdal.Open(dps["DEM_fdr"])
    w_obj = gdal.Open(dps["DEM_width"])

    if verbose:
        print("Extracting flowpath from DEM...")

    if cl_gdf.shape[0] == 1:
        intype = "point"
    else:
        intype = "centerline"  # deprecated
        raise DeprecationWarning(
            "elev_profile only supports single 'point' coordinate pairs, not multipoint 'centerlines'"
        )

    # Here, we get the MERIT flowline corresponding to the centerline. If we
    # are provided only a single point, the flowline is delineated to its
    # terminal point. Otherwise, the flowline is delineated to the limits of
    # the provided centerline. Elevation and width profiles are then extracted.
    # In addition, if a centerline is provided rather than a point, it is
    # intersected against the MERIT flowline and values are interpolated along
    # its path.
    if intype == "point":
        # Trace the centerline all the way up to the headwaters
        ds_lonlat = np.array(
            [
                cl_gdf.geometry.values[0].coords.xy[0][0],
                cl_gdf.geometry.values[0].coords.xy[1][0],
            ]
        )
        if "DA" in cl_gdf.keys():
            ds_da = cl_gdf.DA.values[0]
        else:
            ds_da = None
        cr_ds_mapped, _ = mu.map_cl_pt_to_flowline(
            ds_lonlat, da_obj, nrows, ncols, ds_da
        )

        # Mapping may be impossible
        if np.nan in cr_ds_mapped:
            if verbose is True:
                print(
                    "Cannot map provided point to a flowline; no way to extract centerline."
                )
            return cl_gdf, None

        flowpath = mu.trace_flowpath(fdr_obj, da_obj, cr_ds_mapped)
        es = _get_rc_values(hdem_obj, flowpath, nodata=-9999)
        wids = _get_rc_values(w_obj, flowpath, nodata=-9999)

    # Store the elevation profile and distances of the MERIT-derived flowpath
    coords_fp = ru.xy_to_coords(flowpath[1], flowpath[0], da_obj.GetGeoTransform())
    merit_gdf = gpd.GeoDataFrame(
        data={
            "geometry": [Point(x, y) for x, y in zip(coords_fp[0], coords_fp[1])][::-1],
            "Elevation (m)": es[::-1],
            "Width (m)": wids[::-1],
            "row": flowpath[0],
            "col": flowpath[1],
        },
        crs=CRS.from_epsg(4326),
    )
    merit_gdf["Distance (m)"] = _compute_dists(merit_gdf)

    return cl_gdf, merit_gdf


def _compute_dists(gdf):
    """Computes cumulative distance in meters between points in gdf.

    Parameters
    ----------
    gdf : GeoDataFrame
        Desc.

    Returns
    -------
    numpy.ndarray
        Desc.

    Raises
    ------
    TypeError
        If gdf does not contain shapely.geometry.Point values.
    """
    gdfc = gdf.copy()

    if type(gdfc.geometry.values[0]) is not Point:
        raise TypeError("Cannot compute distances for non-point type geometries.")

    if gdfc.crs.to_epsg() != 4326:
        gdfc = gdfc.to_crs("EPSG:4326")

    # Compute distances along the centerline for each point
    lats = [pt.coords.xy[1][0] for pt in gdf.geometry.values]
    lons = [pt.coords.xy[0][0] for pt in gdf.geometry.values]
    ds = ru.haversine(lats, lons)
    ds = np.insert(ds, 0, 0)
    dists = np.cumsum(ds)

    return dists


def _get_rc_values(gdobj, rc, nodata=-9999):
    """Returns the values within the raster pointed to by gdobj specified by
    the row, col values in rc. Sets nodata. Returns numpy array.

    Parameters
    ----------
    gdobj : osgeo.gdal.Dataset
        Points to raster to get values from
    rc : tuple
        A tuple containing lists of row and col values.
    nodata : int, optional
        No data value for the raster, by default -9999

    Returns
    -------
    numpy.ndarray
        Raster values
    """

    vals = []
    for r, c in zip(rc[0], rc[1]):
        vals.append(gdobj.ReadAsArray(xoff=int(c), yoff=int(r), xsize=1, ysize=1)[0][0])
    vals = np.array(vals)
    vals[vals == nodata] = np.nan

    return vals
