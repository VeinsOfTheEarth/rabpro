"""
Elevation Profile Computation (elev_profile.py)
===============================================

Description
"""

import os
import sys

import geopandas as gpd
import numpy as np
import scipy.interpolate as si
from osgeo import gdal
from pyproj import CRS
from shapely.geometry import LineString, Point

from rabpro import merit_utils as mu
from rabpro import utils as ru


def main(cl_gdf, verbose=False, nrows=50, ncols=50):
    """ Description

    Parameters
    ----------
    cl_gdf : GeoDataFrame
        Desc. Should have a column called 'DA' that stores drainage areas.
        Should be in EPSG 4326 for use of the Haversine formula
    verbose : bool
        Defaults to False.
    nrows : int
        Desc. Defaults to 50.
    ncols : int
        Desc. Defaults to 50.

    Returns
    -------
    cl_gdf : GeoDataFrame
        Desc.
    merit_gdf : GeoDataFrame
        Desc.

    """

    # cl_gdf should have a column called 'DA' that stores drainage areas
    # cl_gdf should be in 4326 for use of the Haversine formula...could add a
    # check and use other methods, but simpler this way.

    # Get data locked and loaded
    dps = ru.get_datapaths()
    hdem_obj = gdal.Open(dps["DEM_elev_hp"])
    da_obj = gdal.Open(dps["DEM_uda"])
    fdr_obj = gdal.Open(dps["DEM_fdr"])
    w_obj = gdal.Open(dps["DEM_width"])

    if verbose:
        print("Extracting flowpath from DEM...")

    if cl_gdf.shape[0] == 1:
        intype = "point"
    else:
        intype = "centerline"

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
            [cl_gdf.geometry.values[0].coords.xy[0][0], cl_gdf.geometry.values[0].coords.xy[1][0],]
        )
        if "DA" in cl_gdf.keys():
            ds_da = cl_gdf.DA.values[0]
        else:
            ds_da = None
        cr_ds_mapped, _ = mu.map_cl_pt_to_flowline(ds_lonlat, da_obj, nrows, ncols, ds_da)

        # Mapping may be impossible
        if np.nan in cr_ds_mapped:
            if verbose is True:
                print("Cannot map provided point to a flowline; no way to extract centerline.")
            return cl_gdf, None

        flowpath = mu.trace_flowpath(fdr_obj, da_obj, cr_ds_mapped)
        es = get_rc_values(hdem_obj, flowpath, nodata=-9999)
        wids = get_rc_values(w_obj, flowpath, nodata=-9999)

    elif intype == "centerline":
        ds_lonlat = np.array(
            [
                cl_gdf.geometry.values[-1].coords.xy[0][0],
                cl_gdf.geometry.values[-1].coords.xy[1][0],
            ]
        )
        us_lonlat = np.array(
            [cl_gdf.geometry.values[0].coords.xy[0][0], cl_gdf.geometry.values[0].coords.xy[1][0],]
        )
        ds_da = cl_gdf.DA.values[-1]
        us_da = cl_gdf.DA.values[0]
        cr_ds_mapped, _ = mu.map_cl_pt_to_flowline(ds_lonlat, da_obj, nrows, ncols, ds_da)
        cr_us_mapped, _ = mu.map_cl_pt_to_flowline(us_lonlat, da_obj, nrows, ncols, us_da)
        flowpath = mu.trace_flowpath(fdr_obj, da_obj, cr_ds_mapped, cr_us_mapped)
        es = get_rc_values(hdem_obj, flowpath, nodata=-9999)
        wids = get_rc_values(w_obj, flowpath, nodata=-9999)

        # Intersect the centerline with the MERIT flowpath to interpolate values
        # to each centerline point

        # Convert the centerline vertices to a set of multilinestrings for intersection
        cli_gdf = gpd.GeoDataFrame(
            geometry=pts_to_line_segments(cl_gdf.geometry.values), crs=cl_gdf.crs
        )

        # Prepare the DEM-flowpath for intersection with the user-provided centerline
        # DEM is already in EPSG:4326
        dem_ll = ru.xy_to_coords(flowpath[1], flowpath[0], hdem_obj.GetGeoTransform())
        dem_cl_lonlat = [Point(d0, d1) for d0, d1 in zip(dem_ll[0], dem_ll[1])]
        demi_gdf = gpd.GeoDataFrame(
            geometry=pts_to_line_segments(dem_cl_lonlat), crs=CRS.from_epsg(4326)
        )

        # Intersect centerline with DEM-flowpath
        res_intersection = gpd.sjoin(cli_gdf, demi_gdf, op="intersects", how="inner")

        if verbose:
            print(
                "Found {} intersections between provided centerline and DEM flowpath.".format(
                    len(res_intersection)
                )
            )

        # Map each point of the centerline to the DEM-flowpath
        mapper = {}
        att_keys = ru.parse_keys(cl_gdf)
        if att_keys["distance"] is None:
            dists = compute_dists(cl_gdf)
        else:
            dists = cl_gdf[att_keys["distance"]].values
        for r in res_intersection.iterrows():
            clidx = r[0]
            demidx = r[1]["index_right"]

            # Get intersection point
            ls_cl = cli_gdf.geometry.values[clidx]
            ls_dem = demi_gdf.geometry.values[demidx]
            int_pt = ls_cl.intersection(ls_dem).coords.xy

            # Determine which point to map the intersection to by finding the closest
            # along centerline and DEM-flowpath
            us_dist_cl = ru.haversine(
                (ls_cl.coords.xy[1][0], int_pt[1][0]), (ls_cl.coords.xy[0][0], int_pt[0][0]),
            )[0]
            ds_dist_cl = ru.haversine(
                (ls_cl.coords.xy[1][1], int_pt[1][0]), (ls_cl.coords.xy[0][1], int_pt[0][0]),
            )[0]
            if us_dist_cl < ds_dist_cl:
                cl_idx = clidx
            else:
                cl_idx = clidx + 1

            us_dist_dem = ru.haversine(
                (ls_dem.coords.xy[1][0], int_pt[1][0]), (ls_dem.coords.xy[0][0], int_pt[0][0]),
            )[0]
            ds_dist_dem = ru.haversine(
                (ls_dem.coords.xy[1][1], int_pt[1][0]), (ls_dem.coords.xy[0][1], int_pt[0][0]),
            )[0]
            if us_dist_dem < ds_dist_dem:
                dem_idx = demidx
            else:
                dem_idx = demidx + 1
            mapper[cl_idx] = dem_idx

        # Make the elevation profile using the mapping
        cl_elevs = np.ones(len(cl_gdf)) * np.nan
        cl_wids = np.ones(len(cl_gdf)) * np.nan
        for k in mapper.keys():
            cl_elevs[k] = es[mapper[k]]
            cl_wids[k] = wids[mapper[k]]

        # Assign the first and last centerline elevation values to match the DEM
        # These points have already been mapped
        cl_elevs[0] = es[0]
        cl_elevs[-1] = es[-1]

        nans = np.isnan(cl_elevs)

        # Fill in the nans using a linear interpolation between known values
        # First find all the groups of nans that need interpolating across
        e_nangroups = find_nangroups(cl_elevs)
        w_nangroups = find_nangroups(cl_wids)

        # Do the interpolations
        cl_elevs = interpolate_nangroups(cl_elevs, dists, e_nangroups)
        cl_wids = interpolate_nangroups(cl_wids, dists, w_nangroups)

        # Store the elevation profile and distances in the centerline geodataframe
        cl_gdf["MERIT Elev (m)"] = cl_elevs
        cl_gdf["Distance (m)"] = compute_dists(cl_gdf)
        cl_gdf["MERIT Width (m)"] = cl_wids
        cl_gdf["intersected_DEM_flowline?"] = ~nans

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
    merit_gdf["Distance (m)"] = compute_dists(merit_gdf)

    return cl_gdf, merit_gdf


def compute_dists(gdf):
    """ Computes cumulative distance in meters between points in gdf.

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


def get_rc_values(gdobj, rc, nodata=-9999):
    """ Returns the values within the raster pointed to by gdobj specified by
    the row, col values in rc. Sets nodata. Returns numpy array.

    Parameters
    ----------
    gdobj : [type]
        Points to raster to get values from
    rc : [type]
        [description]
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


def pts_to_line_segments(pts):
    """ Converts a list of shapely points to a set of line segments. Points
    should be in order.

    Parameters
    ----------
    pts : list of shapely.geometry.Point
        [description]

    Returns
    -------
    list of shapely.geometry.LineString
        length N-1, where N=length(pts)
    """
    ls = [LineString((pts[i], pts[i + 1])) for i in range(len(pts) - 1)]

    return ls


def find_nangroups(arr):
    """ Returns groups of nans in an array.

    Parameters
    ----------
    arr : [type]
        [description]

    Returns
    -------
    list
        [description]
    """
    nans = np.isnan(arr)
    nangroups = []
    nangroup = []
    for i, n in enumerate(nans):
        if not n:
            if len(nangroup) > 0:
                nangroups.append(nangroup)
            nangroup = []
        else:
            nangroup.append(i)

    return nangroup


def interpolate_nangroups(arr, dists, nangroups):
    """ Linearly interpolates across groups of nans in a 1-D array.

    Parameters
    ----------
    arr : [type]
        [description]
    dists : [type]
        [description]
    nangroups : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    for ng in nangroups:
        if type(ng) is int:
            ng = [ng]
        if 0 in ng or len(arr) - 1 in ng:
            continue
        interp = si.interp1d(
            (dists[ng[0] - 1], dists[ng[-1] + 1]), (arr[ng[0] - 1], arr[ng[-1] + 1])
        )
        interp_pts = dists[np.arange(ng[0], ng[-1] + 1)]
        arr[np.arange(ng[0], ng[-1] + 1)] = interp(interp_pts)

    return arr
