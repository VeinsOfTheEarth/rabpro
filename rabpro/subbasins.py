"""
Subbasin Computation (subbasins.py)
===================================

Functions to calculate subbasin geometries.
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from osgeo import gdal
from pyproj import CRS
from shapely.geometry import Point, MultiPolygon

from rabpro import merit_utils as mu
from rabpro import utils as ru


def main_hb(cl_gdf, verbose=False):
    """[summary]

    Parameters
    ----------
    cl_gdf : GeoDataFrame
        Centerline geometry
    verbose : bool, optional
        By default False

    Returns
    -------
    subbasins_gdf : GeoDataFrame
        Contains subbasin geometries
    sb_inc_gdf : GeoDataFrame
        [description]
    cl_das : [type]
        Drainage areas

    Raises
    ------
    RuntimeWarning
        If cl_gdf has no CRS defined
    """
    datapaths = ru.get_datapaths()

    # Convert the cl_gdf to EPSG:4326 if necessary in order to align with HydroSheds
    was_transformed = False
    if cl_gdf.crs is None:
        raise RuntimeWarning("Centerline geodataframe has no defined CRS.")
    elif cl_gdf.crs.to_authority() != ("EPSG", "4326"):
        orig_crs = cl_gdf.crs  # Save the original crs to put it back later
        cl_gdf = cl_gdf.to_crs(epsg=4326)
        was_transformed = True

    # Load the appropriate HydroBasins shapefile as a geodataframe
    HB_gdf = load_continent_basins(cl_gdf, datapaths["HydroBasins1"], datapaths["HydroBasins12"])

    # Find the chain of polygons
    if verbose:
        print("Finding subbasin polygon chain...")
    chainids = initial_basin_chain(HB_gdf, cl_gdf)

    # Map centerline points to chain basins and get drainage areas for each point
    if verbose:
        print("Mapping centerline points to polygons in chain...")
    idxmap, DA = map_points_to_chain(HB_gdf, cl_gdf, chainids)

    # Delineate each of the subbasins
    if verbose:
        print("Delineating subbasins...")
    subbasins_gdf, sb_inc_gdf = delineate_subbasins(idxmap, HB_gdf)

    # Map the Basin Length from each cl point to the appropriate basin via idxmap
    basin_dists = HB_gdf.iloc[idxmap].DIST_SINK.values
    # This isn't the most rigorous way to perform this mapping, but it works
    basin_dists = np.sort(list(set(basin_dists)))[::-1]
    flow_length_in_basin = -np.diff(basin_dists)
    # First (most-upstream) basin is incalculable, so insert a nan
    flow_length_in_basin = np.insert(flow_length_in_basin, 0, np.nan)

    # Construct the map of cl points -> which delineated subbasin. This will be
    # used for assigning raster statistics to each centerline point.
    # Re-index the indexmap from 0 --> len(idxmap)
    idxmap = np.array(idxmap, dtype=np.int)
    unique_sb_idcs = np.sort(np.unique(idxmap, return_index=True)[1])
    ct = 0
    for ui in unique_sb_idcs:
        idxmap[idxmap == idxmap[ui]] = ct
        ct = ct + 1

    # Transform back to original CRS if necessary
    if was_transformed is True:
        cl_gdf = cl_gdf.to_crs(orig_crs)

    # Add drainage areas to cl_gdf
    cl_das = subbasins_gdf.areas.values[idxmap]

    # Rename areas to DA
    renamer = {"areas": "DA"}

    subbasins_gdf = subbasins_gdf.rename(columns=renamer)
    sb_inc_gdf = sb_inc_gdf.rename(columns=renamer)

    return subbasins_gdf, sb_inc_gdf, cl_das


def load_continent_basins(cl_gdf, level_one, level_twelve):
    """[summary]

    Parameters
    ----------
    cl_gdf : GeoDataFrame
        Centerline data. Should be in EPSG:4326.
    level_one : str
        Path to level 1 HydroBasins data
    level_twelve : str
        Path to level 12 HydroBasins data

    Returns
    -------
    GeoDataFrame
        [description]

    Raises
    ------
    ValueError
        If the provided point in cl_gdf doesn't fall in HydroBasins
    """

    # Prepare load level 1 dataframe
    level_one_path = str(Path(level_one) / "hybas_all_lev01_v1c.shp")
    level_one_df = gpd.read_file(level_one_path)

    # Find the first point of the centerline to figure out which continent we're in
    xy_cl = cl_gdf.geometry.values[0].coords.xy
    cl_us_pt = gpd.GeoDataFrame(geometry=[Point(xy_cl[0][0], xy_cl[1][0])])
    cl_us_pt.crs = cl_gdf.crs

    # Intersect with level-1 HydroBasins to figure out which continent we're in
    clpt_level_onei = gpd.sjoin(cl_us_pt, level_one_df, op="intersects")
    if len(clpt_level_onei) == 0:
        raise ValueError(
            "Provided point ({}) does not fall within HydroBasins. Check that lat/lon are not reversed in input.".format(
                [xy_cl[0][0], xy_cl[1][0]]
            )
        )
    id_no = clpt_level_onei.PFAF_ID[0]

    # Load the appropriate level 12 dataframe
    loadnames = ["af", "eu", "si", "as", "au", "sa", "na", "ar", "gr"]
    level_twelve_path = str(
        Path(level_twelve) / str("hybas_" + loadnames[id_no - 1] + "_lev12_v1c.shp")
    )

    # Load the appropriate level-12 Hydrobasins continent shapefile
    HB_gdf = gpd.read_file(level_twelve_path)

    #    # Return the crs to original if trasnformed
    #    if was_transformed:
    #        cl_gdf = cl_gdf.to_crs(orig_crs)

    return HB_gdf


def initial_basin_chain(HB_gdf, cl_gdf, buf_wid=0.1):
    """
    Finds the chain of drainage basins from the upstream-most centerline point
    to the sink (e.g. ocean).

    Parameters
    ----------
    HB_gdf : GeoDataFrame
        [description]
    cl_gdf : GeoDataFrame
        Centerline coordinates
    buf_wid : float, optional
        [description], by default 0.1

    Returns
    -------
    list
        [description]

    Raises
    ------
    RuntimeError
        If it cannot find a chain of basins that includes more than half the
        input centerline points
    """

    def get_chain(HB_gdf, basin_id_start):
        """
        Given an input geodataframe called HB_gdf that is created from a
        HydroBasins shapefile, this function returns all the downstream basins
        of an input basin_id_start corresponding to a HYBAS_ID
        """
        chain = [basin_id_start]
        while 1:
            next_basin_id = HB_gdf[HB_gdf.HYBAS_ID.values == chain[-1]].index[0]
            chain.append(HB_gdf.NEXT_DOWN[next_basin_id])
            if chain[-1] == 0:
                chain.pop()
                break

        return chain

    def frac_pts_within_chain(HB_gdf, chain, cl_gdf, buf_wid=0.1):
        """
        Given an input chain created by get_chain and a geodataframe containing
        centerline coordinates (or any coordinates), this function returns the
        fraction of the input coordinates that are within the chain. The chain
        is first buffered by buf_wid, which is in units of the chain's native
        projection (i.e. WGS84, no projection -> units = degrees).
        """

        chainids = [HB_gdf.index[HB_gdf.HYBAS_ID == c].values[0] for c in chain]
        chainHB_gdf = HB_gdf.loc[chainids]

        # Combine the chain basins into single polygon
        chainHB_gdf = chainHB_gdf.dissolve(by="MAIN_BAS")

        # Buffer the chain polygon
        chainHB_gdf["geometry"] = chainHB_gdf["geometry"].buffer(buf_wid)

        # Intersect chain polygon with centerline points
        chaincl_HB_gdf = gpd.sjoin(cl_gdf, chainHB_gdf, op="intersects")

        ## Output chain shapefile for visualzation in GIS
        #        chainHB_gdf.to_file(r"X:\temp" + 'test_chain.shp')

        # Fraction of points within the buffered chain
        frac_pts_within = len(chaincl_HB_gdf) / len(cl_gdf)

        return frac_pts_within

    # Get intersection of centerline points and basins
    basin_intersect = gpd.sjoin(cl_gdf, HB_gdf, op="intersects")

    # Find index of basin that upstream-most point is in
    basin_id_start = basin_intersect.HYBAS_ID.values[0]

    # Get the set of IDs comprising the chain
    chain = get_chain(HB_gdf, basin_id_start)

    # Get the fraction of centerline points within the chain
    frac_pts_within = frac_pts_within_chain(HB_gdf, chain, cl_gdf, buf_wid)

    # Check if the centerline coordinates are mostly within the chain   -
    # if not, loop through the neighboring basins and see if any of them
    # serve as appropriate chain basins
    if frac_pts_within < 0.5:
        print("Initial basin guess was not correct...trying neighboring basins.")

        # Get the polygon IDs that border the initially-guessed basin
        initial_pgon = HB_gdf[HB_gdf.HYBAS_ID == basin_id_start]

        # Find its neighbors via intersection
        neighbors = gpd.sjoin(initial_pgon, HB_gdf, op="intersects")
        neigh_idcs = set(neighbors.HYBAS_ID_right.values)

        # Remove neighbors we've already looked at
        neigh_idcs = neigh_idcs - set(chain)

        while frac_pts_within < 0.5:
            basin_id_start = neigh_idcs.pop()
            chain = get_chain(HB_gdf, basin_id_start)
            frac_pts_within = frac_pts_within_chain(HB_gdf, chain, cl_gdf, buf_wid)
            neigh_idcs = neigh_idcs - set(chain)

            if len(neigh_idcs) == 0 and frac_pts_within < 0.5:
                raise RuntimeError(
                    "Could not find a chain of basins that includes more than half the input centerline points."
                )
        print("Found an initial basin whose chain contains > 50% of centerline points.")

    chainids = [HB_gdf.index[HB_gdf.HYBAS_ID == i].values[0] for i in chain]

    return chainids


def map_points_to_chain(HB_gdf, cl_gdf, chainids):
    """
    Maps all centerline points to level-12 drainage basins within chain. 
    1. Identify the polygon each point falls into. 
    2. Push the points that don't fall into a polygon into the nearest
    (Euclidean distance) one. 
    3. Check that drainage area does not decrease as we move downstream.
    If so, assign point to previous (next upstream) basin.

    Parameters
    ----------
    HB_gdf : GeoDataFrame
        [description]
    cl_gdf : GeoDataFrame
        Centerline coordinates
    chainids : [type]
        [description]

    Returns
    -------
    idxap : list
        Maps each centerline point to a basin index within the HB_gdf
        GeoDataFrame
    DA : int
        Drainage areas for each centerline point
    """

    # Re-intersect the individual basin polygons with the centerline points
    chainHB_gdf = HB_gdf.loc[chainids]
    cl_in_basins = gpd.sjoin(cl_gdf, chainHB_gdf, op="intersects")

    # In case a centerline point is EXACTLY on the border of two basins, push
    # it into the upstream one
    u_idx, u_ct = np.unique(cl_in_basins.index.values, return_counts=True)
    idx_in_two_basins = np.where(u_ct > 1)[0]
    remrow = np.array([], dtype=np.int)
    for i in idx_in_two_basins:
        idx = u_idx[i]
        rowidcs = np.where(idx == cl_in_basins.index.values)[0]
        # Find upstream-most basin
        idxkeep = np.argmax(cl_in_basins.iloc[rowidcs].DIST_MAIN.values)
        rowidcs = np.delete(rowidcs, idxkeep)
        remrow = np.concatenate((remrow, rowidcs))
    # Now drop the rows that represent a second-intersection
    cl_in_basins.drop(cl_in_basins.index[remrow], inplace=True)

    # Extract Drainage Areas from polygons, put Nones where the point falls outside of any polygons
    # Also create a map between each point and its corresponding polygon index in
    # the original dataframe
    idxmap = []
    DA = []
    for i in range(len(cl_gdf)):
        try:
            DA.append(cl_in_basins.loc[i].UP_AREA.values)
            idxmap.append(cl_in_basins.loc[i].index_right.values)
        except:
            DA.append(None)
            idxmap.append(None)

    # Assign the cl points that are not in a chain basin the DA of the nearest
    # chain basin
    # Which points need to be fixed?
    fixid = [i for i, j in enumerate(idxmap) if j is None]
    # Fix 'em
    for fid in fixid:
        clgeom = cl_gdf.loc[fid].geometry

        # Compute distance between point and all polygons in chain [CAN REDUCE SEARCH DOMAIN HERE FOR FASTER PROCESSING]
        dists = []
        for i in chainHB_gdf.index:
            dists.append(clgeom.distance(chainHB_gdf.geometry[i]))
        minidx = dists.index(min(dists))
        # Assign DA of nearest chain polygon
        idxmap[fid] = chainHB_gdf.index.values[minidx]
        DA[fid] = chainHB_gdf.UP_AREA.values[minidx]

    # Now that DAs have been assigned for all points, find those that cannot be
    # correct (i.e. those that decrease with downstream distance) and fix them
    for i in range(1, len(DA)):
        if DA[i] < DA[i - 1]:
            DA[i] = DA[i - 1]
            idxmap[i] = idxmap[i - 1]

    return idxmap, DA


def delineate_subbasins(idxmap, HB_gdf):
    """ Finds all the upstream contributing basins for each basin in idxmap.
    This could perhaps be optimized, but the current implementation just solves
    each basin in idxmap independently.

    Parameters
    ----------
    idxmap : list
        [description]
    HB_gdf : GeoDataFrame
        [description]

    Returns
    -------
    subHB_gdf : GeoDataFrame
        Contains the polygons of each basin's catchment
    inc_df : GeoDataFrame
        Contains the polygons of the incremental catchments. The upstream-most
        basin will be the largest polygon in most cases, but that depends on the
        input centerline.
    """

    # idxmap contains only the polygons (indices) in the chain that contain
    # centerline points, arranged in us->ds direction. Use it to determine
    # which basins to delineate
    chainids = [
        x for i, x in enumerate(idxmap) if x not in idxmap[0:i]
    ]  # unique-ify list - don't wanna lose order
    chainids = np.ndarray.tolist(
        np.array(chainids, dtype=int)
    )  # convert to native int from numpy int

    # Get (incremental) indices of all subbasins
    subbasin_idcs = find_contributing_basins(chainids, HB_gdf)

    # Make polygons of the incremental subbasins
    inc_df = gpd.GeoDataFrame(
        index=range(0, len(subbasin_idcs)), columns=["geometry", "areas"], crs=HB_gdf.crs,
    )
    subHB_gdf = gpd.GeoDataFrame(
        index=range(0, len(subbasin_idcs)), columns=["geometry", "areas"], crs=HB_gdf.crs,
    )

    for i, si in enumerate(subbasin_idcs):
        # Incremental subbasins
        inc_df.geometry.values[i] = ru.union_gdf_polygons(HB_gdf, si, buffer=True)
        subHB_gdf.areas.values[i] = np.max(HB_gdf.iloc[list(si)].UP_AREA.values)

    # Combine the incremental subbasins to get the polygons of entire subbasins
    # for each centerline point; buffer and un-buffer the polygons to account for
    # "slivers"
    for i in range(len(inc_df)):
        if i == 0:
            subHB_gdf.geometry.values[i] = inc_df.geometry.values[i]
            inc_df.areas.values[i] = subHB_gdf.areas.values[i]
        #            inc_df.loc[i].areas = subHB_gdf.loc[i].areas
        else:
            # Put geometries into dataframe
            #            temp_gdf = gpd.GeoDataFrame(index=range(0,2), columns=['geometry'], crs=HB_gdf.crs)
            #            temp_gdf.geometry = [inc_df.loc[i].geometry, subHB_gdf.loc[i-1].geometry]
            #            subHB_gdf.loc[i].geometry = ru.union_gdf_polygons(temp_gdf, range(0, 2))
            #            inc_df.loc[i].areas = subHB_gdf.loc[i].areas - subHB_gdf.loc[i-1].areas
            temp_gdf = gpd.GeoDataFrame(index=range(0, 2), columns=["geometry"], crs=HB_gdf.crs)
            temp_gdf.geometry = [
                inc_df.geometry.values[i],
                subHB_gdf.geometry.values[i - 1],
            ]
            subHB_gdf.geometry.values[i] = ru.union_gdf_polygons(temp_gdf, range(0, 2))
            inc_df.areas.values[i] = subHB_gdf.areas.values[i] - subHB_gdf.areas.values[i - 1]

    return subHB_gdf, inc_df


def find_contributing_basins(chainids, HB_gdf):
    """
    Given an input GeoDataFrame of HydroBasins shapefiles and a list of chainids
    denoting which basins are part of the chain, this function walks upstream
    from the upstream-most basin by following the "NEXT_DOWN" attribute until
    all possible basins are included. This process is repeated, but stops when
    the previous basin is encountered. The result is a list of sets, where each
    set contains the INCREMENTAL basin indices for each subbasin. i.e. the
    most-downstream subbasin would be found by unioning all the sets.

    IMPORTANT: chainids must be arranged in US->DS direction.

    Parameters
    ----------
    chainids : list
        Denotes which basins are part of the chain
    HB_gdf : GeoDataFrame
        HydroBasins shapefiles

    Returns
    -------
    list of sets
        each set contains the incremental basin indices for each subbasin
    """
    subbasin_idcs = []
    visited_subbasins = set()

    for idx in chainids:
        sb_idcs = set([idx])
        sb_check = set(HB_gdf[HB_gdf.NEXT_DOWN == HB_gdf.HYBAS_ID[idx]].index)
        while sb_check:
            idx_check = sb_check.pop()

            if idx_check in visited_subbasins:
                continue

            sb_idcs.add(idx_check)

            basin_id_check = HB_gdf.HYBAS_ID[idx_check]
            sb_check = (
                sb_check | set(HB_gdf[HB_gdf.NEXT_DOWN == basin_id_check].index) - visited_subbasins
            )

        # Store the incremental indices
        subbasin_idcs.append(sb_idcs)

        # Update the visited subbasins (so we don't redo them)
        visited_subbasins = visited_subbasins | sb_idcs

    return subbasin_idcs


def main_merit(cl_gdf, da, nrows=51, ncols=51, map_only=False, verbose=False):
    """ Calculates subbasins using MERIT

    Parameters
    ----------
    cl_gdf : GeoDataFrame
        Centerline coordinates
    da : int
        Drainage area
    nrows : int
        [desc], by default 51
    ncols : int
        [desc], by default 51
    mapped : bool
        Don't return subbasins. By default False
    verbose : bool
        by default False

    Returns
    -------
    basins : GeoDataFrame
        Table of subbasins
    mapped : dict
        Contains mapped values and info

    """

    # Dictionary to store mapped values and info
    mapped = {"successful": False}

    # Boot up the data
    dps = ru.get_datapaths()
    da_obj = gdal.Open(dps["DEM_uda"])
    fdr_obj = gdal.Open(dps["DEM_fdr"])

    # Get the starting row,column for the delineation with MERIT
    ds_lonlat = np.array(
        [cl_gdf.geometry.values[-1].coords.xy[0][0], cl_gdf.geometry.values[-1].coords.xy[1][0],]
    )
    cr_start_mapped, map_method = mu.map_cl_pt_to_flowline(ds_lonlat, da_obj, nrows, ncols, da)

    # If mapping the point was unsuccessful, return nans
    if np.nan in cr_start_mapped:
        mapped["coords"] = (np.nan, np.nan)
        mapped["map_method"] = np.nan
        mapped["da"] = np.nan
        mapped["meridian_cross"] = np.nan
        return None, mapped
    else:
        mapped["successful"] = True
        mapped["da"] = float(
            da_obj.ReadAsArray(
                xoff=int(cr_start_mapped[0]), yoff=int(cr_start_mapped[1]), xsize=1, ysize=1,
            )[0][0]
        )
        mapped["map_method"] = map_method
        mapped["coords"] = ru.xy_to_coords(
            cr_start_mapped[0], cr_start_mapped[1], da_obj.GetGeoTransform()
        )

    # If we only want to map the point and not delineate the basin
    if map_only:
        return None, mapped

    if verbose:
        print("Delineating basin from MERIT...", end="")

    # Get all the pixels in the basin
    # cr_start_mapped = (2396, 4775)
    idcs = mu.get_basin_pixels(cr_start_mapped, da_obj, fdr_obj)

    if verbose:
        print("done.")
        print("Making basin polygon(s)...", end="")

    # Make a polygon with the indices
    polygons, split = mu.idcs_to_geopolygons(idcs, fdr_obj)
    mapped["meridian_cross"] = split

    # Attempt to ensure a valid polygon (can still return invalid ones)
    polygons = ru.validify_polygons(polygons)

    if verbose:
        print("done.")

    # Store as geodataframe
    if len(polygons) > 1:
        polygon = MultiPolygon(polygons)
    else:
        polygon = polygons[0]

    basins = gpd.GeoDataFrame(geometry=[polygon], columns=["DA"], crs=CRS.from_epsg(4326))

    # Append the drainage area of the polygon
    basins["DA"].values[0] = mapped["da"]

    return basins, mapped
