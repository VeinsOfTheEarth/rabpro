"""
Subbasin delineation (subbasins.py)
===================================

Functions to calculate subbasin geometries.
"""

import numpy as np
from osgeo import gdal
from pyproj import CRS
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, MultiPolygon

from rabpro import utils as ru
from rabpro import merit_utils as mu


def main_merit(gdf, da, nrows=51, ncols=51, map_only=False, verbose=False):
    """Calculates subbasins using MERIT

    Parameters
    ----------
    gdf : GeoDataFrame
        Centerline coordinates
    da : numeric
        Drainage area in km^2
    nrows : int
        by default 51
    ncols : int
        by default 51
    map_only : bool
        If True, will map the coordinate to a MERIT flowline but not delineate 
        the basin. The default is False.
    verbose : bool
        by default False

    Returns
    -------
    basins : GeoDataFrame
        The delineated watershed
    mapped : dict
        Information about the mapping quality.

    """

    # Dictionary to store mapped values and info
    mapped = {"successful": False}

    # Boot up the data
    dps = ru.get_datapaths(rebuild_vrts=False)
    da_obj = gdal.Open(dps["DEM_uda"])
    fdr_obj = gdal.Open(dps["DEM_fdr"])

    # Get the starting row,column for the delineation with MERIT
    ds_lonlat = np.array(
        [
            gdf.geometry.values[-1].coords.xy[0][0],
            gdf.geometry.values[-1].coords.xy[1][0],
        ]
    )
    cr_start_mapped, map_method = mu.map_cl_pt_to_flowline(
        ds_lonlat, da_obj, nrows, ncols, da
    )

    # If mapping the point was unsuccessful, return nans
    if np.nan in cr_start_mapped:
        mapped["coords"] = (np.nan, np.nan)
        mapped["map_method"] = np.nan
        mapped["da_km2"] = np.nan
        mapped["meridian_cross"] = np.nan
        mapped["da_pct_dif"] = np.nan
        return None, mapped
    else:
        mapped["successful"] = True
        mapped["da_km2"] = float(
            da_obj.ReadAsArray(
                xoff=int(cr_start_mapped[0]),
                yoff=int(cr_start_mapped[1]),
                xsize=1,
                ysize=1,
            )[0][0]
        )
        mapped["da_pct_dif"] = (
            np.abs(mapped["da_km2"] - gdf["da_km2"].values[0])
            / gdf["da_km2"].values[0]
            * 100
        )
        mapped["map_method"] = map_method
        mapped["coords"] = ru.xy_to_coords(
            cr_start_mapped[0], cr_start_mapped[1], da_obj.GetGeoTransform()
        )

    # If we only want to map the point and not delineate the basin
    if map_only:
        return None, mapped

    # Get all the pixels in the basin
    # cr_start_mapped = (2396, 4775)
    idcs = mu._get_basin_pixels(cr_start_mapped, da_obj, fdr_obj)

    if verbose:
        print("Making basin polygon from pixels...", end="")

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

    basins = gpd.GeoDataFrame(
        geometry=[polygon], columns=["da_km2"], crs=CRS.from_epsg(4326)
    )

    # Append the drainage area of the polygon
    basins["da_km2"].values[0] = mapped["da_km2"]

    return basins, mapped


def main_hb(gdf, verbose=False):
    """Calculates subbasins using Hydrobasins

    Parameters
    ----------
    gdf : GeoDataFrame
        Contains geometry and DA (drainage area) columns.
    verbose : bool, optional
        By default False

    Returns
    -------
    subbasins_gdf : GeoDataFrame
        Contains subbasin geometries
    sb_inc_gdf : GeoDataFrame
        Contains the polygons of the incremental catchments. The upstream-most
        basin will be the largest polygon in most cases, but that depends on the
        input centerline.
    cl_das : numpy.ndarray
        Drainage areas

    Raises
    ------
    RuntimeWarning
        If gdf has no CRS defined

    Examples
    --------
    .. code-block:: python

        import rabpro
        coords = (56.22659, -130.87974)
        rpo = rabpro.profiler(coords, name='basic_test')
        test = rabpro.subbasins.main_hb(rpo.gdf)
    """
    datapaths = ru.get_datapaths(rebuild_vrts=False)
    mapped = {}
    mapped["successful"] = False

    # Convert the gdf to EPSG:4326 if necessary in order to align with HydroSheds
    was_transformed = False
    if gdf.crs is None:
        raise RuntimeWarning("Centerline geodataframe has no defined CRS.")
    elif gdf.crs.to_authority()[1] != "4326":
        orig_crs = gdf.crs  # Save the original crs to put it back later
        gdf = gdf.to_crs(epsg=4326)
        was_transformed = True

    # Load the appropriate HydroBasins shapefile as a geodataframe
    HB_gdf = load_continent_basins(
        gdf, datapaths["HydroBasins1"], datapaths["HydroBasins12"]
    )

    if HB_gdf is None:
        return None, mapped

    # Find the most-downstream HydroBasins polygon
    HB_start = _map_to_HB_basin(gdf, HB_gdf)

    # Get all upstream HB polygons (includes HB_start)
    HB_upstream  = _upstream_HB_basins(HB_start['HYBAS_ID'], HB_gdf)

    # Union all HB basins
    basin_pgon = ru.union_gdf_polygons(HB_upstream, range(0, len(HB_upstream)))
    basin_da = sum(ru.area_4326(basin_pgon))

    mapped['successful'] = True
    mapped['da_km2'] = np.sum(HB_upstream['SUB_AREA'].values) + HB_start['SUB_AREA']
    mapped['HYBAS_ID'] = HB_start['HYBAS_ID']
    if 'da_km2' in gdf.keys():
        mapped['da_pct_dif'] = np.abs(mapped['da_km2'] - gdf['da_km2'].values[0]) / gdf['da_km2'].values[0] * 100
    else:
        mapped['da_pct_dif'] = np.nan

    # Export upstream basin gdf
    basins = gpd.GeoDataFrame(
        geometry=[basin_pgon], data={"da_km2": [basin_da]}, crs=CRS.from_epsg(4326)
    )

    # Transform back to original CRS if necessary
    if was_transformed is True:
        basins = basins.to_crs(orig_crs)

    return basins, mapped


def _upstream_HB_basins(hybas_id, HB_gdf):
    """
    Finds all the HB basins draining into the one specified by hybas_id.
    """
    visited_subbasins = set([hybas_id])
    to_visit = set([hybas_id])
    while to_visit:
        this_basin = to_visit.pop()
        to_visit.update(
            set(HB_gdf[HB_gdf["NEXT_DOWN"] == this_basin]["HYBAS_ID"].values.tolist())
        )
        visited_subbasins.update(to_visit)
    HB_upstream = HB_gdf[HB_gdf["HYBAS_ID"].isin(visited_subbasins)]

    return HB_upstream


def _map_to_HB_basin(gdf, HB_gdf):
    """
    Maps a coordinate in gdf to the nearest HB_gdf while also considering 
    drainage area. If no drainage area is available, simply finds the 
    HB polygon wherin the coordinate lies.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geometry data. Should be in EPSG:4326.
    HB_gdf : GeoDataFrame
        Contains the level-12 polygons for the appropriate region containing
        the coordinate in gdf.

    Returns
    -------
    HB_within : GeoDataFrame
        A one-row GeoDataFrame containing the HB polygon that most likely 
        contains the coordinate + DA given in gdf.

    """

    def which_HB_within(gdf, HB_gdf):
        """
        Returns the HB row containing the coordinate in gdf. Looping seems
        to be faster than GeoPandas builtins.
        """
        HB_within = None
        for i, geom in enumerate(HB_gdf.geometry):
            if geom.contains(gdf.geometry.values[0]):
                HB_within = HB_gdf.iloc[i]
                break

        return HB_within

    # No drainage area provided, just use whichever HydroBasins polygon the
    # point falls within
    if "DA" not in gdf.keys():
        HB_within = which_HB_within(gdf, HB_gdf)
        return HB_within

    # Else look for any HB polygons within a 5 km buffer around the point
    gdf_b = gdf.copy()
    gdf_b = gdf_b.to_crs(CRS.from_epsg(3857))
    gdf_b["geometry"] = gdf_b.buffer(5000)
    gdf_b = gdf_b.to_crs(CRS.from_epsg(4326))
    HB_possibles = gpd.sjoin(HB_gdf, gdf_b, op="intersects")

    # Select the HB polygon with the smallest da_dif_pct nearby
    HB_possibles["da_dif"] = HB_possibles["UP_AREA"].values - gdf["da_km2"].values[0]
    HB_possibles["da_dif_pct"] = (
        np.abs(HB_possibles["da_dif"] / gdf["da_km2"].values[0]) * 100
    )
    HB_possibles = HB_possibles.sort_values(by="da_dif_pct", ascending=True)

    # The smallest drainage area difference basin is the one we choose
    return HB_possibles.iloc[0]


def load_continent_basins(gdf, level_one, level_twelve):
    """Load a HydroBasins continent polygon

    Parameters
    ----------
    gdf : GeoDataFrame
        Geometry data. Should be in EPSG:4326.
    level_one : str
        Path to level 1 HydroBasins data
    level_twelve : str
        Path to level 12 HydroBasins data

    Returns
    -------
    GeoDataFrame
        HydroBasins

    """

    # Prepare load level 1 dataframe
    level_one_path = str(Path(level_one) / "hybas_all_lev01_v1c.shp")
    level_one_df = gpd.read_file(level_one_path)

    # Find the first point of the centerline to figure out which continent we're in
    xy_cl = gdf.geometry.values[0].coords.xy
    cl_us_pt = gpd.GeoDataFrame(geometry=[Point(xy_cl[0][0], xy_cl[1][0])])
    cl_us_pt.crs = gdf.crs

    # Intersect with level-1 HydroBasins to figure out which continent we're in
    clpt_level_onei = gpd.sjoin(cl_us_pt, level_one_df, op="intersects")
    if len(clpt_level_onei) == 0:
        print(
            """Provided coordinate ({}) does not lie within HydroBasins polygons.
            Check that lat/lon are not reversed in input. Exiting.""".format(
                [xy_cl[0][0], xy_cl[1][0]]
            )
        )
        return None

    id_no = clpt_level_onei.PFAF_ID[0]

    # Load the appropriate level 12 dataframe
    loadnames = ["af", "eu", "si", "as", "au", "sa", "na", "ar", "gr"]
    level_twelve_path = str(
        Path(level_twelve) / str("hybas_" + loadnames[id_no - 1] + "_lev12_v1c.shp")
    )

    # Load the appropriate level-12 Hydrobasins continent shapefile
    HB_gdf = gpd.read_file(level_twelve_path)

    return HB_gdf


""" Code Graveyard -- these are functions needed when rabpro considered 
centerlines instead of single points """

# def find_contributing_basins(chainids, HB_gdf):
#     """
#     Given an input GeoDataFrame of HydroBasins shapefiles and a list of chainids
#     denoting which basins are part of the chain, this function walks upstream
#     from the upstream-most basin by following the "NEXT_DOWN" attribute until
#     all possible basins are included. This process is repeated, but stops when
#     the previous basin is encountered. The result is a list of sets, where each
#     set contains the INCREMENTAL basin indices for each subbasin. i.e. the
#     most-downstream subbasin would be found by unioning all the sets.

#     IMPORTANT: chainids must be arranged in US->DS direction.

#     Parameters
#     ----------
#     chainids : list
#         Denotes which basins are part of the chain
#     HB_gdf : GeoDataFrame
#         HydroBasins shapefiles

#     Returns
#     -------
#     list of sets
#         each set contains the incremental basin indices for each subbasin
#     """
#     subbasin_idcs = []
#     visited_subbasins = set()

#     for idx in chainids:
#         sb_idcs = set([idx])
#         sb_check = set(HB_gdf[HB_gdf.NEXT_DOWN == HB_gdf.HYBAS_ID[idx]].index)
#         while sb_check:
#             idx_check = sb_check.pop()

#             if idx_check in visited_subbasins:
#                 continue

#             sb_idcs.add(idx_check)

#             basin_id_check = HB_gdf.HYBAS_ID[idx_check]
#             sb_check = (
#                 sb_check
#                 | set(HB_gdf[HB_gdf.NEXT_DOWN == basin_id_check].index)
#                 - visited_subbasins
#             )

#         # Store the incremental indices
#         subbasin_idcs.append(sb_idcs)

#         # Update the visited subbasins (so we don't redo them)
#         visited_subbasins = visited_subbasins | sb_idcs

#     return subbasin_idcs


# def _delineate_subbasins(idxmap, HB_gdf):
#     """Finds all the upstream contributing basins for each basin in idxmap.
#     This could perhaps be optimized, but the current implementation just solves
#     each basin in idxmap independently.

#     Parameters
#     ----------
#     idxmap : list

#     HB_gdf : GeoDataFrame

#     Returns
#     -------
#     subHB_gdf : GeoDataFrame
#         Contains the polygons of each basin's catchment
#     inc_df : GeoDataFrame
#         Contains the polygons of the incremental catchments. The upstream-most
#         basin will be the largest polygon in most cases, but that depends on the
#         input centerline.
#     """

#     # idxmap contains only the polygons (indices) in the chain that contain
#     # centerline points, arranged in us->ds direction. Use it to determine
#     # which basins to delineate
#     chainids = [
#         x for i, x in enumerate(idxmap) if x not in idxmap[0:i]
#     ]  # unique-ify list - don't wanna lose order
#     chainids = np.ndarray.tolist(
#         np.array(chainids, dtype=int)
#     )  # convert to native int from numpy int

#     # Get (incremental) indices of all subbasins
#     subbasin_idcs = find_contributing_basins(chainids, HB_gdf)

#     # Make polygons of the incremental subbasins
#     inc_df = gpd.GeoDataFrame(
#         index=range(0, len(subbasin_idcs)),
#         columns=["geometry", "areas"],
#         crs=HB_gdf.crs,
#     )
#     subHB_gdf = gpd.GeoDataFrame(
#         index=range(0, len(subbasin_idcs)),
#         columns=["geometry", "areas"],
#         crs=HB_gdf.crs,
#     )

#     for i, si in enumerate(subbasin_idcs):
#         # Incremental subbasins
#         inc_df.geometry.values[i] = ru.union_gdf_polygons(HB_gdf, si, buffer=True)
#         subHB_gdf.areas.values[i] = np.max(HB_gdf.iloc[list(si)].UP_AREA.values)

#     # Combine the incremental subbasins to get the polygons of entire subbasins
#     # for each centerline point; buffer and un-buffer the polygons to account for
#     # "slivers"
#     for i in range(len(inc_df)):
#         if i == 0:
#             subHB_gdf.geometry.values[i] = inc_df.geometry.values[i]
#             inc_df.areas.values[i] = subHB_gdf.areas.values[i]
#         #            inc_df.loc[i].areas = subHB_gdf.loc[i].areas
#         else:
#             # Put geometries into dataframe
#             #            temp_gdf = gpd.GeoDataFrame(index=range(0,2), columns=['geometry'], crs=HB_gdf.crs)
#             #            temp_gdf.geometry = [inc_df.loc[i].geometry, subHB_gdf.loc[i-1].geometry]
#             #            subHB_gdf.loc[i].geometry = ru.union_gdf_polygons(temp_gdf, range(0, 2))
#             #            inc_df.loc[i].areas = subHB_gdf.loc[i].areas - subHB_gdf.loc[i-1].areas
#             temp_gdf = gpd.GeoDataFrame(
#                 index=range(0, 2), columns=["geometry"], crs=HB_gdf.crs
#             )
#             temp_gdf.geometry = [
#                 inc_df.geometry.values[i],
#                 subHB_gdf.geometry.values[i - 1],
#             ]
#             subHB_gdf.geometry.values[i] = ru.union_gdf_polygons(temp_gdf, range(0, 2))
#             inc_df.areas.values[i] = (
#                 subHB_gdf.areas.values[i] - subHB_gdf.areas.values[i - 1]
#             )

#     return subHB_gdf, inc_df


# def _initial_basin_chain(HB_gdf, cl_gdf, buf_wid=0.1):
#     """
#     Finds the chain of drainage basins from the upstream-most centerline point
#     to the sink (e.g. ocean).

#     Parameters
#     ----------
#     HB_gdf : GeoDataFrame

#     cl_gdf : GeoDataFrame
#         Centerline coordinates
#     buf_wid : float, optional
#         by default 0.1

#     Returns
#     -------
#     list

#     Raises
#     ------
#     RuntimeError
#         If it cannot find a chain of basins that includes more than half the
#         input centerline points
#     """

#     def get_chain(HB_gdf, basin_id_start):
#         """
#         Given an input geodataframe called HB_gdf that is created from a
#         HydroBasins shapefile, this function returns all the downstream basins
#         of an input basin_id_start corresponding to a HYBAS_ID
#         """
#         chain = [basin_id_start]
#         while 1:
#             next_basin_id = HB_gdf[HB_gdf.HYBAS_ID.values == chain[-1]].index[0]
#             chain.append(HB_gdf.NEXT_DOWN[next_basin_id])
#             if chain[-1] == 0:
#                 chain.pop()
#                 break

#         return chain

#     def frac_pts_within_chain(HB_gdf, chain, cl_gdf, buf_wid=0.1):
#         """
#         Given an input chain created by get_chain and a geodataframe containing
#         centerline coordinates (or any coordinates), this function returns the
#         fraction of the input coordinates that are within the chain. The chain
#         is first buffered by buf_wid, which is in units of the chain's native
#         projection (i.e. WGS84, no projection -> units = degrees).
#         """

#         chainids = [HB_gdf.index[HB_gdf.HYBAS_ID == c].values[0] for c in chain]
#         chainHB_gdf = HB_gdf.loc[chainids]

#         # Combine the chain basins into single polygon
#         chainHB_gdf = chainHB_gdf.dissolve(by="MAIN_BAS")

#         # Buffer the chain polygon
#         chainHB_gdf["geometry"] = chainHB_gdf["geometry"].buffer(buf_wid)

#         # Intersect chain polygon with centerline points
#         chaincl_HB_gdf = gpd.sjoin(cl_gdf, chainHB_gdf, predicate="intersects")

#         ## Output chain shapefile for visualzation in GIS
#         #        chainHB_gdf.to_file(r"X:\temp" + 'test_chain.shp')

#         # Fraction of points within the buffered chain
#         frac_pts_within = len(chaincl_HB_gdf) / len(cl_gdf)

#         return frac_pts_within

#     # Get intersection of centerline points and basins
#     basin_intersect = gpd.sjoin(cl_gdf, HB_gdf, predicate="intersects")

#     # Find index of basin that upstream-most point is in
#     basin_id_start = basin_intersect.HYBAS_ID.values[0]

#     # Get the set of IDs comprising the chain
#     chain = get_chain(HB_gdf, basin_id_start)

#     # Get the fraction of centerline points within the chain
#     frac_pts_within = frac_pts_within_chain(HB_gdf, chain, cl_gdf, buf_wid)

#     # Check if the centerline coordinates are mostly within the chain   -
#     # if not, loop through the neighboring basins and see if any of them
#     # serve as appropriate chain basins
#     if frac_pts_within < 0.5:
#         print("Initial basin guess was not correct...trying neighboring basins.")

#         # Get the polygon IDs that border the initially-guessed basin
#         initial_pgon = HB_gdf[HB_gdf.HYBAS_ID == basin_id_start]

#         # Find its neighbors via intersection
#         neighbors = gpd.sjoin(initial_pgon, HB_gdf, predicate="intersects")
#         neigh_idcs = set(neighbors.HYBAS_ID_right.values)

#         # Remove neighbors we've already looked at
#         neigh_idcs = neigh_idcs - set(chain)

#         while frac_pts_within < 0.5:
#             basin_id_start = neigh_idcs.pop()
#             chain = get_chain(HB_gdf, basin_id_start)
#             frac_pts_within = frac_pts_within_chain(HB_gdf, chain, cl_gdf, buf_wid)
#             neigh_idcs = neigh_idcs - set(chain)

#             if len(neigh_idcs) == 0 and frac_pts_within < 0.5:
#                 raise RuntimeError(
#                     "Could not find a chain of basins that includes more than half the input centerline points."
#                 )
#         print("Found an initial basin whose chain contains > 50% of centerline points.")

#     chainids = [HB_gdf.index[HB_gdf.HYBAS_ID == i].values[0] for i in chain]

#     return chainids


# def _map_points_to_chain(HB_gdf, cl_gdf, chainids):
#     """
#     Maps all centerline points to level-12 drainage basins within chain.
#     1. Identify the polygon each point falls into.
#     2. Push the points that don't fall into a polygon into the nearest
#     (Euclidean distance) one.
#     3. Check that drainage area does not decrease as we move downstream.
#     If so, assign point to previous (next upstream) basin.

#     Parameters
#     ----------
#     HB_gdf : GeoDataFrame

#     cl_gdf : GeoDataFrame
#         Centerline coordinates
#     chainids :

#     Returns
#     -------
#     idxap : list
#         Maps each centerline point to a basin index within the HB_gdf
#         GeoDataFrame
#     DA : int
#         Drainage areas for each centerline point
#     """

#     # Re-intersect the individual basin polygons with the centerline points
#     chainHB_gdf = HB_gdf.loc[chainids]
#     cl_in_basins = gpd.sjoin(cl_gdf, chainHB_gdf, predicate="intersects")

#     # In case a centerline point is EXACTLY on the border of two basins, push
#     # it into the upstream one
#     u_idx, u_ct = np.unique(cl_in_basins.index.values, return_counts=True)
#     idx_in_two_basins = np.where(u_ct > 1)[0]
#     remrow = np.array([], dtype=int)
#     for i in idx_in_two_basins:
#         idx = u_idx[i]
#         rowidcs = np.where(idx == cl_in_basins.index.values)[0]
#         # Find upstream-most basin
#         idxkeep = np.argmax(cl_in_basins.iloc[rowidcs].DIST_MAIN.values)
#         rowidcs = np.delete(rowidcs, idxkeep)
#         remrow = np.concatenate((remrow, rowidcs))
#     # Now drop the rows that represent a second-intersection
#     cl_in_basins.drop(cl_in_basins.index[remrow], inplace=True)

#     # Extract Drainage Areas from polygons, put Nones where the point falls outside of any polygons
#     # Also create a map between each point and its corresponding polygon index in
#     # the original dataframe
#     idxmap = []
#     DA = []
#     for i in range(len(cl_gdf)):
#         try:
#             DA.append(cl_in_basins.loc[i].UP_AREA.values)
#             idxmap.append(cl_in_basins.loc[i].index_right.values)
#         except:
#             DA.append(None)
#             idxmap.append(None)

#     # Assign the cl points that are not in a chain basin the DA of the nearest
#     # chain basin
#     # Which points need to be fixed?
#     fixid = [i for i, j in enumerate(idxmap) if j is None]
#     # Fix 'em
#     for fid in fixid:
#         clgeom = cl_gdf.loc[fid].geometry

#         # Compute distance between point and all polygons in chain [CAN REDUCE SEARCH DOMAIN HERE FOR FASTER PROCESSING]
#         dists = []
#         for i in chainHB_gdf.index:
#             dists.append(clgeom.distance(chainHB_gdf.geometry[i]))
#         minidx = dists.index(min(dists))
#         # Assign DA of nearest chain polygon
#         idxmap[fid] = chainHB_gdf.index.values[minidx]
#         DA[fid] = chainHB_gdf.UP_AREA.values[minidx]

#     # Now that DAs have been assigned for all points, find those that cannot be
#     # correct (i.e. those that decrease with downstream distance) and fix them
#     for i in range(1, len(DA)):
#         if DA[i] < DA[i - 1]:
#             DA[i] = DA[i - 1]
#             idxmap[i] = idxmap[i - 1]

#     return idxmap, DA

