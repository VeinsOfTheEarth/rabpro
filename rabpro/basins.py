"""
Basin delineation (basins.py)
===================================

Functions to calculate basin geometries.
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
    """Calculates basins using MERIT

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
    dps = ru.get_datapaths()
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
    """Calculates basins using HydroBASINS

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
        test = rabpro.basins.main_hb(rpo.gdf)
    """
    datapaths = ru.get_datapaths()
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
    HB_upstream = _upstream_HB_basins(HB_start["HYBAS_ID"], HB_gdf)

    # Union all HB basins
    basin_pgon = ru.union_gdf_polygons(HB_upstream, range(0, len(HB_upstream)))
    basin_da = sum(ru.area_4326(basin_pgon))

    mapped["successful"] = True
    mapped["da_km2"] = np.sum(HB_upstream["SUB_AREA"].values) + HB_start["SUB_AREA"]
    mapped["HYBAS_ID"] = HB_start["HYBAS_ID"]
    if "da_km2" in gdf.keys():
        mapped["da_pct_dif"] = (
            np.abs(mapped["da_km2"] - gdf["da_km2"].values[0])
            / gdf["da_km2"].values[0]
            * 100
        )
    else:
        mapped["da_pct_dif"] = np.nan

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
