"""
rabpro (core.py)
================
Class for running rabpro commands on your data.

"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from osgeo import gdal
from pyproj import CRS

from rabpro import elev_profile as ep
from rabpro import merit_utils as mu
from rabpro import utils as ru
from rabpro import basins
from rabpro import basin_stats as bs
from rabpro import data_utils as du


class profiler:
    """The profiler class organizes data and methods for using rabpro. This is
    a parent class to the centerline and point classes which inherit profiler
    methods and attributes.

    Attributes
    ----------
    coords : tuple OR list OR str OR GeoDataFrame
        Coordinates of point(s) to delineate. A single point may be provided as
        a (lat, lon) tuple. Point(s) may also be provided via a
        shapefile or .csv file. If provided as .csv file, the columns must be
        labeled 'latitude' and 'longitude'. If a shapefile path is provided, the
        shapefile may also contain columns for widths and/or along-valley
        distances. These columns should have "width" and "distance" somewhere in
        their column names, respectively, in order for rabpro to exploit them.
    da : number
        Represents the drainage area in square kilometers of the downstream-most
        point in the provided coords.
    name : str
        Name of location/river, used for filename saving purposes only.
    verbose : bool
        If True, will print updates as processing progresses.
    update_gee_metadata : bool
        If True, will attempt to download the latest GEE Data Catalog metadata.

    Methods
    -------
    delineate_basin:
        Computes the watersheds for each lat/lon pair
    elev_profile:
        Compute the elevation profile
    basin_stats:
        Computes watershed statistics
    export:
        Exports computed data

    """

    def __init__(
        self,
        coords,
        da=None,
        name="unnamed",
        path_results=None,
        verbose=True,
        update_gee_metadata=True,
    ):

        self.name = name
        self.verbose = verbose

        # Parse the provided coordinates into a GeoDataFrame (if not already)
        if type(coords) is tuple:  # A single point was provided
            self.gdf = self._coordinates_to_gdf([coords])
        elif type(coords) is list:
            # A list of tuples was provided (centerline) # pragma: no cover
            self.gdf = self._coordinates_to_gdf(coords)
            raise DeprecationWarning(
                "elev_profile only supports single 'point' coordinate pairs, not multipoint 'centerlines'"
            )
        elif type(coords) is str:  # A path to .csv or .shp file was provided
            ext = coords.split(".")[-1]
            if ext == "csv":
                self.gdf = self._csv_to_gdf(coords)
            elif ext == "shp" or ext == "json" or ext == "geojson":
                self.gdf = gpd.read_file(coords)
        elif type(coords) is gpd.geodataframe.GeoDataFrame:
            # A GeoDataFrame was provided.
            # Convert it to EPSG:4326
            self.gdf = coords
            if self.gdf.crs.to_epsg() != 4326:
                print("Reprojecting provided coordinates to EPSG:4326.")
                self.gdf = self.gdf.to_crs(CRS.from_epsg(4326))
        else:  # pragma: no cover
            raise ValueError("Invalid coordinate input type.")

        # Determine the method for delineation
        self.da = da
        # self.method = self._which_method(force_merit)

        # Append drainage area to gdf
        if len(self.gdf) == 1:
            if self.da is not None:
                self.gdf["da_km2"] = da

        # Prepare and fetch paths for exporting results
        self.paths = ru.get_exportpaths(
            self.name, basepath=path_results, overwrite=True
        )

        # Ensure data structure exists
        self.datapaths = ru.get_datapaths(update_gee_metadata=update_gee_metadata)

        # Flags for data availability
        self.available_merit = False
        self.available_hb = False

        # Check availability MERIT data
        n_layers_geotiffs, n_vrts = du.does_merit_exist(self.datapaths)
        if n_layers_geotiffs < 4:
            print(
                (
                    "{} of 4 MERIT-Hydro layers exist. If basin delineation "
                    + "with MERIT-Hydro is desired, MERIT tiles may be downloaded "
                    + "via rabpro.data_utils.download_merit_hydro()."
                ).format(n_layers_geotiffs)
            )
        # Try to build virtual rasters if not already built
        if n_vrts < 4:
            ru.build_virtual_rasters(
                self.datapaths, skip_if_exists=True, verbose=verbose
            )
            n_layers_geotiffs, n_vrts = du.does_merit_exist(self.datapaths)

        if n_vrts == 4:
            self.available_merit = True

        # Check availability of HydroBasins data
        lev1, lev12 = du.does_hydrobasins_exist(self.datapaths)
        if lev1 + lev12 == 0:
            print(
                "No HydroBasins data was found. Use rabpro.data_utils.download_hydrobasins() to download."
            )
        else:
            self.available_hb = True

    def _coordinates_to_gdf(self, coords):
        """
        Converts a list of coordinates to a `GeoDataFrame`. Coordinates should
        be (lat, lon) pairs with EPSG==4326.
        """
        geoms = [shapely.geometry.Point((xy[1], xy[0])) for xy in coords]

        gdf = gpd.GeoDataFrame(geometry=geoms)
        gdf.crs = CRS.from_epsg(4326)

        return gdf

    def _csv_to_gdf(self, csvpath):
        """
        Creates a `GeoDataFrame` from an input path to a csv. The csv must contain
        columns named latitude and longitdue in EPSG==4326.
        """

        df = pd.read_csv(csvpath)
        df.columns = map(str.lower, df.columns)  # make all columns lowercase

        if "latitude" not in df.keys():
            raise KeyError("Latitude value not provided in .csv file.")
        if "longitude" not in df.keys():
            raise KeyError("Latitude value not provided in .csv file.")

        lats = df.latitude.values
        lons = df.longitude.values
        geoms = [shapely.geometry.Point((lon, lat)) for lat, lon in zip(lats, lons)]

        gdf = gpd.GeoDataFrame(geometry=geoms)
        gdf.crs = CRS.from_epsg(4326)

        return gdf

    def delineate_basin(
        self,
        search_radius=None,
        map_only=False,
        force_merit=False,
        force_hydrobasins=False,
    ):
        """Computes the watersheds for each lat/lon pair and adds their
        drainage areas to the self.gdf `GeoDataFrame`.

        There are two methods available for delineating basins: HydroBASINS and
        MERIT. HydroBASINS is appropriate for large basins (1000 km^2 and
        larger), and MERIT can provide more detailed basin delineations for
        smaller basins. The method used depends on the size of the basin, which
        is interpreted via the provided drainage area. If no drainage area was
        provided, HydroBASINS will be used. Otherwise, if the provided drainage
        area is less than 1000 km^2, MERIT will be used. MERIT may also be forced
        for larger basins using the 'force_merit' argument when instantiating
        the profiler.

        Parameters
        ----------
        search_radius : numeric, optional
            in meters, by default None
        map_only : bool, optional
            If we only want to map the point and not delineate the basin, by default False
        force_merit : bool, optional
            Forces the use of MERIT to delineate basins.
        force_hydrobasins : bool, optional
            Forces the use of HydroBASINS to delineate basins.
        """

        # Determine method
        if self.da is None or force_hydrobasins is True:
            self.method = "hydrobasins"
            if self.da is None:
                print(
                    "Warning: no drainage area was provided. HydroBASINS will be used to delineate the basin, but result should be visually verified and coordinate updated if results are not as expected."
                )
        elif self.da <= 1000 or force_merit is True:
            self.method = "merit"
        else:
            self.method = "hydrobasins"

        if map_only is True:
            self.method = "merit"

        if self.method == "merit" and self.available_merit is False:
            print("MERIT data are not available; no delineation can be done.")
        elif self.method == "hydrobasins" and self.available_hb is False:
            print("HydroBasins data are not available; no delineation can be done.")

        if self.verbose is True and map_only is False:
            print("Delineating watershed using {}.".format(self.method))

        if self.method == "hydrobasins":
            self.watershed, self.mapped = basins.main_hb(self.gdf, self.verbose)

        elif self.method == "merit":
            if search_radius is not None:
                dps = ru.get_datapaths()
                ds_lonlat = np.array(
                    [
                        self.gdf.geometry.values[-1].coords.xy[0][0],
                        self.gdf.geometry.values[-1].coords.xy[1][0],
                    ]
                )
                self.nrows, self.ncols = mu._nrows_and_cols_from_search_radius(
                    ds_lonlat[0],
                    ds_lonlat[1],
                    search_radius,
                    gdal.Open(dps["DEM_uda"]).GetGeoTransform(),
                )
            else:
                self.nrows, self.ncols = 50, 50

            self.watershed, self.mapped = basins.main_merit(
                self.gdf,
                self.da,
                nrows=self.nrows,
                ncols=self.ncols,
                map_only=map_only,
                verbose=self.verbose,
            )

            # Ensure the provided coordinate was mappable
            if self.watershed is None:
                if not map_only:  # pragma: no cover
                    print(
                        "Could not find a suitable flowline to map given coordinate and DA. No basin can be delineated.",
                        "You can set da=None to force an attempt with HydroBASINS.",
                    )
            else:
                self.gdf["da_km2"] = [None for p in range(self.gdf.shape[0])]
                self.gdf["da_km2"].values[0] = self.watershed["da_km2"][0]

                # Ensure the MERIT-delineated polygon's area is within 10% of the mapped value
                # If the basin crosses the -180/180 meridian, need to use a projection that splits elsewhere
                # 2193 for new zealand, seems to work fine for areas though
                rp_epsg = 2193 if self.mapped["meridian_cross"] else 3410

                reproj_ea_meters = self.watershed.to_crs(crs=CRS.from_epsg(rp_epsg))
                pgon_area = (
                    reproj_ea_meters.geometry.values[0].area / 10**6
                )  # square km
                pct_diff = (
                    abs(pgon_area - self.mapped["da_km2"]) / self.mapped["da_km2"] * 100
                )
                if pct_diff > 10:  # pragma: no cover
                    print(
                        f"Check delineated basin. There is a difference of {pct_diff}% between MERIT DA and polygon area."
                    )

    def elev_profile(self, dist_to_walk_km=None):
        """
        Compute the elevation profile. The profile is computed such that the
        provided coordinate is the centerpoint (check if this is true).

        Parameters
        ----------
        dist_to_walk_km : numeric
            The distance to trace the elevation profile from the provided
            point. This distance applies to upstream and downstream--i.e. the
            total profile distance is twice this value. If not specified,
            will be automatically computed as 10 channel widths from provided
            DA value if not specified OR 5 km, whichever is larger.

        """
        if not hasattr(self, "nrows"):
            self.nrows = 50
            self.ncols = 50

        if dist_to_walk_km is None:
            # Check if the watershed has been delineated; pull drainage area
            # from that
            if "da_km2" in self.gdf.keys():
                da = self.gdf["da_km2"].values[0]
            elif hasattr(self, "watershed"):
                da = self.watershed["da_km2"].values[0]
            else:
                raise KeyError(
                    "If the dist_to_walk_km parameter is not specified, a drainage area must be provided when instantiating the profiler."
                )

            dist_to_walk_km = ru.dist_from_da(da)
            dist_to_walk_km = max(dist_to_walk_km, 5)

        self.gdf, self.flowline = ep.main(
            self.gdf, dist_to_walk_km, self.verbose, self.nrows, self.ncols
        )

    def basin_stats(self, datasets, reducer_funcs=None, folder=None, test=False):
        """
        Computes watershed statistics.

        Parameters
        ----------
        datasets : list of Dataset
            Datasets to compute stats over. See the subbasin_stats.Dataset class
        reducer_funcs : list of functions
            List of functions to apply to each feature over each dataset. Each
            function should take in an ee.Feature() object. For example, this is
            how the function and header are applied on a feature:
            feature.set(f.__name__, function(feature))
        folder : str
            Google Drive folder to store results in
        """

        return bs.compute(
            datasets,
            basins_gdf=self.watershed,
            reducer_funcs=reducer_funcs,
            folder=folder,
            verbose=self.verbose,
            test=test,
        )

    def export(self, what="all"):
        """
        Exports data computed by rapbro.

        Parameters
        ----------
        what : list, optional
            Which data should be exported? Choose from
            'all' - all computed data
            'flowline' - flowline json is exported with elevation and distance attributes
            'watershed' - watershed polygon
            The default is 'all'.

        """
        if what == "all":
            what = ["flowline", "watershed"]

        if type(what) is str:
            what = [what]

        for w in what:
            if w not in ["flowline", "watershed"]:
                raise KeyError(
                    f"Requested export {w} not available. Choose from {['flowline', 'subbasins']}."
                )
            elif w == "watershed":
                if hasattr(self, "watershed"):
                    self.watershed.to_file(self.paths["watershed"], driver="GeoJSON")
                    if self.verbose:
                        print(
                            "Watershed written to {}.".format(self.paths["watershed"])
                        )
                else:
                    print("No basins found for export.")
            elif w == "flowline":
                if hasattr(self, "flowline"):
                    self.flowline.to_file(self.paths["flowline"], driver="GeoJSON")
                    if self.verbose:
                        print("Flowline written to {}.".format(self.paths["watershed"]))
                else:
                    print("No flowline found for export.")
