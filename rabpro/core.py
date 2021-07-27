"""
RaBPro (core.py)
================
Class for running RaBPro commands on your data.

"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from osgeo import gdal
from pyproj import CRS

from rabpro import elev_profile as ep
from rabpro import merit_utils as mu
from rabpro import subbasins as sb
from rabpro import subbasin_stats as ss
from rabpro import utils as rpu


class profiler:
    """ The profiler class organizes data and methods for using RaBPro. This is
    a parent class to the centerline and point classes which inherit profiler
    methods and attributes. You should use `rabpro.profiler` rather than
    `rabpro.core.profiler`.

    Attributes
    ----------
    coords : tuple OR list OR str OR GeoDataFrame
        Coordinates of point(s) to delineate. A single point may be provided as
        a (lat, lon) tuple. If a river centerline is being provided, its points
        must be arranged US->DS. River centerline coordinates may be provided as
        a list of (lat, lon) pairs. Point(s) may also be provided via a
        shapefile or .csv file. If provided as .csv file, the columns must be
        labeled 'latitude' and 'longitude'. If a shapefile path is provided, the
        shapefile may also contain columns for widths and/or along-valley
        distances. These columns should have "width" and "distance" somewhere in
        their column names, respectively, in order for RaBPro to exploit them.
    da : number
        Represents the drainage area in square kilometers of the downstream-most
        point in the provided coords.
    name : str
        Name of location/river, used for filename saving purposes only.
    verbose : bool
        If True, will print updates as processing progresses.

    """

    def __init__(
        self, coords, da=None, name="unnamed", path_results=None, force_merit=False, verbose=True,
    ):

        self.name = name
        self.verbose = verbose

        # Parse the provided coordinates into a GeoDataFrame (if not already)
        if type(coords) is tuple:  # A single point was provided
            self.gdf = self._coordinates_to_gdf([coords])
        elif type(coords) is list:  # A list of tuples was provided (centerline)
            self.gdf = self._coordinates_to_gdf(coords)
        elif type(coords) is str:  # A path to .csv or .shp file was provided
            ext = coords.split(".")[-1]
            if ext == "csv":
                self.gdf = self._csv_to_gdf(coords)
            elif ext == "shp" or ext == "json" or ext == "geojson":
                self.gdf = gpd.read_file(coords)
        elif type(coords) is gpd.geodataframe.GeoDataFrame:  # A GeoDataFrame was provided.
            # Convert it to EPSG:4326
            self.gdf = coords
            if self.gdf.crs.to_epsg() != 4326:
                print("Reprojecting provided coordinates to EPSG:4326.")
                self.gdf = self.gdf.to_crs(CRS.from_epsg(4326))
        else:
            raise ValueError("Invalid coordinate input type.")

        # Determine the method for delineation
        self.da = da
        self.method = self._which_method(force_merit)

        # Append drainage area to gdf
        if len(self.gdf) == 1:
            if self.da is not None:
                self.gdf["DA"] = da

        # Prepare and fetch paths for exporting results
        self.paths = rpu.get_exportpaths(self.name, basepath=path_results, overwrite=True)

        # This line will ensure that all the virtual rasters are built and available.
        rpu.get_datapaths()

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

    def _which_method(self, force_merit, merit_thresh=500):
        """ Returns the method to use for delineating watersheds.
        """

        method = "hydrobasins"
        if force_merit or (self.da is not None and self.da < merit_thresh):
            method = "merit"

        return method

    def delineate_basins(self, search_radius=None, map_only=False):
        """ Computes the watersheds for each lat/lon pair and adds their drainage
        areas to the self.gdf `GeoDataFrame`.

        There are two methods used for delineating basins: HydroBASINS and
        MERIT. HydroBASINS is appropriate for large basins (500 km^2 and
        larger), and MERIT can provide more detailed basin delineations for
        smaller basins. The method used depends on the size of the basin, which
        is interpreted via the provided drainage area. If no drainage area was
        provided, HydroBASINS will be used. Otherwise, if the provided drainage
        area is less than 500 km^2, MERIT will be used. MERIT may also be forced
        for larger basins using the 'force_merit' argument when instantiating
        the profiler.

        Parameters
        ----------
        search_radius : numeric, optional
            in meters, by default None
        map_only : bool, optional
            [description], by default False
        """

        if self.method == "hydrobasins":
            self.basins, self.basins_inc, cl_das = sb.main_hb(self.gdf, self.verbose)
            self.gdf["DA"] = cl_das

        elif self.method == "merit":
            if search_radius is not None:
                dps = rpu.get_datapaths()
                ds_lonlat = np.array(
                    [
                        self.gdf.geometry.values[-1].coords.xy[0][0],
                        self.gdf.geometry.values[-1].coords.xy[1][0],
                    ]
                )
                self.nrows, self.ncols = mu.nrows_and_cols_from_search_radius(
                    ds_lonlat[0],
                    ds_lonlat[1],
                    search_radius,
                    gdal.Open(dps["DEM_uda"]).GetGeoTransform(),
                )
            else:
                self.nrows, self.ncols = 50, 50

            self.basins, self.mapped = sb.main_merit(
                self.gdf,
                self.da,
                nrows=self.nrows,
                ncols=self.ncols,
                map_only=map_only,
                verbose=self.verbose,
            )

            # Ensure the provided coordinate was mappable
            if self.basins is None:
                if not map_only:
                    print(
                        "Could not find a suitable flowline to map given coordinate and DA. No basin can be delineated."
                    )
            else:
                self.gdf["DA"] = [None for p in range(self.gdf.shape[0])]
                self.gdf["DA"].values[0] = self.basins["DA"][0]

                # Ensure the MERIT-delineated polygon's area is within 10% of the mapped value
                # If the basin crosses the -180/180 meridian, need to use a projection that splits elsewhere
                # 2193 for new zealand, seems to work fine for areas though
                rp_epsg = 2193 if self.mapped["meridian_cross"] else 3410

                reproj_ea_meters = self.basins.to_crs(crs=CRS.from_epsg(rp_epsg))
                pgon_area = reproj_ea_meters.geometry.values[0].area / 10 ** 6  # square km
                pct_diff = abs(pgon_area - self.mapped["da"]) / self.mapped["da"] * 100
                if pct_diff > 10:
                    print(
                        f"Check delineated basin. There is a difference of {pct_diff}% between MERIT DA and polygon area."
                    )

    def elev_profile(self):
        """
        Compute the elevation profile.
        """

        if not hasattr(self, "nrows"):
            self.nrows = 50
            self.ncols = 50

        self.gdf, self.merit_gdf = ep.main(self.gdf, self.verbose, self.nrows, self.ncols)

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

        return ss.main(
            self.basins, datasets, reducer_funcs=reducer_funcs, folder=folder, verbose=self.verbose, test=test
        )

    def export(self, what="all"):
        """
        Exports data computed by rapbro.

        Parameters
        ----------
        what : list, optional
            Which data should be exported? Choose from
            'all' - all computed data
            'elevs' - centerline json is exported with elevation and distance attributes
            'subbasins' - subbasins and incremental subbasins shapefiles
            The default is 'all'.

        """
        if what == "all":
            what = ["elevs", "subbasins"]
        if self.verbose:
            print(f"Exporting {what} to {self.paths['basenamed']}.")

        if type(what) is str:
            what = [what]

        for w in what:
            if w not in ["elevs", "subbasins"]:
                raise KeyError(
                    f"Requested export {w} not available. Choose from {['elevs', 'subbasins']}."
                )
            elif w == "subbasins":
                if hasattr(self, "basins"):
                    self.basins.to_file(self.paths["subbasins"], driver="GeoJSON")
                    if hasattr(self, "basins_inc"):
                        self.basins_inc.to_file(self.paths["subbasins_inc"], driver="GeoJSON")
                    if self.verbose:
                        print("Basins geojson written successfully.")
                else:
                    print("No subbasins found for export.")
            elif w == "elevs":
                if "Elevation (m)" in self.gdf.keys():
                    self.gdf.to_file(self.paths["centerline_results"], driver="GeoJSON")
                if hasattr(self, "merit_gdf"):
                    self.merit_gdf.to_file(self.paths["dem_results"], driver="GeoJSON")
                    if self.verbose:
                        print("Centerline geojsons written successfully.")
                else:
                    print("No elevations found for export.")
