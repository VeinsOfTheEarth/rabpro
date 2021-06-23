# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:55:00 2020

@author: Jon
"""

from osgeo import gdal
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
import elev_profile as ep
import subbasins as sb
import subbasin_stats as ss
import utils as rpu
import merit_utils as mu
from pyproj import CRS


class profiler:
    """
    The profiler class organizes data and methods for using RaBPro. This is
    a parent class to the centerline and point classes which inherit profiler
    methods and attributes.

    Attributes
    ----------
    coords : tuple OR list OR str OR GeoDataFrame
        Coordinates of point(s) to delineate. A single point may be provided as
        a (lat, lon) tuple. If a river centerline is being provided, its points
        must be arranged US->DS. River centerline coordinates may be provided
        as a list of (lat, lon) pairs. Point(s) may also be provided via a
        shapefile or .csv file. If provided as .csv file, the columns must
        be labeled 'latitude' and 'longitude'. If a shapefile path is provided,
        the shapefile may also contain columns for widths and/or along-valley
        distances. These columns should have "width" and "distance" somewhere
        in their column names, respectively, in order for RaBPro to exploit
        them.
    da : number
        Represents the drainage area in square kilometers of the downstream-most
        point in the provided coords.
    name : str
        Name of location/river, used for filename saving purposes only.
    verbose : bool
        If True, will print updates as processing progresses.


    """

    def __init__(
        self,
        coords,
        da=None,
        name="unnamed",
        path_results=None,
        force_merit=False,
        verbose=True,
    ):

        self.name = name
        self.verbose = verbose

        # Parse the provided coordinates into a GeoDataFrame (if not already)
        if type(coords) is tuple:  # A single point was provided
            self.gdf = self.coordinates_to_gdf([coords])
        elif type(coords) is list:  # A list of tuples was provided (centerline)
            self.gdf = self.coordinates_to_gdf(coords)
        elif type(coords) is str:  # A path to .csv or .shp file was provided
            ext = coords.split(".")[-1]
            if ext == "csv":
                self.gdf = self.csv_to_gdf(coords)
            elif ext == "shp" or ext == "json" or ext == "geojson":
                self.gdf = gpd.read_file(coords)
        elif (
            type(coords) is gpd.geodataframe.GeoDataFrame
        ):  # A GeoDataFrame was provided.
            # Convert it to EPSG:4326
            self.gdf = coords
            if self.gdf.crs.to_epsg() != 4326:
                self.gdf = self.gdf.to_crs(CRS.from_epsg(4326))
                print("Reprojecting provided coordinates to EPSG:4326.")
        else:
            raise ValueError("Invalid coordinate input type.")

        # Determine the method for delineation
        self.da = da
        self.method = self.which_method(force_merit)

        # Append drainage area to gdf
        if len(self.gdf) == 1:
            if self.da is not None:
                self.gdf["DA"] = da

        # Prepare and fetch paths for exporting results
        self.paths = rpu.get_exportpaths(
            self.name, basepath=path_results, overwrite=True
        )

        # This line will ensure that all the virtual rasters are built
        # and available.
        _ = rpu.get_datapaths()

    def coordinates_to_gdf(self, coords):
        """
        Converts a list of coordinates to a GeoDataFrame. Coordinates should
        be (lat, lon) pairs with EPSG==4326.
        """
        geoms = []
        for xy in coords:
            geoms.append(shapely.geometry.Point((xy[1], xy[0])))

        gdf = gpd.GeoDataFrame(geometry=geoms)
        gdf.crs = CRS.from_epsg(4326)

        return gdf

    def csv_to_gdf(self, csvpath):
        """
        Creates a GeoDataFrame from an input path to a csv. The csv must contain
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
        geoms = []
        for lat, lon in zip(lats, lons):
            geoms.append(shapely.geometry.Point((lon, lat)))

        gdf = gpd.GeoDataFrame(geometry=geoms)
        gdf.crs = CRS.from_epsg(4326)

        return gdf

    def ensure_crs_match(self):
        """
        Checks that the crs of the coordinates GeoDataFrame matches that
        supplied by keyword argument.
        """

        if self.gdf.crs is not None:
            if self.EPSG != self.gdf.crs.to_epsg():
                raise ValueError(
                    "Provided EPSG is {}, but GeoDataFrame thinks EPSG is {}. Rectify before continuing.".format(
                        self.EPSG, self.coords.crs["init"][5:]
                    )
                )
        else:
            self.gdf.crs = CRS.from_epsg(self.EPSG)

        return

    def is_centerline(self):
        """
        Determines if the provided coordinates define a centerline. Basically
        just checks for the number of input pairs.
        """
        if self.gdf.shape[0] == 1:
            return False
        else:
            return True

    def which_method(self, force_merit, merit_thresh=500):
        """
        Returns the method to use for delineating watersheds.
        """

        method = "hydrobasins"
        if self.da is not None:
            if self.da < merit_thresh:
                method = "merit"

        if force_merit is True:
            method = "merit"

        return method

    def delineate_basins(self, search_radius=None, map_only=False):
        """
        Computes the watersheds for each lat/lon pair and adds their drainage
        areas to the self.gdf GeoDataFrame.

        There are two methods used for delineating basins: HydroBASINS and MERIT.
        HydroBASINS is appropriate for large basins (500 km^2 and larger), and
        MERIT can provide more detailed basin delineations for smaller basins.
        The method used depends on the size of the basin, which is interpreted
        via the provided drainage area. If no drainage area was provided,
        HydroBASINS will be used. Otherwise, if the provided drainage area is
        less than 500 km^2, MERIT will be used. MERIT may also be forced for
        larger basins using the 'force_merit' argument when instantiating the
        profiler.
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
                if map_only is False:
                    print(
                        "Could not find a suitable flowline to map given coordinate and DA. No basin can be delineated."
                    )
            else:
                self.gdf["DA"] = [None for p in range(self.gdf.shape[0])]
                self.gdf["DA"].values[0] = self.basins["DA"][0]

                # Ensure the MERIT-delineated polygon's area is within 10% of the mapped value
                # If the basin crosses the -180/180 meridian, need to use a projection that splits elsewhere
                if self.mapped["meridian_cross"] is True:
                    rp_epsg = 2193  # new zealand, seems to work fine for areas though
                else:
                    rp_epsg = 3410
                reproj_ea_meters = self.basins.to_crs(crs=CRS.from_epsg(rp_epsg))
                pgon_area = (
                    reproj_ea_meters.geometry.values[0].area / 10 ** 6
                )  # square km
                pct_diff = abs(pgon_area - self.mapped["da"]) / self.mapped["da"] * 100
                if pct_diff > 10:
                    print(
                        "Check delineated basin. There is a difference of {} % between MERIT DA and polygon area.".format(
                            pct_diff
                        )
                    )

    def elev_profile(self):

        if hasattr(self, "nrows") is False:
            self.nrows = 50
            self.ncols = 50

        self.gdf, self.merit_gdf = ep.main(
            self.gdf, self.verbose, self.nrows, self.ncols
        )

    def smooth_elevations(self, windowsize, k=1):
        """
        Applies a Savitzky-Golay filter to the raw elevation profile. I suggest
        leaving the polynomial order (k) at 1, which is a moving average. Higher
        order polynomials will oftentimes result in portions of the profile
        that go 'uphill.'

        Parameters
        ----------
        windowsize : int
            Size of the window in number of centerline vertices.
        k : int, optional
            order of the polynomial to use for regression. The default is 1.

        Returns
        -------
        elev_smooth : np.array
            The smoothed elevations.
        """
        if ~hasattr(self, "stats"):
            print("Elevations have not yet been computed.")
        if windowsize % 2 == 0:
            windowsize = int(windowsize + 1)

        elev_smooth = savgol_filter(self.elevs["elev_raw"], windowsize, k)
        return elev_smooth

    def basin_stats(self, years="all"):
        """
        Computes watershed statistics.

        Keywords
        ----------
        years : str OR list
            If years = 'all', then all available years of statistics will be
            returned. Else years should be a two-entry list like [2000 2012]
            that specify the start and end year of the desired analysis.
        """

        if years == "all":
            years = [1900, 2200]

        self.stats = ss.main(
            self.gdf, self.basins, years[0], years[1], verbose=self.verbose
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
            'stats' - csv of all subbasin statistics computed
            The default is 'all'.

        Returns
        -------
        None.

        """
        if what == "all":
            what = ["elevs", "subbasins", "stats"]
        if self.verbose is True:
            print("Exporting {} to {}.".format(what, self.paths["basenamed"]))

        if type(what) is str:
            what = [what]

        for w in what:
            if w not in ["elevs", "subbasins", "stats"]:
                raise KeyError(
                    "Requested export {} not available. Choose from {}.".format(
                        w, ["elevs", "subbasins", "stats"]
                    )
                )
            if w == "stats":
                if hasattr(self, "stats"):
                    self.stats.to_csv(self.paths["stats"], index=False)
                    if self.verbose is True:
                        print("Statistics written successfully.")
                else:
                    print("No basin statistics found for export.")
            elif w == "subbasins":
                if hasattr(self, "basins"):
                    self.basins.to_file(self.paths["subbasins"], driver="GeoJSON")
                    if hasattr(self, "basins_inc"):
                        self.basins_inc.to_file(
                            self.paths["subbasins_inc"], driver="GeoJSON"
                        )
                    if self.verbose is True:
                        print("Basins geojson written successfully.")
                else:
                    print("No subbasins found for export.")
            elif w == "elevs":
                if "Elevation (m)" in self.gdf.keys():
                    self.gdf.to_file(self.paths["centerline_results"], driver="GeoJSON")
                if hasattr(self, "merit_gdf") is True:
                    self.merit_gdf.to_file(self.paths["dem_results"], driver="GeoJSON")
                    if self.verbose is True:
                        print("Centerline geojsons written successfully.")
                else:
                    print("No elevations found for export.")
