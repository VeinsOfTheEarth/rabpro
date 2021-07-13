# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:10:12 2020

@author: Jon
"""

# Build merit vrts
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
    """
    cl_gdf should have a column called 'DA' that stores drainage areas
    cl_gdf should be in 4326 for use of the Haversine formula...could add a
    check and use other methods, but simpler this way.
    """
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
            [
                cl_gdf.geometry.values[0].coords.xy[0][0],
                cl_gdf.geometry.values[0].coords.xy[1][0],
            ]
        )
        ds_da = cl_gdf.DA.values[-1]
        us_da = cl_gdf.DA.values[0]
        cr_ds_mapped, _ = mu.map_cl_pt_to_flowline(
            ds_lonlat, da_obj, nrows, ncols, ds_da
        )
        cr_us_mapped, _ = mu.map_cl_pt_to_flowline(
            us_lonlat, da_obj, nrows, ncols, us_da
        )
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
                (ls_cl.coords.xy[1][0], int_pt[1][0]),
                (ls_cl.coords.xy[0][0], int_pt[0][0]),
            )[0]
            ds_dist_cl = ru.haversine(
                (ls_cl.coords.xy[1][1], int_pt[1][0]),
                (ls_cl.coords.xy[0][1], int_pt[0][0]),
            )[0]
            if us_dist_cl < ds_dist_cl:
                cl_idx = clidx
            else:
                cl_idx = clidx + 1

            us_dist_dem = ru.haversine(
                (ls_dem.coords.xy[1][0], int_pt[1][0]),
                (ls_dem.coords.xy[0][0], int_pt[0][0]),
            )[0]
            ds_dist_dem = ru.haversine(
                (ls_dem.coords.xy[1][1], int_pt[1][0]),
                (ls_dem.coords.xy[0][1], int_pt[0][0]),
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


""" Smoothing and computing slopes after this """
""" TURNED THESE OFF -- USER CAN APPLY SMOOTHING """
# if verbose:
#     print('Computing slopes...')
# # Method to reduce number of points while filtering/smoothing
# # To avoid overfitting, we'll only take the midpoints of "flat" stretches
# # of elevations. This was originally written for integer-valued DEM, so
# # we first convert the elevation profile to round integers.
# es_i = np.round(es)
# dists = dists - dists[0] # Begin profile at dist=0
# emin, emax = np.min(es_i), np.max(es_i)
# evals = list(range(int(emax), int(emin-1), -1))
# dvals = []
# rem_evals = []
# for e in evals:
#     idcs = np.where(e==es_i)[0]
#     dist_temp = dists[idcs]
#     if len(dist_temp) == 0:
#         rem_evals.append(e)
#     else:
#         middist = dist_temp[0] + (dist_temp[-1] - dist_temp[0])/2
#         closest_idx = np.argmin(np.abs(dist_temp-middist))
#         dvals.append(dists[idcs[closest_idx]])
# for re in rem_evals:
#     evals.remove(re)
# dvals = np.ndarray.flatten(np.array(dvals))
# # Extend to extents of data--this means we are using the endpoints (not the midpoints) for the first and last segments
# dvals[0] = dists[0]
# dvals[-1] = dists[-1]

# # Evenly-space the signal before filtering. Number of segments depends on total elevation loss; roughly 1 point per 1 meters lost.
# npts = int((emax - emin)/1)
# es_ds = np.linspace(0, dists[-1], npts)
# t,c,k = si.splrep(dvals,evals,s=0,k=1)
# es_es = si.splev(es_ds, (t,c,k), der=0)

# # If the signal has less than 5 points (i.e. less than 5 drops in elevation)
# # treat it as a piecewise linear (See Pechora for example)
# if npts > 5:
#     # Now smooth the evenly-spaced signal
#     window = max(3, int(round(len(evals)/4)))
#     if window % 2 == 0:
#         window = int(window + 1)

#     # Manually pad the signal, as the default "modes" offered by scipy don't cut it
#     # We use a flipped-mirror
#     prepend = np.flipud(2*es_es[0] - es_es[1:window+1])
#     postpend = np.flipud(2*es_es[-1] - es_es[npts -window-1:-1])
#     es_es_pad = np.concatenate((prepend, es_es, postpend))
#     e_filt = signal.savgol_filter(es_es_pad, window_length=window, polyorder=1, mode='interp')

#     # Unpad the filtered signal
#     e_filt = e_filt[window:len(e_filt)-window]

#     # Create quadratic spline to sample elevations from
#     t,c,k = si.splrep(es_ds,e_filt,s=0,k=3)
#     es_final_smooth = si.splev(dists, (t,c,k), der=0)

# else:
#     es_final_smooth = si.splev(dists, (t,c,k), der=0)


# """ Compute slopes from smooth elevations and smooth the slopes """
# window2 = int(0.5 * dists[-1] / int(round((emax - emin))) / np.mean(ds)) # Window is distance required to drop 1/2 meter
# if window2 % 2 == 0:
#     window2 = int(window2 + 1)
# window2 = max(3, window2)
# slope_smooth = si.splev(dists, (t,c,k), der=1)
# slope_smooth = signal.savgol_filter(slope_smooth, window_length=window2, polyorder=1, mode='interp')
# slope_smooth = -slope_smooth

# """ Compute slope via control theory (old RWF method by Jordan Muss) """
# # Turn off warnings for the linear OC code
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# # Use the downsampled version instead of all points
# s_elev = evals
# s_dist = dvals
# # Find slopes along elevation profile
# linOCslopes = calcStreamSlopes(elev=s_elev, dist=s_dist)
# # Assign the slopes to each centerline point
# ns, ks = linOCslopes.report()
# idcsfit = np.cumsum(ns)-1
# idcsfit = np.insert(idcsfit, [0], 0)
# s_breaks = s_dist[idcsfit]
# slope_linOC = np.zeros_like(es_final_smooth)
# for i in range(len(s_breaks)-1):
#     idcs = np.where(np.logical_and(dists >= s_breaks[i], dists <= s_breaks[i+1]))[0]
#     slope_linOC[idcs] = -linOCslopes.report()[1][i] # note negative to switch sense opposite of what Jordan coded

# # Turn warnings back on
# warnings.filterwarnings("default", category=RuntimeWarning)

# # Save elevations and slopes in a dict
# elevdict = dict()
# # The following two are DEM-derived elevation profile/distances
# elevdict['dem_dist'] = np.insert(np.cumsum(haversine(dem_ll[0], dem_ll[1])), 0, 0)
# elevdict['dem_elev'] = es
# # The rest of these are centerline-specific
# elevdict['dist'] = dists
# elevdict['elev_raw'] = cl_elevs
# elevdict['elev_which_mapped'] = ~nans # Which points were mapped to DEM
# elevdict['elev_smooth'] = es_final_smooth
# elevdict['slope_smooth'] = slope_smooth
# elevdict['slope_linOC'] = slope_linOC

# return elevdict


def compute_dists(gdf):
    """
    Computes cumulative distance in meters between points in gdf.
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
    """
    Returns the values within the raster pointed to by gdobj specified by
    the row,col values in rc. Sets nodata. Returns numpy array.
    """

    vals = []
    for r, c in zip(rc[0], rc[1]):
        vals.append(gdobj.ReadAsArray(xoff=int(c), yoff=int(r), xsize=1, ysize=1)[0][0])
    vals = np.array(vals)
    vals[vals == nodata] = np.nan

    return vals


def pts_to_line_segments(pts):
    """
    Converts a list of shapely points to a set of line segments. Points should
    be in order.

    Returns a  list of linestrings of length N-1, where N=length(pts).
    """
    ls = []
    for i in range(len(pts) - 1):
        ls.append(LineString((pts[i], pts[i + 1])))

    return ls


def s_ds(xs, ys):

    ds = np.sqrt((np.diff(xs)) ** 2 + (np.diff(ys)) ** 2)
    s = np.insert(np.cumsum(ds), 0, 0)

    return s, ds


def find_nangroups(arr):
    """
    Returns groups of nans in an array.
    """
    nans = np.isnan(arr)
    nangroups = []
    nangroup = []
    for i, n in enumerate(nans):
        if n == False:
            if len(nangroup) > 0:
                nangroups.append(nangroup)
            nangroup = []
        else:
            nangroup.append(i)

    return nangroup


def interpolate_nangroups(arr, dists, nangroups):
    """
    Linearly interpolates across groups of nans in a 1-D array.
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


class calcStreamSlopes:  # by Jordan Muss
    def __init__(self, elev=None, dist=None):
        self.slopes = []
        self.fit = []
        self.residuals = []
        self.intercept = []
        if len(elev) != len(dist):
            print(
                "Error in 'calcStreamSlopes': the elevation and distance list lengtru are not equal"
            )
            self.success = False
        elif (elev is None) or (dist is None):
            print(
                "Error in 'calcStreamSlopes': the elevation and distance list must not be empty"
            )
            self.success = False
        else:
            self.elev = elev
            self.dist = dist
            eol = len(self.elev)
            idx_start = 0
            counter = 0
            while idx_start < eol:
                idx_end = self.getOCpoint(idx_start, eol)
                #                 ''' Test that the residuals of the subset reach do not have
                #                     an out of control occurence. This is only done once: '''
                #                 idx_end = self.getOCpoint(idx_start, idx_end)
                """The base reach model is: """
                reach = lm(
                    x=self.dist[idx_start:idx_end], y=self.elev[idx_start:idx_end]
                )
                """ Add points to the subset until the adjusted R-square
                    stops improving: """
                adjRsquare = reach.adj_r_square
                idx_end += 2
                extended_reach = lm(
                    x=self.dist[idx_start:idx_end], y=self.elev[idx_start:idx_end]
                )
                while (adjRsquare < extended_reach.adj_r_square) and (idx_end <= eol):
                    adjRsquare = extended_reach.adj_r_square
                    idx_end += 2
                    extended_reach = lm(
                        x=self.dist[idx_start:idx_end], y=self.elev[idx_start:idx_end]
                    )
                if adjRsquare > extended_reach.adj_r_square:
                    while adjRsquare > extended_reach.adj_r_square:
                        idx_end -= 1
                        extended_reach = lm(
                            x=self.dist[idx_start:idx_end],
                            y=self.elev[idx_start:idx_end],
                        )
                if idx_end > eol:
                    idx_end = eol
                reach = lm(
                    x=self.dist[idx_start:idx_end], y=self.elev[idx_start:idx_end]
                )
                self.slopes += [reach.slope] * (idx_end - idx_start)
                self.residuals.append(reach.residuals)
                idx_start = idx_end
                self.fit.append(reach.adj_r_square)
                self.intercept.append(reach.intercept)
                counter = counter + 1
            self.success = True

    def getOCpoint(self, start_idx, end_idx):
        reach = lm(x=self.dist[start_idx:end_idx], y=self.elev[start_idx:end_idx])
        cs = cusum(reach.residuals, mu=0, sigma=1, num_deviations=1, H=4)
        idx_OC = cs.getTwoWayOC()
        idx_end = start_idx + idx_OC + 1
        return idx_end

    def report(self):
        slopes = ["{:.10f}".format(i) for i in self.slopes]
        slopes = [i for i in self.slopes]
        t = []
        for s in slopes:
            if s not in t:
                t.append(s)
        slope_dict = {
            k: np.ma.masked_where(np.array(slopes) == k, slopes).mask.sum() for k in t
        }
        #        for k in t: print("%d : %s (adj_r_square = %.5f" % (slope_dict[k], k, self.fit[t.index(k)]))
        ns = []
        ks = []
        for k in t:
            ns.append(slope_dict[k])
            ks.append(k)
        return ns, ks


class cusum:  # by Jordan Muss
    def __init__(self, time_series, mu, sigma, num_deviations, H):
        self.ts = time_series
        if not isinstance(self.ts, np.ndarray):
            self.ts = np.array(self.ts)
        self.K = num_deviations / 2.0 * float(sigma)
        self.OC_lim = H * sigma
        self.mu = mu

    def calcCusum(self, Neg=False):
        flag = False
        last_in_control = len(self.ts) - 1
        self.Neg = Neg
        if Neg:
            dev = (self.mu - self.K) - self.ts
        else:
            dev = self.ts - (self.mu + self.K)
        C = self.cplus(dev)
        OC_idx = np.where(C > self.OC_lim)[0]
        OC_idx_all = OC_idx.copy()
        if OC_idx.size > 0:
            flag = True
            OC_idx = OC_idx[0]
            steps_back = np.where(C[:OC_idx][::-1] == 0)[0]
            if steps_back.size == 0:
                last_in_control = 0
            else:
                last_in_control = OC_idx - steps_back[0] - 1
        self.out_of_control = flag
        self.last_in_control = last_in_control
        self.OC_idx = OC_idx
        self.OC_idx_all = OC_idx_all
        self.C = C
        return self

    def getTwoWayOC(self):
        """Get the first point where the process goes out of control using a
        two way (up/down) cusum. Return the eol value if the OC point is
        not found or is the first point (0)."""
        up_OC = down_OC = len(self.ts) - 1
        """ Up: """
        self.calcCusum(Neg=True)
        if self.out_of_control and (self.last_in_control > 0):
            up_OC = self.last_in_control
        """ Down:"""
        self.calcCusum(Neg=False)
        if self.out_of_control and (self.last_in_control > 0):
            down_OC = self.last_in_control
        self.LiC_two_way = min(up_OC, down_OC)
        return self.LiC_two_way

    def cplus(self, dev_list):
        l_pos = np.amax(np.array((dev_list, [0] * len(dev_list))), axis=0)
        first_pos = np.where(l_pos > 0)[0]
        if first_pos.size > 0:
            first_pos = first_pos[0]
            head_list = l_pos[:first_pos]
            tail_list = dev_list[first_pos:]
            cum_tail = np.cumsum(tail_list)
            first_neg = np.where(cum_tail < 0)[0]
            if first_neg.size > 0:
                first_neg = first_neg[0]
                tail_list = self.cplus(tail_list[first_neg:])
                head_list = np.append(head_list, cum_tail[:first_neg].tolist())
            else:
                tail_list = cum_tail
        else:
            head_list = []
            tail_list = np.cumsum(dev_list)
        return np.append(head_list, tail_list)


class lm:  # by Jordan Muss
    def __init__(self, x, y, xName="x", yName="y", forceIntercept=False):
        """Perform a simple single predictor (y~x) linear regression:"""
        """ToDo: add code to force the intercept through zero"""
        from scipy.stats import linregress
        import numpy as np

        self.modelX = x
        self.modelY = y
        self.x_name = xName
        self.y_name = yName
        self.model = linregress(x, y)
        self.slope = self.model.slope
        self.intercept = self.model.intercept
        self.r = self.model.rvalue
        self.slope_stderr = self.model.stderr
        self.slope_p_value = self.model.pvalue
        self.predicted = self.intercept + [self.slope * i for i in x]
        self.residuals = y - self.predicted
        if forceIntercept:
            data_df = len(x) - 1
            self.df = len(x) - 1
        else:
            data_df = len(x) - 1
            self.df = len(x) - 2
        SSE = np.sum(self.residuals ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        self.r_square = 1
        if SST > 0:
            self.r_square -= SSE / SST
        if self.df > 0:
            self.adj_r_square = 1 - (1 - self.r_square) * (float(data_df) / self.df)
        else:
            self.adj_r_square = 0
