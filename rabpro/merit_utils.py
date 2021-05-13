# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:40:33 2020

@author: Jon
"""
import rivgraph.im_utils as im
from shapely.geometry import Polygon
from shapely import ops
import math
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
import utils as ru


def trace_flowpath(fdr_obj, da_obj, cr_stpt, cr_enpt=None, n_steps=None):    
    """
    Walks along a flow direction grid from stpt to enpt. Returns a list of 
    pixels from stpt to enpt. Walks from downstream to upstream.
    
    fdr_obj - flow direction object opened with gdal.Open(). Assumes flow
              direction symbology matches MERIT-Hydro:
                32 64 128
                16     1
                8   4  2
    cr_stpt - column, row of point to start walk
    cr_enpt - column, row of point to end walk
    n_steps - optional; number of steps (pixels) to walk before halting
    """      
    imshape = (fdr_obj.RasterXSize, fdr_obj.RasterYSize)           
    intodirs = np.array((2, 4, 8, 1, 3, 16, 128, 64, 32), dtype=np.uint8) # The 3 is a dummy variable placeholder. Basically want it to be any value that cannot occur in the flow directions map.
    rowdict = {2: -1, 4: -1, 8: -1, 1: 0, 16: 0, 128: 1, 64: 1, 32: 1}
    coldict = {2: -1, 4: 0, 8: 1, 1: -1, 16: 1, 128: -1, 64: 0, 32: 1}

    stpti = np.ravel_multi_index(cr_stpt, imshape)

    da = [da_obj.ReadAsArray(xoff=int(cr_stpt[0]), yoff=int(cr_stpt[1]), xsize=1, ysize=1)[0][0]]
    do_pt = [stpti]
    ct = 0
    while 1:
        cr = np.unravel_index(do_pt[-1], imshape)
        
        # First find all the candidate pixels that drain to this one
        nb_fdr = neighborhood_vals_from_raster(cr, (3,3), fdr_obj, nodataval=np.nan).reshape(1,9).flatten()
        # nb_fdr = fdr_obj.ReadAsArray(xoff=int(cr[0])-1, yoff=int(cr[1])-1, xsize=3, ysize=3).reshape(1, 9).flatten()
        candidates = np.where(nb_fdr == intodirs)[0]
        if len(candidates) == 0:
            break
        elif len(candidates) == 1:
            fdr = nb_fdr[candidates[0]]
        else:
            nb_das = neighborhood_vals_from_raster(cr, (3,3), da_obj, nodataval=np.nan).reshape(1,9).flatten()
            # nb_das = da_obj.ReadAsArray(xoff=int(cr[0])-1, yoff=int(cr[1])-1, xsize=3, ysize=3).reshape(1,9).flatten()
            fdr = nb_fdr[candidates[np.argmax(nb_das[candidates])]]
        
        # Take the step
        row = cr[1] + rowdict[fdr]
        col = cr[0] + coldict[fdr]
        
        # Handle meridian wrapping
        if col < 0:
            col = fdr_obj.RasterXSize + col
        elif col > fdr_obj.RasterXSize - 1:
            col = col - fdr_obj.RasterXSize
        
        do_pt.append(np.ravel_multi_index((col, row), imshape))
        da.append(da_obj.ReadAsArray(xoff=int(col), yoff=int(row), xsize=1, ysize=1)[0][0])
        
        # Halt if we've reached the endpoint
        if cr == cr_enpt:
            break
        
        # Halt if we've reached the requested length
        ct = ct + 1
        if ct == n_steps:
            break

    colrow = np.unravel_index(do_pt, imshape)
        
    return (colrow[1], colrow[0])


def neighborhood_vals_from_raster(cr, shape, vrt_obj, nodataval=np.nan, wrap=None):
    """
    Queries a (virtual) raster object to return an array of neighbors surrounding
    a given point specified by cr (column, row). A shape can be provided to
    return as large of a neighborhood as desired; both dimensions must be odd.
    This function is almost always unnecessary and could be replaced with a
    single call to gdal's ReadAsArray(), except that throws errors when requesting
    a neighborhood that is beyond the boundaries of the raster. Also note that
    requests for negative offsets do not throw errors, which is dangerous.
    This function checks for those cases and handles them. 
    
    An option is provided to 'wrap' in cases where neighborhoods are requested 
    beyond the bounds of the raster. In these cases, the raster is effectively
    "grown" by appending copies of itself to ensure no nodata are returned.
    (This isn't actually how the code works, just the simplest explanation.)

    Parameters
    ----------
    cr : tuple
        (column, row) indices within the virtual raster specifying the point
        around which a neighborhood is requested.
    shape : tuple
        Two-element tuple (nrows, ncols) specifying the shape of the neighborhood
        around cr to query.
    vrt_obj : gdal.Dataset
        Dataset object pointing to the raster from which to read; created by
        gdal.Open(path_to_raster).
    nodataval : object
        Value to assign neighbors that are beyond the bounds of the raster.
        Default is np.nan.
    wrap : str or None
        String of 'h', 'v', or 'hv' denoting if horizontal and/or vertical
        wrapping is desired. If None, no wrapping is performed. Default is None.

    Returns
    -------
    Ivals : np.array
        Array of same dimensions as shape containing the neighborhood values.

    """       
    nan_int = -9999 # denotes nan in an integer array since np.nan can't be stored as an integer
    
    if wrap is None:
        wrap = ''

    # Ensure valid sizes provided
    for s in shape:
        if s % 2 != 0:
            RuntimeError('Requested sizes must be odd.')
    
    # Compute offsets
    halfcol = int((shape[1]-1)/2)
    halfrow = int((shape[0]-1)/2)

    # Ensure not requesting data beyond bounds of virtual raster
    imshape_idcs = (vrt_obj.RasterXSize - 1, vrt_obj.RasterYSize - 1)
    max_requ_c = cr[0] + halfcol  
    min_requ_c = cr[0] - halfcol
    max_requ_r = cr[1] + halfrow
    min_requ_r = cr[1] - halfrow
    
    c_idcs = np.arange(min_requ_c, max_requ_c + 1)
    r_idcs = np.arange(min_requ_r, max_requ_r + 1)
    
    # Handle beyond-boundary cases 
    individually_flag = False
    if max_requ_c > imshape_idcs[0]:
        individually_flag = True
        replace = c_idcs > imshape_idcs[0]
        if 'h' in wrap:
            c_idcs[replace] = np.arange(0, np.sum(replace))
        else:
            c_idcs[replace] = nan_int
        
    if max_requ_r > imshape_idcs[1]:
        individually_flag = True
        replace = r_idcs > imshape_idcs[1]
        if 'v' in wrap:
            r_idcs[replace] = np.arange(0, np.sum(replace))
        else:
            r_idcs[replace] = nan_int
    
    if min_requ_c < 0:
        individually_flag = True
        replace = c_idcs < 0
        if 'h' in wrap:
            c_idcs[replace] =  np.arange(imshape_idcs[0], imshape_idcs[0]-np.sum(replace), -1)
        else:
            c_idcs[replace] = nan_int
            
    if min_requ_r < 0:
        individually_flag = True
        replace = r_idcs < 0 
        if 'v' in wrap:
            r_idcs[replace] = np.arange(imshape_idcs[1], imshape_idcs[1]-np.sum(replace), -1)
        else:
            r_idcs[replace] = nan_int
  
    if individually_flag is True:
        Ivals = np.ones(shape).T * nodataval
        for ic, c in enumerate(c_idcs):
            for ir, r in enumerate(r_idcs):
                if c == nan_int or r == nan_int:
                    continue
                else:
                    Ivals[ir,ic] = vrt_obj.ReadAsArray(xoff=int(c), yoff=int(r), xsize=int(1), ysize=int(1))[0][0]
    else:
         Ivals = vrt_obj.ReadAsArray(xoff=int(cr[0]-halfcol), yoff=int(cr[1]-halfrow), xsize=int(shape[1]), ysize=int(shape[0]))       
         
    return Ivals
 

def get_basin_pixels(start_cr, da_obj, fdr_obj, fdir_map=[32,64,128,16,1,8,4,2]):
    """
    Returns the indices of all pixels draining to the pixel defined by 
    start_cr.

    Parameters
    ----------
    start_rc : TYPE
        DESCRIPTION.
    remove : TYPE, optional
        DESCRIPTION. The default is set().

    Returns
    -------
    done : TYPE
        DESCRIPTION.
        
    64 128 1
    32     2
    16  8  4
    """
            
    # Make arrays for finding neighboring indices
    imshape = (fdr_obj.RasterXSize, fdr_obj.RasterYSize) 
    intodirs = np.flipud(np.array(fdir_map, dtype=np.uint8))
    intodirs = np.insert(intodirs, 4, 0)
    
    coladds = np.array((-1, 0, 1, -1, 0, 1, -1, 0, 1)) * imshape[1]
    rowadds = np.array((-1, -1, -1, 0, 0, 0, 1, 1, 1)) 

    start_idx = np.ravel_multi_index(start_cr, imshape)
    done = set()
    todo = set([start_idx])
    while todo:

        doidx = todo.pop()
        done.add(doidx)
        
        do_cr = np.unravel_index(doidx, imshape)
        nb_fdr = neighborhood_vals_from_raster(do_cr, (3,3), fdr_obj, nodataval=-999, wrap='h').reshape(1,9).flatten()
        where_into = intodirs == nb_fdr
        
        if where_into.sum() == 0:
            continue
        
        # Adjust for near-boundary cases, only perform column wrapping though
        if do_cr[0] == 0 or do_cr[0] == imshape[0]-1:
            neighs_into = []
            wii = np.where(where_into)[0]
            for wi in wii:
                temp_col = int(do_cr[0] + coladds[wi] / imshape[1])
                if temp_col < 0:
                    temp_col = imshape[0] + temp_col
                elif temp_col > imshape[0] - 1:
                    temp_col = temp_col - imshape[0]
                neighs_into.append(np.ravel_multi_index((temp_col, do_cr[1] + rowadds[wi]), imshape))
        else:            
            neighs_into = doidx + rowadds[where_into] + coladds[where_into]
        
        for ni in neighs_into:
            if ni not in done:
                todo.add(ni)
                            
    return list(done)


def blob_to_polygon_shapely(I, ret_type='coords', buf_amt=0.001):
    """
    Should return a list of polygons or coords.

    Parameters
    ----------
    I : TYPE
        DESCRIPTION.
    ret_type : TYPE, optional
        DESCRIPTION. The default is 'coords'.
    buf_amt : TYPE, optional
        DESCRIPTION. The default is 0.001.

    Raises
    ------
    KeyError
        DESCRIPTION.

    Returns
    -------
    ret : TYPE
        DESCRIPTION.

    """
    # Storage
    ret = []
    
    # Get perimeter pixels of the blob
    rp, _ = im.regionprops(I, props=['perimeter'])
    
    for p in rp['perimeter']:    
        # Make slightly-buffered shapely polygons of each pixel's outline
        pix_pgons = []
        for x, y in zip(p[:,1],p[:,0]):
            pix_pgons.append(Polygon([(x, y),(x+1, y),(x+1, y+1),(x, y+1),(x,y)]).buffer(buf_amt))
        
        # Union the polygons and extract the boundary
        unioned = ops.unary_union(pix_pgons).buffer(-buf_amt)
        perimx, perimy = unioned.exterior.xy # I think unioned should always be a polygon and thus not throw errors, but not sure--could make MultiPolygons

        if ret_type == 'coords':
            ret.append(np.vstack((perimx, perimy)))
        elif ret_type == 'pgon':
            ret.append(Polygon(zip(perimx, perimy)))
        else:
            raise KeyError('Choose either "coords" or "pgon" as return types.')
            
    return ret


def idcs_to_geopolygons(idcs, gdobj, buf_amt=0.001):
    """
    Given a list of of pixel indices within a raster specified by gdobj, creates
    georeferenced polygons of the blobs formed by the union of the pixels.
    
    "Wrapping" is also checked for--this is to handle cases where the dateline
    meridian is crossed and return is therefore a set of polygons rather than
    a continuous one.

    Parameters
    ----------
    idcs : list of integers
        Pixel indices within the raster specified by gdobj that should be included
        in the polygon.
    gdobj : osgeo.gdal.Dataset
        Object created by gdal.Open() on a raster or virtual raster.
    buf_amt : numeric, optional
        Amount by which to buffer pixels before unioning-helps close tiny gaps. 
        The default is 0.001.

    Returns
    -------
    pgons : list
        List of georefernced polygons; one per blob of indices.
    crossing : bool
        If True, the polygons represent those that cross the dateline meridian
        (i.e. 180 degrees -> -180 degrees) and have been split.

    """
    def Icr_to_geopolygon(cr, mins, maxs, gt):
        # mins : [xmin, ymin]
        # maxs : [ymin, ymax]
        # Ishape : nrows, ncols in I
        # gt : geotransform
        
        pgons = []
        ncols, nrows = maxs[0] - mins[0] + 1, maxs[1] - mins[1] + 1
        I = np.zeros((nrows, ncols), dtype=np.bool)
        I[cr[1]-mins[1], cr[0]-mins[0]] = True
        coords = blob_to_polygon_shapely(I, ret_type='coords', buf_amt=0.001)
        for c in coords:
            coords_trans = ru.xy_to_coords(c[0] + mins[0] - 0.5, c[1] + mins[1] - 0.5, gdobj.GetGeoTransform())
            pgons.append(Polygon(zip(coords_trans[0], coords_trans[1])))

        return pgons
         
    # Storage
    pgons = []
    
    # Transform the coordinates
    imshape = (gdobj.RasterXSize, gdobj.RasterYSize)
    cr = np.unravel_index(idcs, imshape)
    
    # Max/min of coordinates
    xmax, xmin = np.max(cr[0]), np.min(cr[0])
    ymax, ymin = np.max(cr[1]), np.min(cr[1])
    
    # Check for wrapping
    crossing = False
    if xmax - xmin >= imshape[0]-1: # We have wrapping
        crossing = True
        # Split into west and east groups
        west = cr[0] < (imshape[0]-1)/2
        east = ~west
        
        for ew in [east, west]:
            cr_ew = np.vstack((cr[0][ew], cr[1][ew]))
            xmax, xmin = np.max(cr_ew[0]), np.min(cr_ew[0])
            ymax, ymin = np.max(cr_ew[1]), np.min(cr_ew[1])
            pgons.extend(Icr_to_geopolygon(cr_ew, (xmin, ymin), (xmax, ymax), gdobj.GetGeoTransform()))
    else:
        pgons.extend(Icr_to_geopolygon(cr, (xmin, ymin), (xmax, ymax), gdobj.GetGeoTransform()))
    
    return pgons, crossing

def idcs_to_polygon(idcs, gdobj, verbose=False):
    """
    Converts a set of indices within the MERIT dataset to a 
    polygon tracing their exterior.
    """    
    imshape = (gdobj.RasterXSize, gdobj.RasterYSize)
    # Make a raster to store the pixel locations
    cr = np.unravel_index(list(idcs), imshape)
    xmax, xmin = np.max(cr[0]), np.min(cr[0])
    ymax, ymin = np.max(cr[1]), np.min(cr[1])
    ncols, nrows = xmax - xmin + 1, ymax - ymin + 1
    I = np.zeros((nrows, ncols), dtype=np.bool)
    I[cr[1]-ymin, cr[0]-xmin] = True
    sumI = I.sum()
    
    # We need to return a non-self-intersecting polygon, which is problematic
    # when we have 8-connected pixels at the boundary. This method is not ideal
    # as it trims the watershed just for numerical convenience, but after some
    # research/thinking, a proper solution would be quite complex. Will add
    # verbosity to inform use of how many pixels were trimmed.
    I = im.largest_blobs(I, action='keep', connectivity=1)
    sumI_filt = I.sum()
    
    if verbose is True:
        nremoved = sumI - sumI_filt
        if nremoved > 0:
            print('{} corner-connected pixels out of {} were trimmed from basin...'.format(nremoved, sumI), end='')
    
    # Similarly, need to fill in 8-connected holes
    sumI = I.sum()
    I = im.fill_holes(I)
    if verbose is True:
        nadded = I.sum() - sumI
        if nadded > 0:
            print('{} pixels out of {} were filled in basin...'.format(nremoved, sumI), end='')
    
    # # Get the polygon of the remaining blob
    # ys, xs = blob_to_polygon(I)    
    # coords = ru.xy_to_coords(xs + xmin, ys + ymin, gdobj.GetGeoTransform())
    # pgon = Polygon(zip(coords[0], coords[1]))
    
    yss, xss = blob_to_polygon_shapely(I)
    coords = ru.xy_to_coords(xss + xmin - 0.5, yss + ymin - 0.5, gdobj.GetGeoTransform())
    pgon = Polygon(zip(coords[0], coords[1]))
        
    return pgon


def nrows_and_cols_from_search_radius(lon, lat, search_radius, gt):
    """
    search_radius is in meters.
    """
    
    # Determine the number of rows and columns to search
    los, las = [], []
    for b in [0, 180, 90, 270]:
        lo, la = ru.lonlat_plus_distance(lon, lat, search_radius/1000, bearing=b)
        los.append(lo)
        las.append(la)
    boundsxy = ru.lonlat_to_xy(np.array([min(los), max(los)]), np.array([min(las), max(las)]), gt)
    nrows = abs(boundsxy[0,1] - boundsxy[1,1])
    ncols = abs(boundsxy[0,0] - boundsxy[1,0])
    
    return nrows, ncols


def float_precision(x):
    """
    Taken from https://stackoverflow.com/questions/3018758/determine-precision-and-scale-of-particular-number-in-python
    If x is cast as a float(), it will return as many decimal places as seen upon print(x) -- i.e. no round-off for
    0.99999 etc.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return 0
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    scale = int(math.log10(frac_digits))
    return scale


def nrows_and_cols_from_coordinate_precision(lon, lat, gt):
    """
    UNUSED
    Returns the number of rows and columns to account for the precision of the
    provided coordinate. Assumes +/- one unit of the most precise coordinate.
    E.g. if the coordinate is 45.42, returns the nrows and ncols associated
    with +/- 0.01 degrees. 
    
    Assumes gt has same units as lon and lat.

    Parameters
    ----------
    lon : TYPE
        DESCRIPTION.
    lat : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    prec_lon = 10**-float_precision(lon)
    prec_lat = 10**-float_precision(lat)
    
    ncols = math.ceil(prec_lon/gt[1] * 2)
    nrows = math.ceil(prec_lat/abs(gt[5]) * 2)
    
    return nrows, ncols
    
    


def map_cl_pt_to_flowline(lonlat, da_obj, nrows, ncols, da=None):
    """
    Maps a point of known drainage area to a flowline in a FAC product. Here,
    we use MERIT-Hydro, whose FAC is in square kilometers. 
    
    Returns the row, col of the mapped-to pixel.
    
    lonlat - two-entry list or tuple or whatever of lon, lat of point to map
    da_obj - gdal object of drainage area raster created by gdal.Open()
    da - known drainage area of lonlat
    
    solve_method - indicates the reason why mapping succeeded/failed:
        1 - (success) DA provided; a nearby flowline pixel was found within 15% of the provided DA
        2 - (success) DA provided; match was found on a nearby flowline that is within our DA certainty bounds
        4 - (success) DA not provided; mapped to the nearest flowline (>1km^2)
        5 - (fail) DA not provided; no nearby flowlines exist
        6 - (fail) DA provided; but no nearby DAs were close enough to map to    
     
    """     
    # Need odd window values for the new value-puller
    if nrows % 2 == 0:
        nrows = nrows + 1
    if ncols % 2 == 0:
        ncols = ncols + 1
    
    cr = ru.lonlat_to_xy(lonlat[0], lonlat[1], da_obj.GetGeoTransform())
    pull_shape = (nrows, ncols)
    Idas = neighborhood_vals_from_raster(cr[0], pull_shape, da_obj, nodataval=np.nan)
    
    # Make an error image based on provided drainage area, if provided
    # This is difficult to do correctly because there are uncertainties with
    # the provided drainage area, as well as uncertainties in the MERIT 
    # drainage area. Furthermore, larger uncertainties are expected for 
    # larger drainage basins, and smaller for smaller. 
    
    def get_DA_error_bounds(da):
        """
        Returns the upper and lower drainage area values for a given target
        drainage area; the bounds correspond to the range of MERIT drainage
        areas to consider as candidates for mapping.
        
        The idea is that smaller basins (order 1-10 km^2) are allowed a greater
        % difference when searching for the point to map, while larger ones 
        are permitted smaller % differences. The reverese is true if considering
        absolute differences (i.e. 10 km^2 error is a much higher % for a 1 km^2
        basin than a 1000 km^2 basin).
        """
        
        # Make a set of points defining the allowable % error vs. da curve.
        # This curve will be linearly interpolated for the provided da to 
        # return the upper and lower bounds.
        das = [0.01, 0.1, 1, 100, 1000, 10000, 100000]
        pcts = [100, 75, 50, 25, 20, 15, 10]
        
        pct = np.interp(da, das, pcts)
        interval = np.abs(da * pct / 100)
        upper = da + interval
        lower = max(da - interval, 1) 
        lower = min(lower, da) # In case provided DA is less than 1
        return lower, upper
    
    # We first check if the point is positioned very-near perfectly to avoid
    # moving it around unnecessarily
    if da is not None:
        if np.abs(Idas[int((nrows-1)/2), int((ncols-1)/2)] - da) / da * 100 <= 15: # If the coordinate's DA is within 15% of MERIT's, we assume it's correct
            col_mapped = cr[0][0] 
            row_mapped = cr[0][1] 
            solve_method = 1 # a nearby flowline had a close DA
            return (col_mapped, row_mapped), solve_method
        
    # Compute a river network mask. Thresholds for candidate DAs are given by
    # the get_DA_error_bounds function.
    if da is not None:
        lower, upper = get_DA_error_bounds(da)
        Irn = np.logical_and(Idas >= lower, Idas <= upper) # Threshold to get the flowlines
        if Irn.sum() == 0: # If no valid flowlines are found, no mapping can be performed
            solve_method = 6 # A DA was provided but no nearby DAs were close enough to map to    
            return (np.nan, np.nan), solve_method
    else: # If no DA was provided, use all local flowlines (assumes DA > 1km^2)
        Irn = Idas > 1
        if Irn.sum() == 0:
            solve_method = 5 # No DA was provided, and no nearby flowlines exist
            return (np.nan, np.nan), solve_method
               
    # Compute errors based on distance away from provided coordinates
    Idist = np.ones(Idas.shape, dtype=np.bool)
    Idist[int((nrows-1)/2), int((ncols-1)/2)] = False
    # Idist = np.log(distance_transform_edt(Idist) + 1) / np.log(100)
    Idist = np.sqrt(distance_transform_edt(Idist))
    Idist = Idist/np.max(Idist)
    
    # If DA is provided, create error image that combines distance and DA differences
    if da is not None:        
        Iabs_err = np.abs(Idas-da)
        Iabs_err[np.logical_or(Idas>upper, Idas<lower)] = np.nan
    
        # If we don't have any valid pixels, can't use the provided drainage area
        if np.nansum(Iabs_err) == 0:
            da = None
        else: # Else we make a weighted error image using both distance and DA differences
            # Since we have useful DA information, we provide a small area around the 
            # center point that is unpenalized for distance. This accounts for very
            # small errors in location to avoid moving the point off a valid 
            # streamline.
            # Idist[nrows-8:nrows+9, ncols-8:ncols+9] = 0
    
            wt_da = 2
            wt_dist = 1
            Ierr = ((Iabs_err / da) * wt_da) + (Idist * wt_dist) 
            Ierr[~Irn] = np.nan
            solve_method = 2 # A DA was provided and a match was found on a nearby flowline that is within our certainty bounds
           
    # If no useful DA is provided, the nearest point along the stream network is 
    # chosen to map to. 
    if da is None:  
        Ierr = Idist
        Ierr[~Irn] = np.nan
        solve_method = 4 # A DA was not provided; we map to the nearest flowline (>1km^2)
    
    # Select the pixel in the drainage network that has the lowest error
    min_err = np.nanmin(Ierr)
    me_idx = np.where(Ierr==min_err)
    if len(me_idx[0]) > 1: # In the case of ties, choose the one that has the lower Idist error
        use_me_idx = np.argmin(Idist[me_idx[0], me_idx[1]])
        me_idx = np.array([[me_idx[0][use_me_idx]], [me_idx[1][use_me_idx]]])
    
    col_mapped = cr[0][0] + me_idx[1][0] - (ncols-1)/2 
    row_mapped = cr[0][1] + me_idx[0][0] - (nrows-1)/2 
    # col_mapped = cr[0][0]  
    # row_mapped = cr[0][1] 

    return (int(col_mapped), int(row_mapped)), solve_method # longitude, latitude


def dem_path(fdr_obj, cr_us, cr_ds=None, n_steps=None):
    """
    Walks along a flow direction grid from stpt to enpt. Returns a list of 
    pixels from stpt to enpt. If only the downstream point is provided, will
    trace the centerline. Walks from upstream to downstream.
    
    fdr_obj - flow direction object opened with gdal.Open(). Assumes flow
              direction symbology matches MERIT-Hydro:
                32 64 128
                16     1
                8   4  2
    cr_stpt - column, row of point to start walk
    cr_enpt - column, row of point to end walk
    """  
    imshape = (fdr_obj.RasterXSize, fdr_obj.RasterYSize)
        
    # Dictionary for looking up row, column additions when walking
    rowdict = {32: -1, 64: -1, 128: -1, 16: 0, 1: 0, 8: 1, 4: 1, 2: 1}
    coldict = {32: -1, 64: 0, 128: 1, 16: -1, 1: 1, 8: -1, 4: 0, 2: 1}
    
    us_pti = np.ravel_multi_index(cr_us, imshape)
    if cr_ds is not None:
        ds_pti = np.ravel_multi_index(cr_ds, imshape)
    else:
        ds_pti = None
    
    do_pt = [us_pti]
    ct = 1
    while 1:
        cr = np.unravel_index(do_pt[-1], imshape)
        fdr = fdr_obj.ReadAsArray(xoff=int(cr[0]), yoff=int(cr[1]), xsize=1, ysize=1)
        fdr = fdr[0][0]
        
        # If we reach a pixel with no direction
        if fdr not in rowdict.keys():
            break

        row = cr[1] + rowdict[fdr]
        col = cr[0] + coldict[fdr]
        
        do_pt.append(np.ravel_multi_index((col, row), imshape)) 
        
        # Halt if we've reached the end coordinate
        if ds_pti == do_pt[-1]:
            break
        
        # Halt if we've reached the number of steps
        ct = ct + 1
        if ct == n_steps:
            break

    colrow = np.unravel_index(do_pt, imshape)
        
    return (colrow[1], colrow[0])


def blob_to_polygon(I, ret_type='coords'):
    """
    Does not handle corner-connected cases. Make sure to reduce the blob
    to 4-connected before running. Also fill holes so that there are no
    8-connected holes inside the mask.
    
    ret_type : str
        One of 'coords' or 'pgon'.
    """

    def edges(x, y):
        
        pixedges = [((x-0.5, y-0.5), (x+0.5, y-0.5)),
                    ((x-0.5, y+0.5),(x+0.5, y+0.5)),
                    ((x-0.5, y-0.5),(x-0.5, y+0.5)),
                    ((x+0.5, y-0.5),(x+0.5, y+0.5))]
        
        return pixedges

    rp, _ = im.regionprops(I, props=['perimeter'])
    
    xs = rp['perimeter'][0][:,1]
    ys = rp['perimeter'][0][:,0]    
    
    # Map pixel corners to their coordinates
    xs4 = np.array([xs + 0.5, xs + 0.5, xs - 0.5, xs - 0.5]).flatten()
    ys4 = np.array([ys + 0.5, ys - 0.5, ys + 0.5, ys - 0.5]).flatten()
    xy4 = list(set(zip(xs4, ys4)))    
    
    # Get all valid edges (those surrounding all boundary pixels)
    all_edges = [edges(x,y) for x,y in zip(xs,ys)]
    all_edges = set([item for sublist in all_edges for item in sublist])
    
    # Determine the "connectivity" for each node of xy4 (max possible = 4)
    conn = []
    for xy in xy4:
        
        x = xy[0]
        y = xy[1]
        checkx = np.array([x+0.5, x+0.5, x-0.5, x-0.5], dtype=np.int)
        checky = np.array([y+0.5, y-0.5, y+0.5, y-0.5], dtype=np.int)
        
        rem = np.logical_or(np.logical_or(checkx < 0, checkx > I.shape[1] - 1), np.logical_or(checky < 0, checky > I.shape[0] - 1))
        checkx = checkx[~rem]
        checky = checky[~rem]
        
        conn.append(np.sum(I[checky, checkx]))
        
    poss_idcs = np.array(conn) < 4
    edge_coords = [xy4[i] for i, tf in enumerate(poss_idcs) if tf == True]
    conns = np.array(conn)[poss_idcs]
    
    walk = [edge_coords.pop()]
    dxs = [0, 0, 1, -1]
    dys = [1, -1, 0, 0]
    while edge_coords:
        cur_pt = walk[-1]
        # if cur_pt == (3.5, 67.5):
        #     break
        poss_walk = []
        poss_edges = []
        for dx, dy in zip(dxs, dys):
            pt = (cur_pt[0] + dx, cur_pt[1] + dy)
            if pt in edge_coords:
                poss_walk.append(edge_coords.index(pt))
                poss_edges.append((cur_pt, pt))
        
        valid_edges = [pe in all_edges or pe[::-1] in all_edges for pe in poss_edges]
        poss_walk = [pw for i, pw in enumerate(poss_walk) if valid_edges[i]==True]
        walkconns = conns[poss_walk]
        walkidx = np.argmin(walkconns)
        conns = np.delete(conns, poss_walk[walkidx])
        walk.append(edge_coords.pop(poss_walk[walkidx]))
        
    cols = np.array([w[0] for w in walk])
    rows = np.array([w[1] for w in walk])
    
    # plt.close('all')
    # plt.imshow(If)
    # for w in walk:
    #     plt.plot(w[0], w[1], 'ok')

    if ret_type == 'coords':
        return rows, cols
    elif ret_type == 'pgon':
        pgon = Polygon(zip(cols, I.shape[1]-rows))
        return pgon
    else:
        raise KeyError('Choose either "coords" or "pgon" as return types.')
