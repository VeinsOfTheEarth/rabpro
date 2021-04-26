# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import warnings # JPS
warnings.simplefilter(action='ignore', category=FutureWarning) # JPS to ignore the annoying future warnings (2 per run)

import numpy as np
from numpy import matlib
import warnings
from affine import Affine
from shapely.geometry import shape
from rasterstats.io import read_features, Raster # JPS
from rasterstats.utils import (rasterize_geom, get_percentile, check_stats, # JPS
                    remap_categories, key_assoc_val, boxify_points)
import ogr, gdal

""" NOTE CHANGES """
# I wanted to have the ability to supply a separate raster to use as an 
# additional mask, so I added some code to acheive that. The main complication
# is that this additional raster may not have perfect cell alignment (e.g.
# different resolutions), so it must be resampled to match the underlying
# raster on which statistics are computed. scipy has a nice "resize" function to
# do this, but it doesn't behave desirably when you are downscaling, so I also 
# included a function that manually downscales. Any changes to the code are 
# tagged with '#JPS'


def raster_stats(*args, **kwargs):
    """Deprecated. Use zonal_stats instead."""
    warnings.warn("'raster_stats' is an alias to 'zonal_stats'"
                  " and will disappear in 1.0", DeprecationWarning)
    return zonal_stats(*args, **kwargs)


def zonal_stats(*args, **kwargs):
    """The primary zonal statistics entry point.

    All arguments are passed directly to ``gen_zonal_stats``.
    See its docstring for details.

    The only difference is that ``zonal_stats`` will
    return a list rather than a generator."""
    return list(gen_zonal_stats(*args, **kwargs))


def downsample_binary_image(I, newsize): # JPS
    """
    Given an input binary image and a new size, this downsamples (i.e. reduces 
    resolution) the input image to the new size. A pixel is considered "on"
    in the new image if 'thresh' fraction of its area is covered by the 
    higher-res image. E.g. set 'thresh' to zero if you want the output image
    to be "on" everywhere at least a single pixel is "on" in the original
    image.
    """
    
    thresh = 0.05 # fraction that each larger pixel needs to contain of smaller pixels to consider it "on"
    
    # Get locations of all smaller pixels
    row,col = np.where(I>0)
    
    # Get the scaling factor in both directions
    rowfact = newsize[0]/I.shape[0]
    colfact = newsize[1]/I.shape[1]
    
    # Scale the row,col coordinates and turn them into integers
    rowcol = np.vstack((np.array(row * rowfact, dtype=np.uint16),np.array(col * colfact, dtype=np.uint16)))
    
    # Get the number of smaller pixels within each larger pixel
    rc_unique, rc_counts = np.unique(rowcol, axis=1, return_counts=True)
    
    # Filter out the large-pixel coordinates that don't contain enough area
    area_ratio = rowfact * colfact
    area_fracs = rc_counts * area_ratio
    rc_unique = rc_unique[:,np.where(area_fracs>=thresh)[0]]
    
    # Create the downsampled image
    Iout = np.zeros(newsize, dtype=np.bool)
    Iout[rc_unique[0,:], rc_unique[1,:]] = 1
    
    return Iout


def areagrid(rast_obj): #JPS
    """
    Given an input raster object created by rasterstats's Raster class, return
    a grid where each cell value represents the true area of the cell.
    IMPORTANT: this assumes a WGS84 spheroid--if this is not the case, the
    a and b values should be changed. For more info, see the following page:
    https://gis.stackexchange.com/questions/127165/more-accurate-way-to-calculate-area-of-rasters
    """ 
    
    # Gather some properties of the raster and its geolocation
    xres = rast_obj.affine[0]
#    ulX = rast_obj.affine[2]
    ulY = rast_obj.affine[5]
    rows = rast_obj.shape[0]
    cols = rast_obj.shape[1]
#    lrX = ulX + rast_obj.affine[0] * cols
    lrY = ulY + rast_obj.affine[4] * rows
    
    # Create latitude vector
    lats = np.linspace(ulY,lrY,rows+1)
    
    # These values correspond to the WGS84 ellipsoid
    a = 6378137
    b = 6356752.3142
    
    # Degrees to radians
    lats = lats * np.pi/180
    
    # Intermediate vars
    e = np.sqrt(1-(b/a)**2)
    sinlats = np.sin(lats)
    zm = 1 - e * sinlats
    zp = 1 + e * sinlats
    
    # Distance between meridians
    #        q = np.diff(longs)/360
    q = xres/360
    
    # Compute areas for each latitude in square km
    areas_to_equator = np.pi * b**2 * ((2*np.arctanh(e*sinlats) / (2*e) + sinlats / (zp*zm))) / 10**6
    areas_between_lats = np.diff(areas_to_equator)
    areas_cells = np.abs(areas_between_lats) * q
    
    area_grid = np.transpose(matlib.repmat(areas_cells,cols,1))
    
    return area_grid



def fractional_pixel_weights(fsrc, geom): # JPS
    
    """ 
    Returns a grid of the same size as fsrc, where each value represents the
    fraction of the cell that is filled by the polygon in geom.
    
    fsrc is a rasterstats-created object (I think rasterio).
    """
    
    gt = fsrc.affine
    xs = np.arange(gt[2], gt[2] +  gt[0]* (1 + fsrc.shape[1]), gt[0])
    ys = np.arange(gt[5], gt[5] +  gt[4]* (1 + fsrc.shape[0]), gt[4])
    
    # Convert geom into ogr geometry
    geom_ogr = ogr.CreateGeometryFromWkt(geom.to_wkt())

    # Loop through each grid cell, compute the intersecting area
    overlapping_areas = np.empty((len(ys)-1, len(xs)-1))
    for ix in range(len(xs)-1):
        xmin = xs[ix]
        xmax = xs[ix + 1]
        for iy in range(len(ys)-1):
            ymax = ys[iy]
            ymin = ys[iy + 1]
            
            # Intersecting area
            coords_wkt = "POLYGON ((" + str(xmin) + ' ' + str(ymax) + ', ' + str(xmax) + ' ' + str(ymax) + ', ' + str(xmax) + ' ' + str(ymin) + ', ' + str(xmin) + ' ' + str(ymin) + ', ' + str(xmin) + ' ' + str(ymax) + "))"
            polycell = ogr.CreateGeometryFromWkt(coords_wkt)
            
            overlapping_areas[iy, ix] = polycell.Intersection(geom_ogr).Area()
        
    # Ratio of overlapped area to pixel area
    frac_intersected = overlapping_areas / (abs(gt[0] * gt[4]))
    
    return frac_intersected




def gen_zonal_stats(
        vectors, 
        raster,
        maskraster = False, # JPS
        maskaffine = None, # JPS
        recycled_arrays = None, # JPS - this is to prevent re-doing computations when multiple bands from the same raster are interrogated. Adds a bit of complexity but saves a lot of runtime.
        layer=0,
        band=1,
        nodata=None,
        affine=None,
        stats=None,
        all_touched=False,
        categorical=False,
        category_map=None,
        add_stats=None,
        zone_func=None,
        raster_out=False,
        prefix=None,
        geojson_out=False, **kwargs):
    
    
    """ For Debugging and developing """
#    i=2
#    vectors = paths['sb_inc_geojson']
#    raster = control['rastpath'][i]
#    maskraster =  control['maskraster'][i]
#    nodata = control['nodatavals'][i]
#    stats = None
#    categorical = False
#    affine = None
#    maskaffine = None
#    raster_out = True
#    band = 1
#    layer = 0
#    all_touched = False
#    recycled_arrays = None


    """Zonal statistics of raster values aggregated to vector geometries.

    Parameters
    ----------
    vectors: path to an vector source or geo-like python objects

    raster: ndarray or path to a GDAL raster source
        If ndarray is passed, the ``affine`` kwarg is required.

    layer: int or string, optional
        If `vectors` is a path to an fiona source,
        specify the vector layer to use either by name or number.
        defaults to 0

    band: int, optional
        If `raster` is a GDAL source, the band number to use (counting from 1).
        defaults to 1.

    nodata: float, optional
        If `raster` is a GDAL source, this value overrides any NODATA value
        specified in the file's metadata.
        If `None`, the file's metadata's NODATA value (if any) will be used.
        defaults to `None`.

    affine: Affine instance
        required only for ndarrays, otherwise it is read from src

    stats:  list of str, or space-delimited str, optional
        Which statistics to calculate for each zone.
        All possible choices are listed in ``utils.VALID_STATS``.
        defaults to ``DEFAULT_STATS``, a subset of these.

    all_touched: bool, optional
        Whether to include every raster cell touched by a geometry, or only
        those having a center point within the polygon.
        defaults to `False`

    categorical: bool, optional

    category_map: dict
        A dictionary mapping raster values to human-readable categorical names.
        Only applies when categorical is True

    add_stats: dict
        with names and functions of additional stats to compute, optional

    zone_func: callable
        function to apply to zone ndarray prior to computing stats

    raster_out: boolean
        Include the masked numpy array for each feature?, optional

        Each feature dictionary will have the following additional keys:
        mini_raster_array: The clipped and masked numpy array
        mini_raster_affine: Affine transformation
        mini_raster_nodata: NoData Value

    prefix: string
        add a prefix to the keys (default: None)

    geojson_out: boolean
        Return list of GeoJSON-like features (default: False)
        Original feature geometry and properties will be retained
        with zonal stats appended as additional properties.
        Use with `prefix` to ensure unique and meaningful property names.

    Returns
    -------
    generator of dicts (if geojson_out is False)
        Each item corresponds to a single vector feature and
        contains keys for each of the specified stats.

    generator of geojson features (if geojson_out is True)
        GeoJSON-like Feature as python dict
    """
    stats, run_count = check_stats(stats, categorical)

    # Handle 1.0 deprecations
    transform = kwargs.get('transform')
    if transform:
        warnings.warn("GDAL-style transforms will disappear in 1.0. "
                      "Use affine=Affine.from_gdal(*transform) instead",
                      DeprecationWarning)
        if not affine:
            affine = Affine.from_gdal(*transform)
            maskaffine = Affine.from_gdal(*transform) # JPS

    cp = kwargs.get('copy_properties')
    if cp:
        warnings.warn("Use `geojson_out` to preserve feature properties",
                      DeprecationWarning)

    bn = kwargs.get('band_num')
    if bn:
        warnings.warn("Use `band` to specify band number", DeprecationWarning)
        band = bn
        
    feature_stats = dict()
    feature_stats['areagrid'] = []
    feature_stats['mask'] = []
    feature_stats['rast'] = []
    feature_stats['frac_area'] = []
    
    if recycled_arrays is None:
        
        recycle = dict()
        recycle['const_mask'] = []
        recycle['areagrid'] = []
        recycle['frac_area'] = []
        
        with Raster(raster, affine, nodata, band) as rast:
            # Get raster resolution
            res = gdal.Open(raster).GetGeoTransform()[1]
            
            features_iter = read_features(vectors, layer)
            for f, feat in enumerate(features_iter):
                
                geom = shape(feat['geometry'])
        
                if 'Point' in geom.type:
                    geom = boxify_points(geom, rast)
        
                geom_bounds = tuple(geom.bounds)
                
                # Read the clipped raster data
                fsrc = rast.read(bounds=geom_bounds)
                     
#                if f == 8:
#                    break

                # Check if we need to consider fractional pixels, or compute mask of pixel centers within polygon
                area_pixel = res**2
                avg_n_pixels = geom.area / area_pixel
                npix_thresh = 10 # hard-coded; this means that the area of the polygon should be greater than the area of npix_thresh pixels, or that the sample should contain at least 10 pixels, else we do a fractional pixel weighting
                # If we are using fractional-pixel areas, we don't need to know if pixel centers are within the polygon (mask_pc_in)
                if avg_n_pixels < npix_thresh:
                    frac_areas = fractional_pixel_weights(fsrc, geom)
                    mask_pc_in = np.ones(fsrc.array.shape, dtype=np.bool)
                else:
                    frac_areas = np.ones(fsrc.shape)
                    mask_pc_in = rasterize_geom(geom, like=fsrc, all_touched=all_touched)
                
                # Generate nodata mask
                isnodata = (fsrc.array == fsrc.nodata)
                    
                # Add nan mask (if necessary)
                if np.issubdtype(fsrc.array.dtype, np.floating) and \
                   np.isnan(fsrc.array.min()):
                    isnodata = (isnodata | np.isnan(fsrc.array))
    
                # JPS - If a mask raster is specified, load it
                # Masks from rasters are generally conservative; when resampling, 
                # if any portion of a mask raster pixel ends up in the resampled,
                # it is considered part of the mask (this can be changed by increasing
                # thresholds)
                if maskraster:
                    
                    # Fetch the masking image
                    mrast = Raster(maskraster, maskaffine, nodata=-1, band=1)
                    msrc = mrast.read(bounds=geom_bounds)
                    maskrast_array = np.array(msrc.array > 50) # mask wherever JRC occurrence is greater than 50%
                    
                    # Must resample the mask image as it is likely not exactly the
                    # same resolution as the raster being analyzed
                    scaleratio = np.sqrt(np.prod(maskrast_array.shape) / np.prod(fsrc.array.shape))
                    
                    # If we're upsampling (significantly), use scipy's ndimage.zoom. Else use
                    # custom downsampling function.
                    if scaleratio > 2:
                        maskrast_array = downsample_binary_image(maskrast_array, isnodata.shape)
                    else:
                        import skimage.transform as st 
                        maskrast_array = st.resize(maskrast_array, isnodata.shape, mode='constant')
                        maskrast_array = maskrast_array>0.01 
                        maskrast_array = ~maskrast_array # we want ones to be data we let through
                else:
                    maskrast_array = np.ones(isnodata.shape, dtype=np.bool)
                             
                # Create and store area grid (areas in km^2)
                ag = areagrid(fsrc)
                
                # Create mask
                const_mask = np.ones(fsrc.array.shape, dtype=np.bool) # this mask doesn't change for different time steps of a raster
                const_mask[mask_pc_in == False] = False
                const_mask[maskrast_array == False] = False

                mask = np.copy(const_mask)
                mask[isnodata==True] = False
                
                # Store areagrid
                feature_stats['areagrid'].append(ag)
                # Store mask
                feature_stats['mask'].append(mask)
                # Store raster
                feature_stats['rast'].append(fsrc.array)
                # Store fractional pixel weights
                feature_stats['frac_area'].append(frac_areas)
                
                # Return recycleable arrays in case other bands of the same raster are to be analyzed
                recycle['const_mask'].append(const_mask)
                recycle['areagrid'].append(ag)
                recycle['frac_area'].append(frac_areas)
            
            return feature_stats, recycle
                
    else: # If firstloop is not true (i.e. we're going to re-use previous calculations to save time)
        with Raster(raster, affine, nodata, band) as rast:
            features_iter = read_features(vectors, layer)
            for f, feat in enumerate(features_iter):
                
                geom = shape(feat['geometry'])
        
                if 'Point' in geom.type:
                    geom = boxify_points(geom, rast)
        
                geom_bounds = tuple(geom.bounds)
        
                fsrc = rast.read(bounds=geom_bounds)
        
                # nodata mask
                isnodata = (fsrc.array == fsrc.nodata)
                    
                # add nan mask (if necessary)
                if np.issubdtype(fsrc.array.dtype, np.floating) and \
                   np.isnan(fsrc.array.min()):
                    isnodata = (isnodata | np.isnan(fsrc.array))
                
                # Create full mask
                mask = recycled_arrays['const_mask'][f]
                mask[isnodata==True] = False
                
                # Create and store area grid (areas in km^2)
                feature_stats['areagrid'].append(recycled_arrays['areagrid'][f])
                # Store mask
                feature_stats['mask'].append(mask)
                # Store raster
                feature_stats['rast'].append(fsrc.array)
                # Carry over frac_area
                feature_stats['frac_area'].append(recycled_arrays['frac_area'][f])
               
            return feature_stats, recycled_arrays

            


#            # execute zone_func on masked zone ndarray
#            if zone_func is not None:
#                if not callable(zone_func):
#                    raise TypeError(('zone_func must be a callable '
#                                     'which accepts function a '
#                                     'single `zone_array` arg.'))
#                zone_func(masked)
#
#            if masked.compressed().size == 0:
#                # nothing here, fill with None and move on
#                feature_stats = dict([(stat, None) for stat in stats])
#                if 'count' in stats:  # special case, zero makes sense here
#                    feature_stats['count'] = 0
#            else:
#                if run_count:
#                    keys, counts = np.unique(masked.compressed(), return_counts=True)
#                    pixel_count = dict(zip([np.asscalar(k) for k in keys],
#                                           [np.asscalar(c) for c in counts]))
#
#
#                if categorical:
#                    feature_stats = dict(pixel_count)
#                    if category_map:
#                        feature_stats = remap_categories(category_map, feature_stats)
#                else:
#                    feature_stats = {}
#
#                if 'min' in stats:
#                    feature_stats['min'] = float(masked.min())
#                if 'max' in stats:
#                    feature_stats['max'] = float(masked.max())
#                if 'mean' in stats:
#                    feature_stats['mean'] = float(masked.mean())
#                if 'count' in stats:
#                    feature_stats['count'] = int(masked.count())
#                # optional
#                if 'sum' in stats:
#                    feature_stats['sum'] = float(masked.sum())
#                if 'std' in stats:
#                    feature_stats['std'] = float(masked.std())
#                if 'median' in stats:
#                    feature_stats['median'] = float(np.median(masked.compressed()))
#                if 'majority' in stats:
#                    feature_stats['majority'] = float(key_assoc_val(pixel_count, max))
#                if 'minority' in stats:
#                    feature_stats['minority'] = float(key_assoc_val(pixel_count, min))
#                if 'unique' in stats:
#                    feature_stats['unique'] = len(list(pixel_count.keys()))
#                if 'range' in stats:
#                    try:
#                        rmin = feature_stats['min']
#                    except KeyError:
#                        rmin = float(masked.min())
#                    try:
#                        rmax = feature_stats['max']
#                    except KeyError:
#                        rmax = float(masked.max())
#                    feature_stats['range'] = rmax - rmin
#
#                for pctile in [s for s in stats if s.startswith('percentile_')]:
#                    q = get_percentile(pctile)
#                    pctarr = masked.compressed()
#                    feature_stats[pctile] = np.percentile(pctarr, q)
#
#            if 'nodata' in stats:
#                featmasked = np.ma.MaskedArray(fsrc.array, mask=np.logical_not(rv_array))
#                feature_stats['nodata'] = float((featmasked == fsrc.nodata).sum())
#
#            if add_stats is not None:
#                for stat_name, stat_func in add_stats.items():
#                    feature_stats[stat_name] = stat_func(masked)
#
#            if raster_out:
#                feature_stats['mini_raster_array'] = masked
#                feature_stats['mini_raster_affine'] = fsrc.affine
#                feature_stats['mini_raster_nodata'] = fsrc.nodata
##                feature_stats['maskraster'] = maskrast_array
#
#            if prefix is not None:
#                prefixed_feature_stats = {}
#                for key, val in feature_stats.items():
#                    newkey = "{}{}".format(prefix, key)
#                    prefixed_feature_stats[newkey] = val
#                feature_stats = prefixed_feature_stats
#
#            if geojson_out:
#                for key, val in feature_stats.items():
#                    if 'properties' not in feat:
#                        feat['properties'] = {}
#                    feat['properties'][key] = val
#                yield feat
#            else:
#                yield feature_stats
