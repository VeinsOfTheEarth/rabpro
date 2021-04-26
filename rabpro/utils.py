# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:33:21 2018

@author: Jon
"""

import os, shutil
import shapely
import gdal, osr, ogr, osgeo
import subprocess
import geopandas as gpd
import numpy as np
import pandas as pd
import cv2
from skimage import measure
from pathlib import Path
import platform
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def get_rabpropath():
    """
    Returns a pathlib Path object of RaBPro's basepath.
    """
#    filepath = os.getcwd()
    import rabpro as rp
    filepath = os.path.dirname(rp.__file__)

    filepath = filepath.lower()
    st_idx = filepath.index('rabpro')
    en_idx = st_idx + len('rabpro')
    rabpropath = Path(filepath[:en_idx])
    
    return rabpropath


def get_datapaths():
    """
    Returns a dictionary of paths to all data that RaBPro uses.
    """
    basepath = get_rabpropath()
    datapath = basepath / 'Data'
    metadata_path = datapath / 'data_metadata.csv' 
    if metadata_path.is_file() is False:
        metadata_path = datapath / 'data_metadata_darwin.csv'
    metadata = pd.read_csv(metadata_path)
    rel_paths = [rp.replace('\\', os.sep) for rp in metadata.rel_path.values]
    dpaths = [str(basepath / Path(p)) for p in rel_paths]
    dnames = metadata.dataID.values
    datapaths = {dn:dp for dp, dn in zip(dpaths, dnames)}
    datapaths['metadata'] = metadata_path
    
    # Inserted for special case!
    datapaths['metadata'] =  r"X:\RaBPro\Results\Arctic Gages\data_metadata_Arctic_gages.csv"
    
    # Ensure that DEM virtual rasters are built
    if os.path.isfile(datapaths['DEM']) is False:
        print('Building virtual raster DEM from MERIT tiles...')
        build_vrt(os.path.dirname(os.path.realpath(datapaths['DEM'])), outputfile=datapaths['DEM'])
    
    if os.path.isfile(datapaths['DEM_coarse']) is False:
        print('Building coarse virtual raster DEM from MERIT tiles...')
        build_vrt(os.path.dirname(os.path.realpath(datapaths['DEM'])), outputfile=datapaths['DEM_coarse'], res=0.001)

    if os.path.isfile(datapaths['DEM_fdr']) is False:
        print('Building flow direction virtual raster DEM from MERIT tiles...')
        build_vrt(os.path.dirname(os.path.realpath(datapaths['DEM_fdr'])), outputfile=datapaths['DEM_fdr'])

    if os.path.isfile(datapaths['DEM_uda']) is False:
        print('Building drainage areas virtual raster DEM from MERIT tiles...')
        build_vrt(os.path.dirname(os.path.realpath(datapaths['DEM_uda'])), outputfile=datapaths['DEM_uda'])

    if os.path.isfile(datapaths['DEM_elev_hp']) is False:
        print('Building hydrologically-processed elevations virtual raster DEM from MERIT tiles...')
        build_vrt(os.path.dirname(os.path.realpath(datapaths['DEM_elev_hp'])), outputfile=datapaths['DEM_elev_hp'])
    
    if os.path.isfile(datapaths['DEM_width']) is False:
        print('Building width virtual raster from MERIT tiles...')
        build_vrt(os.path.dirname(os.path.realpath(datapaths['DEM_width'])), outputfile=datapaths['DEM_width'])

    return datapaths


def get_exportpaths(name, basepath=None, overwrite=False):
    """
    Returns a dictionary of paths for exporting RaBPro results. Also creates
    results folders when necessary.
    
    If overwrite is True, only the directory named "name" will be overwritten,
    not the entire 'Results' directory.
    """
    if basepath is None:
        basepath = get_rabpropath()
        results = basepath / 'Results'
    else:
        results = Path(basepath)
    
    # Make a results directory if it doesn't exist
    if results.exists() is False:
        results.mkdir(parents=True, exist_ok=True)
        
    namedresults = results / name

    # Make a results directory if it doesn't exist
    if namedresults.exists() is False:
        namedresults.mkdir(parents=True, exist_ok=True)
    elif overwrite is True:
        clear_directory(namedresults)
        
    # Results path dictionary
    exportpaths = {
            'base' : str(results),
            'basenamed' : str(namedresults),
            'stats' : str(namedresults / 'subbasin_stats.csv'),
            'subbasins' : str(namedresults / 'subbasins.json'),
            'subbasins_inc' : str(namedresults / 'subbasins_inc.json'),
            'centerline_results' : str(namedresults / 'centerline_results.json'),
            'dem_results' : str(namedresults / 'dem_flowpath.json')            
            }
    
    return exportpaths


def prepare_paths(controlfile, clear_results=True):
    """
    Returns a dictionary of paths required to run RaBPro
    """
    
    # Load the paths file and create paths dict
    # This assumes the control file is in RaBPro folder
    base_dir = get_rabpropath()
        
    # Get paths from control file
    paths = fetch_paths_from_file(controlfile)
    
    # Prepare output folder and files
    paths['outfolder'] = os.path.join(base_dir, 'Results', paths['run_name'])
    paths['cl_results'] = os.path.join(paths['outfolder'], 'cl_results.csv')
    paths['subbasin_results'] = os.path.join(paths['outfolder'], 'subbasin_results.csv')
    paths['results_pp'] = os.path.join(paths['outfolder'], 'results_pp.csv')
    paths['erosion_file'] = paths['width_file'].replace('T1','').replace('Width','Erode')
    
    
    paths['cl_geojson'] = os.path.join(paths['outfolder'], paths['run_name'] + '_cl.geojson')
    paths['sb_geojson'] = os.path.join(paths['outfolder'], paths['run_name'] + '_subbasins.geojson')
    paths['sb_inc_geojson'] = os.path.join(paths['outfolder'], paths['run_name'] + '_subbasins_inc.geojson')
    
    # Clear out the folder or create one if necessary
    if clear_results is True or os.path.exists(paths['outfolder']) is False:
        create_folder(paths['outfolder']) 
    
    # Prepare HydroBasin data paths
    paths['lev1'] = os.path.join(base_dir, 'Data', 'HydroBasins', 'level_one')
    paths['lev12'] = os.path.join(base_dir, 'Data', 'HydroBasins', 'level_twelve')
    
    # Prepare raster paths
    paths['rasters'] = os.path.join(base_dir, 'Data', 'Rasters')
    paths['watermask'] = os.path.join(paths['rasters'], 'WaterMask', 'JRCoccurrence.vrt')
    paths['watermask_coarse'] = os.path.join(paths['rasters'], 'WaterMask', 'JRCoccurrence_coarse.vrt')
    paths['topo_slope'] = os.path.join(paths['rasters'], 'Slope', '_slope.vrt')
    paths['topo_slope_coarse'] = os.path.join(paths['rasters'], 'Slope', '_slope_coarse.vrt')
    paths['rast_metadata'] = os.path.join(paths['rasters'], '_raster_metadata.csv')
    
    if clear_results:
        # Ensure coordinates are upstream -> downstream
        ensure_us_to_ds(paths['cl_file'])
        ensure_us_to_ds(paths['width_file'])
        
        # Initialize results csv
        cl_df = pd.read_csv(paths['cl_file'])
        keys = cl_df.keys()
        lati = [i for i,j in enumerate(keys) if j.lower()=='lat' or j.lower()=='latitude']
        loni = [i for i,j in enumerate(keys) if j.lower()=='lon' or j.lower()=='longitude']
        disti = [i for i,j in enumerate(keys) if 't1distance' in j.lower()] 
        if len(disti) == 0:
            disti = [i for i,j in enumerate(keys) if 'dist' in j.lower()] 
    
        
        results_save = pd.DataFrame(columns=['Lat', 'Lon', 'Dist'])
        results_save['Lat'] = cl_df[keys[lati]]
        results_save['Lon'] = cl_df[keys[loni]]
        results_save['Dist'] = cl_df[keys[disti]]
        
        results_save.to_csv(paths['cl_results'],  index=False)

    return(paths)


def parse_keys(gdf):
    """
    Attempts to interpret the column names of the input dataframe. In particular,
    looks for widths and distances along centerline.
    """
    keys = gdf.keys()
    parsed = {'distance':None, 'width':None}
    for k in keys:
        if 'distance' in k.lower():
            parsed['distance'] = k
        if 'width' in k.lower():
            parsed['width'] = k
    
    return parsed
    
    

def build_vrt(tilespath, clipper=None, extents=None, outputfile=None, nodataval=None, res=None, sampling='nearest', ftype='tif', separate=False):
        
    """
    Creates a text file for input to gdalbuildvrt, then builds vrt file with 
    same name. If output path is not specified, vrt is given the name of the 
    final folder in the path. 
    
    INPUTS: 
      tilespath - str:  the path to the file (or folder of files) to be clipped--
                        if tilespath contains an extension (e.g. .tif, .vrt), then
                        that file is used. Otherwise, a virtual raster will be 
                        built of all the files in the provided folder.
                        if filespath contains an extension (e.g. .tif, .vrt), 
                        filenames  of tiffs to be written to vrt. This list 
                        can be created by tifflist and should be in the same
                        folder
        extents - list: (optional) - the extents by which to crop the vrt. Extents
                        should be a 4 element list: [left, right, top, bottom] in
                        the ssame projection coordinates as the file(s) to be clipped
        clipper - str:  path to a georeferenced image, vrt, or shapefile that will be used
                        to clip
     outputfile - str:  path (including filename w/ext) to output the vrt. If 
                        none is provided, the vrt will be saved in the 'filespath'
                        path
            res - flt:  resolution of the output vrt (applied to both x and y directions)
       sampling - str:  resampling scheme (nearest, bilinear, cubic, cubicspline, lanczos, average, mode)
      nodataval - int:  (optional) - value to be masked as nodata
          ftype - str:  'tif' if buuilding from a list of tiffs, or 'vrt' if 
                        building from a vrt
    
    OUTPUTS:
        vrtname - str:  path+filname of the built virtual raster    
    """
    base, folder, file, ext = parse_path(tilespath)
    
    # Set output names  
    if outputfile is None:
        if clipper:
            cliptxt = '_clip'
        else:
            cliptxt = ''
        vrtname = os.path.join(base, folder, folder + cliptxt + '.vrt')
        vrttxtname = os.path.join(base, folder, folder + cliptxt + '.txt')
    else:
        vrtname = os.path.normpath(outputfile)
        vrttxtname = vrtname.replace('.vrt','.txt')
    
    # If a folder was given, make a list of all the text files
    if len(file) == 0: 
    
        filelist = []
        
        if ftype == 'tif':
            checktype = ('tif', 'tiff')
        elif ftype == 'hgt':
            checktype = ('hgt')
        elif ftype == 'vrt':
            checktype = ('vrt')
        elif ftype == 'nc':
            checktype = ('nc')
        else:
            raise TypeError('Unsupported filetype provided-must be tif, hgt, or vrt.')
      
        for f in os.listdir(tilespath):
            if f.lower().endswith(checktype): # ensure we're looking at a tif
                filelist.append(os.path.join(tilespath, f))
    else: 
        filelist = [tilespath] 
    
    if len(filelist) < 1:
        print('Supplied path for building vrt: {}'.format(filelist))
        raise RuntimeError('The path you supplied appears empty.')
                 
    # Clear out .txt and .vrt files if they already exist
    delete_file(vrttxtname)
    delete_file(vrtname)
    
    with open(vrttxtname, 'w') as tempfilelist:
        for f in filelist:
            tempfilelist.writelines('%s\n' %f)

    # Get extents of clipping raster
    if clipper:
        extents = raster_extents(clipper)

    # Build the vrt with input options
    callstring = ['gdalbuildvrt', '-overwrite',]
    
    if np.size(extents) == 4:
        stringadd = ['-te', str(extents[0]), str(extents[3]), str(extents[1]), str(extents[2])]
        for sa in stringadd:
            callstring.append(sa)
    
    if nodataval:
        stringadd = ["-srcnodata", str(nodataval)]
        for sa in stringadd:
            callstring.append(sa)
    
    if res:
        stringadd = ['-resolution', 'user', '-tr', str(res), str(res)]
        for sa in stringadd:
            callstring.append(sa)
        
    if sampling != 'nearest':
        stringadd = ['-r', sampling]
        for sa in stringadd:
            callstring.append(sa)
            
    if separate is True:
        callstring.append('-separate')
    
    stringadd = ['-input_file_list', vrttxtname, vrtname]
    for sa in stringadd:
        callstring.append(sa)
        
    # Make the call
    proc = subprocess.Popen(callstring, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=proc.communicate()

    # Check that vrt built successfully
    if len(stderr) > 3:
        raise RuntimeError('Virtual raster did not build sucessfully. Error: {}'.format(stderr))
    else:
        print(stdout)

    return vrtname


def raster_extents(raster_path):
    
    # Outputs extents as [xmin, xmax, ymin, ymax]                
    
    # Check if file is shapefile, else treat as raster
    fext = raster_path.split('.')[-1]
    if fext == 'shp' or fext == 'SHP':
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shapefile = driver.Open(raster_path, 0) # open as read-only
        layer = shapefile.GetLayer()
        ext = np.array(layer.GetExtent())
        extents = [ext[0], ext[1], ext[3], ext[2]]
    else:
        # Get the clipping raster lat/longs
        rast = gdal.Open(raster_path)
        cgt = rast.GetGeoTransform()
        clip_ULx = cgt[0]
        clip_ULy = cgt[3]
        clip_LRx = cgt[0] + cgt[1] * rast.RasterXSize
        clip_LRy = cgt[3] + cgt[5] * rast.RasterYSize
        extents = [clip_ULx, clip_LRx, clip_ULy, clip_LRy]
        
    return extents


def parse_path(path):

    """
    Parses a file or folderpath into: base, folder (where folder is the 
    outermost subdirectory), filename, and extention. Filename and extension
    are empty if a directory is passed.
    """
    
    if path[0] != os.sep and platform.system() != 'Windows': # This is for non-windows...
        path = os.sep + path
    
    
    # Pull out extension and filename, if exist
    if '.' in path:
        extension = '.' + path.split('.')[-1]
        temp = path.replace(extension,'')
        filename = temp.split(os.sep)[-1]
        drive, temp = os.path.splitdrive(temp)
        path = os.path.join(*temp.split(os.sep)[:-1])
        path = drive + os.sep + path
    else:
        extension = ''
        filename = ''
    
    # Pull out most exterior folder
    folder = path.split(os.sep)[-1]
    
    # Pull out base
    drive, temp = os.path.splitdrive(path)
    base = os.path.join(*temp.split(os.sep)[:-1])
    base = drive + os.sep + base
    
    return base, folder, filename, extension


def fetch_paths_from_file(csvpath):
    """
    Loads the path csv and builds a dictionary to the different file paths.
    """
    paths = dict()

    path_df = pd.read_csv(csvpath)
    keys = path_df.keys()
    for k in keys:
        paths[k] = path_df[k].values[0]

    return paths


def create_folder(folderpath):
    
    """
    Creates a folder or deletes all the files if the folder exists.
    """
    # Check that the folder exists first
    if os.path.exists(folderpath):
        for the_file in os.listdir(folderpath):
            file_path = os.path.join(folderpath, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): 
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    else:
        os.makedirs(folderpath)



def delete_file(file):
    # Deletes a file. Input is file's location on disk (path + filename)
    try:
        os.remove(file)
    except OSError:
        pass



def vrt_vals_from_lonlat(lonlats, vrt_obj, nrows=0, ncols=0):
    """
    Given an input list of lat, lon coordinates (latlons should be a Mx2 numpy
    array), returns the value of a vrt (or tiff) defined by vrt_obj at those
    coordinates. If you want to pull a neighborhood, nrows and ncols defines
    the number of rows and columns on each side of the pixel to pull. E.g. 
    setting nrows = ncols = 1 will return a 3x3 set; =2 returns 5x5, etc.
    
    Has been fixed for column/row confusion/switching.
    """
    nrows = int(nrows)
    ncols = int(ncols)
    
    # How many points do we have?
    nll = int(lonlats.size/2)
    
    if type(nrows) is int:
        nrows = np.ones(nll) * nrows
        
    if type(ncols) is int:
        ncols = np.ones(nll) * ncols
        
    if nll > 1 and len(nrows) != nll:
        raise ValueError('Check padding in vrt_vals_from_latlon.')  
    
    # Lat/lon to row/col
    if nll > 1:
        lons = lonlats[:,0]
        lats = lonlats[:,1]
    else:
        lons = lonlats[0]
        lats = lonlats[1]
    
    colrow = lonlat_to_xy(lons, lats, vrt_obj.GetGeoTransform())

    # Pull value and neighborhood from vrt at row/col
    vrtvals = []
    for i, cr in enumerate(colrow):
        if nrows[i] == ncols[i] == 0:
            vrtvals.append(vrt_obj.ReadAsArray(int(cr[0]-ncols[i]), int(cr[1]-nrows[i]), int(1+2*ncols[i]), int(1+2*nrows[i]))[0][0])
        else:
            vrtvals.append(vrt_obj.ReadAsArray(int(cr[0]-ncols[i]), int(cr[1]-nrows[i]), int(1+2*ncols[i]), int(1+2*nrows[i])))
    
    if len(vrtvals) == 1:
        vrtvals = vrtvals[0]
    
    return vrtvals, colrow



def lonlat_to_xy(lons, lats, gt):
    
       
    lats = np.array(lats)
    lons = np.array(lons)

    xs = ((lons - gt[0]) / gt[1]).astype(int)
    ys = ((lats - gt[3]) / gt[5]).astype(int)
    # xs = round((lons - gt[0]) / gt[1])
    # ys = round((lats - gt[3]) / gt[5])
    
    return np.column_stack((xs, ys))


def xy_to_coords(xs, ys, gt):
    """
    Transforms a set of x and y coordinates to their corresponding coordinates
    within a geotiff image.

    Arguments
    ---------
    (xs, ys) : (np.array(), np.array())
        Specifies the coordinates to transform.
    gt : tuple
        6-element tuple gdal GeoTransform. (uL_x, x_res, rotation, ul_y, rotation, y_res).
        Automatically created by gdal's GetGeoTransform() method.

    Returns
    ----------
    cx, cy : tuple
        Column and row indices of the provided coordinates.
    """

    cx = gt[0] + (xs + 0.5) * gt[1]
    cy = gt[3] + (ys + 0.5) * gt[5]

    return cx, cy


def lonlat_plus_distance(lon, lat, dist, bearing=0):
    """
    Returns the lon, lat coordinates of a point that is dist away (dist in km)
    and with a given bearing (in degrees, 0 is North). Uses a Haversine
    approximation.
    """
    
    bearing = np.radians(bearing)
    R = 6378.1 #Radius of the Earth
        
    latr = np.radians(lat) #Current lat point converted to radians
    lonr = np.radians(lon) #Current long point converted to radians
    
    lat_m = np.arcsin( np.sin(latr)*np.cos(dist/R) +
         np.cos(latr)*np.sin(dist/R)*np.cos(bearing))
    
    lon_m = lonr + np.arctan2(np.sin(bearing)*np.sin(dist/R)*np.cos(latr),
                 np.cos(dist/R)-np.sin(latr)*np.sin(latr))
    
    lat_m = np.degrees(lat_m)
    lon_m = np.degrees(lon_m)
    
    return lon_m, lat_m


def transform_coordinates(xs, ys, inputEPSG, outputEPSG):
    
    if inputEPSG == outputEPSG:
        return xs, ys
        
    # Create an ogr object of multipoints
    points = ogr.Geometry(ogr.wkbMultiPoint)
    for i in range(len(xs)):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(xs[i]), float(ys[i]))
        points.AddGeometry(point)
    
    # Create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)
    
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)
    
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # transform point
    points.Transform(coordTransform)
    
    xyout = np.array([0,0,0])
    for i in range(len(xs)):
        xyout = np.vstack((xyout, points.GetGeometryRef(i).GetPoints()))
    xyout = xyout[1:,0:2]
    
    return xyout[:,0], xyout[:,1]



def tifflist(folderpath, fileonly=0):
    # Returns a list of all the .tif or .TIF files in a folder
    folderpath = os.path.normpath(folderpath) + os.sep

    filelist = []
    for file in os.listdir(folderpath):
        if file.endswith('.tif') or file.endswith('.TIF') or file.endswith('.tiff'): # ensure we're looking at a tif
            filelist.append(folderpath + file)
        
    # If only the filename with extension should be returned (not the full filepath)
    if fileonly != 0:
        tifonly = []
        for file in filelist:
            tifonly.append(os.path.basename(file))
        filelist = tifonly
        
    return filelist



def largest_blobs(I, nlargest=1, action='remove', connectivity=2):
    """
    Returns a binary image with the nlargest blobs removed from the input
    binary image.
    """
    props = ['area', 'coords']
    rp = regionprops(I, props, connectivity=connectivity)
    areas = np.array(rp['area'])
    coords = rp['coords']
    # Sorts the areas array and keeps the nlargest indices
    maxidcs = areas.argsort()[-nlargest:][::-1]
    
    if action == 'remove':
        Ic =  np.copy(I)
        for m in maxidcs:
            Ic[coords[m][:,0],coords[m][:,1]] = False
    elif action == 'keep':
        Ic = np.zeros_like(I)
        for m in maxidcs:
            Ic[coords[m][:,0],coords[m][:,1]] = True
    else:
        print('Improper action specified: either choose remove or keep')
        Ic = I
        
    return Ic



def regionprops(I, props, connectivity=2):
    
    Ilabeled = measure.label(I, background=0, connectivity=connectivity)    
    properties = measure.regionprops(Ilabeled, intensity_image=I)

    out = {}
    # Get the coordinates of each blob in case we need them later
    if 'coords' in props or 'perimeter' in props:
        coords = [p.coords for p in properties]
    
    for prop in props:
        if prop == 'area':
            allprop = [p.area for p in properties]
        elif prop == 'coords':
            allprop = coords
        elif prop == 'centroid':
            allprop = [p.centroid for p in properties]
        elif prop == 'mean':
            allprop = [p.mean_intensity for p in properties]
        elif prop == 'perim_len':
            allprop = [p.perimeter for p in properties]
        elif prop == 'perimeter':
            perim = []
            for blob in coords:
                # Crop to blob to reduce cv2 computation time and save memory
                Ip, pads = crop_binary_coords(blob, npad=1)
                Ip = np.array(Ip, dtype='uint8')              
                             
                _, contours, _ = cv2.findContours(Ip, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # IMPORTANT: findContours returns points as (x,y) rather than (row, col)
                contours = contours[0]
                crows = []
                ccols = []
                for c in contours:
                    crows.append(c[0][1] + pads[1]) # must add back the cropped rows and columns
                    ccols.append(c[0][0] + pads[0])
                cont_np = np.transpose(np.array((crows,ccols))) # format the output
                perim.append(cont_np)
            allprop = perim
        elif prop == 'convex_area':
            allprop = [p.convex_area for p in properties]
        elif prop == 'eccentricity':
            allprop = [p.eccentricity for p in properties]
        elif prop == 'equivalent_diameter':
            allprop = [p.equivalent_diameter for p in properties]
        elif prop == 'major_axis_length':
            allprop = [p.major_axis_length for p in properties]

        else:
            print('{} is not a valid property.'.format(prop))

        out[prop] = np.array(allprop)
        
    return out



def crop_binary_coords(coords, npad=0):
    
    # Coords are of format [y, x]
    
    uly = np.min(coords[:,0]) - npad 
    ulx = np.min(coords[:,1]) - npad
    lry = np.max(coords[:,0]) + npad
    lrx = np.max(coords[:,1]) + npad

    I = np.zeros((lry-uly+1,lrx-ulx+1))    
    I[coords[:,0]-uly,coords[:,1]-ulx] = True

    pads = [ulx, uly, lrx, lry]
    return I, pads


def union_gdf_polygons(gdf, idcs, buffer=True):
    """
    Given an input geodataframe and a list of indices, return a shapely
    geometry that unions the geometries found at idcs into a single
    shapely geometry object.
    
    This function also buffers each polygon slightly, then un-buffers 
    the unioned polygon by the same amount. This is to avoid errors associated
    with floating-point round-off; see here: https://gis.stackexchange.com/questions/277334/shapely-polygon-union-results-in-strange-artifacts-of-tiny-non-overlapping-area
    """ 
    
    if buffer:
        from shapely.geometry import JOIN_STYLE
        # Buffer distance (tiny)    
        eps = .0001

    polys = []
    for i in idcs:     
        if buffer:
            polys.append(gdf.iloc[i].geometry.buffer(eps, 1, join_style=JOIN_STYLE.mitre))
#            polys.append(gdf.iloc[i].geometry.buffer(eps))
        else:
            polys.append(gdf.iloc[i].geometry)
    
    polyout = shapely.ops.unary_union(polys)
    
    if buffer:
#        polyout = polyout.buffer(-eps)
        polyout = polyout.buffer(-eps, 1, join_style=JOIN_STYLE.mitre)

    return polyout
    

def pts_df_to_gdf(df, crs={'init':'epsg:4326'}):
    """
    Given an input dataframe with columns labeled "lat", or "latitude" and "lon"
    or "longitdue" describing point locations, returns a pandas geodataframe.
    """
    keys = df.keys()
    lati = [i for i,j in enumerate(keys) if j.lower()=='lat' or j.lower()=='latitude']
    loni = [i for i,j in enumerate(keys) if j.lower()=='lon' or j.lower()=='longitude']
    geometry = [shapely.geometry.Point(xy) for xy in zip(df[keys[loni[0]]], df[keys[lati[0]]])]
    df = df.drop([keys[loni[0]], keys[lati[0]]], axis=1)
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    
    return gdf
        

def ensure_us_to_ds(clpath):
    
    cl_df = pd.read_csv(clpath)
    
    # Get distances; could have various column names
    keys = cl_df.keys()
    disti = [i for i,j in enumerate(keys) if j.lower()=='distance' or j.lower()=='dist']
    dists = cl_df[keys[disti]].values.ravel()
    
    # Ensure that coordinates are US-->DS order
    if np.sum(np.diff(dists) < 0) > 0:
        print('Coordinates not arranged in US->DS order; rearranging file: ' + clpath)
        sorted_idcs = dists.ravel().argsort()	
        cl_df = cl_df.loc[sorted_idcs]
        cl_df.to_csv(clpath)
        
        

def downsample_binary_image(I, newsize):
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
    Iout = np.zeros(newsize)
    Iout[rc_unique[0,:], rc_unique[1,:]] = 1
    
    return Iout


def get_proj4_crs(path):
    """
    Given a path to a georeferenced raster (.nc, .tif, etc.), returns its
    CRS in proj4 format.
    """
    gdobj = gdal.Open(path)
    srs = osgeo.osr.SpatialReference()
    srs.ImportFromWkt(gdobj.GetProjection())
    p4 = srs.ExportToProj4()
    
    return p4


def clear_directory(Path_obj):
    """
    Given a pathlib Path obj, clears all the contents of the directory. Does
    not remove the directory itself.
    """
    for child in Path_obj.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            clear_directory(child)
            
            
def haversine(lats, lons):
    """
    Computes distances between latitude and longitude pairs of points.
    
    Parameters
    ----------
    lats : numpy array (or interpretable by numpy)
        latitude values
    lons : numpy array (or interpretable by numpy)
        longitude values

    Returns
    -------
    np.array()
        Distances between each point defined by lats, lons.
    """

    R = 6372.8 * 1000
      
    dLat = np.radians(np.diff(lats))
    dLon = np.radians(np.diff(lons))
     
    lat1 = np.radians(lats[0:-1])
    lat2 = np.radians(lats[1:])
    
    a = np.sin(dLat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2) **2
    c = 2*np.arcsin(np.sqrt(a))

    return R * c


def validify_polygons(polys):
    """
    Hacky ways to validify a polygon. If can't be validified, returns the 
    original.

    Parameters
    ----------
    geom : list
        List of shapely polygons.

    Returns
    -------
    geomsv : list
        List of shapely polygons that have been attempted to validify.

    """
    geomsv = []
    for geom in polys:
    
        if type(geom) is Polygon:
            if geom.buffer(0).is_valid is True:
                geomsv.append(geom.buffer(0))
            else:
                geomsv.append(geom)
            
        if type(geom) is MultiPolygon:
            geomu = unary_union(geom)
            geomu = geomu.buffer(0)
            if geomu.is_valid is True:
                geomsv.append(geomu)
            else:
                geomsv.append(geom)
        
    return geomsv



""" Graveyard """
#def build_dem_vrt(dempaths, outpath, nodataval=0, res=None, sampling='nearest', name=''):
#
#    vrttxtname = os.path.join(outpath, name + '.txt')
#    vrtname = os.path.join(outpath, name + '.vrt')
#    
#    # Clear out .txt and .vrt files if they already exist
#    delete_file(vrttxtname)
#    delete_file(vrtname)
#    
#    filelist = []    
#    for dp in dempaths:
#        filelist = filelist + tifflist(dp)
#    
#    with open(vrttxtname, 'w') as tempfilelist:
#        for f in filelist:
#            tempfilelist.writelines('%s\n' %f)
#    
#    # Get SRTM resolution
#    if res is None:
#        srtm_temp = gdal.Open(filelist[-1])
#        gt = srtm_temp.GetGeoTransform()
#        xres = gt[1]
#        yres = abs(gt[5])
#    else:
#        xres = res
#        yres = res
#
#    callstring = ['gdalbuildvrt',
#                  '-tr', str(xres), str(yres),
#                  '-srcnodata', str(nodataval),
#                  '-overwrite',
#                  '-r',  sampling,
#                  '-input_file_list', vrttxtname, 
#                  vrtname]
#    proc = subprocess.Popen(callstring, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#    stdout,stderr=proc.communicate()
#    
#    if len(stderr) < 4:
#        print('VRT built successfully.')
#    else:
#        raise RuntimeError('VRT did not build successfully. Error: {}'.format(stderr))
#
#    return vrtname



