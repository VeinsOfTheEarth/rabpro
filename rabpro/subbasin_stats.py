# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:34:02 2018

@author: Jon
"""
import os
import pandas as pd
import numpy as np
import rasterstats_modified as rsm
import utils as ru


def main(cl_gdf, sb_inc_gdf, stYYYY, enYYYY, verbose=False):
        
    """ Compute subbasin statistics for each raster in control file """
    # For improved speed, rather than computing statistics for each subbasin,
    # fetch values for the subbasins, then compute statistics by combining
    # the values as we move downstream to the next subbasin
    
    # Dictionary for determining which rasters and statistics to compute
    control = get_controls(sb_inc_gdf.DA.values[0], stYYYY, enYYYY)

    # Open shapefile and csv for writing results    
    csv_df = pd.DataFrame(columns=['Basin number'])
    
    statct = 0 # used as an index to map output stats to shapefile attributes (shapefile names get too long--this index is used as the name)
    
    # For each raster
    for r in range(len(control['rID'])):
        
        # Reproject the subbasins vectors to match the raster CRS
        proj4crs = ru.get_proj4_crs(control['rastpath'][r])
        sb_inc_gdf = sb_inc_gdf.to_crs(proj4crs)
        
        # For each band
        for ib, b in enumerate(control['bands'][r]):
            
            # Don't use previously-interated raster values if it's the first iteration (there is nothing to recycle)
            if ib == 0:
                torecycle = None
            else:
                torecycle = recycle    
                
            # Get rasters clipped by subbasin polygons
            rID = control['rID'][r]
            rname = control['rname'][r]
                        
            if verbose:
                print('Computing subbasins stats for {}, band {} of {}...'.format(rID, ib+1, len(control['bands'][r])))
            if control['nodatavals'][r] is None:
                nodata = -9999.9
            else:
                nodata = control['nodatavals'][r]
            rsmout, recycle = rsm.zonal_stats(sb_inc_gdf, 
                                      control['rastpath'][r],
                                      maskraster = control['maskraster'][r],
                                      # nodata = control['nodatavals'][r],
                                      nodata = nodata,
                                      raster_out = True,
                                      recycled_arrays = torecycle,
                                      band = b)
            
            # Initialize stats dict
            statkeys = list(set(control['stats'][r] + ['count', 'sum', 'mean'])) # counts and sums are always returned
            stats = dict()
            for sk in statkeys:
                if sk == 'hyp':
                    stats['hypE'] = []
                    stats['hypA'] = []
                else:
                    stats[sk]=[]
            
            # Need to clear allvals, allvals_area if they exist from previous raster loop
            try: del allvals, all_frac_areas, weights
            except: pass
        
            # Compute stats on the rasters cumulatively (upstream to downstream)
            count = 0
            for ct in range(len(rsmout['rast'])):
                
                # Get the raster data for the ct-th subbasin (or ct-th feature in shapefile)
                areagrid = rsmout['areagrid'][ct] # area of each pixel (used for weighting statistics)
                clippedrast = rsmout['rast'][ct] # raster array of data
                clippedrastmask = rsmout['mask'][ct] # true where pixels are within the polygon
                frac_areas = rsmout['frac_area'][ct] # fraction of each pixel that is within the polygon
            
                # Get positions of non-masked pixels        
                valid_pix = np.where(clippedrastmask==True)
                
                # If we have no valid data in the region, append nan's to the stats
                if len(valid_pix[0]) == 0:
                    for s in stats:
                        stats[s].append(np.nan)
                    
                else:                
                    # Compute count using fractional pixel areas (in these cases the count will not be an integer)
                    count = count + np.sum(frac_areas[valid_pix])
                    
                    # Update allvals as we work through the subbasins
                    try:
                        allvals = np.concatenate([allvals, clippedrast[valid_pix]])
                        all_frac_areas = np.concatenate([all_frac_areas, frac_areas[valid_pix]])
                        weights = np.concatenate([weights, areagrid[valid_pix] * frac_areas[valid_pix]])
                        if 'hypE' in stats:
                            allvals_area = np.concatenate([allvals_area, areagrid[valid_pix]])
                    except:
                        allvals = np.array(clippedrast[valid_pix], dtype=np.float)
                        all_frac_areas = np.array(frac_areas[valid_pix], dtype=np.float)
                        weights = np.array(areagrid[valid_pix] * frac_areas[valid_pix], dtype=np.float)
                        if 'hypE' in stats:
                            allvals_area = areagrid[valid_pix]
        
                    stats['count'].append(count)
                    # Weighted sum
                    stats['sum'].append(np.dot(allvals, all_frac_areas)) 
    #                stats['sum'].append(float(allvals.sum()))
        
                    if 'mean' in stats:
                        try:
                            stats['mean'].append(np.average(allvals, weights=weights))
                        except:
                            pass
                    
                    if 'min' in stats:
                        stats['min'].append(float(allvals.min()))
                    
                    if 'max' in stats:
                        stats['max'].append(float(allvals.max()))
                    
                    if 'std' in stats: # Not quicker to manually compute std. Built-in method is faster.
                        try:
                            # This is a biased estimator of the weighted variance/std, but for large date counts should be fine
                            stats['std'].append(np.sqrt(np.average((allvals-stats['mean'][r])**2, weights=weights)))
                        except:
                            # If we didn't already compute mean...
                            stats['std'].append(np.sqrt(np.average((allvals-np.average(allvals, weights=weights))**2, weights=weights)))
        
                    if 'median' in stats:
                        stats['median'].append(float(np.median(allvals)))
                    
                    if 'range' in stats:
                        try:
                            rmin = stats['min'][-1]
                        except KeyError:
                            rmin = float(allvals.min())
                        try:
                            rmax = stats['max'][-1]
                        except KeyError:
                            rmax = float(allvals.max())
                        stats['range'].append(rmax-rmin)
                    
                    if 'hypE' in stats:
                        # Only compute hypsometric curve for the union of all the subbasins
                        if ct != len(rsmout['rast'])-1:
                            stats['hypE'].append(None)
                            stats['hypA'].append(None)
                        else:
                            E, A = hyps_curve(allvals, allvals_area)
                            stats['hypE'].append(list(E))
                            stats['hypA'].append(list(A))
                    
                    # Percentiles are a little different; need to read the percentile from the stat name
                    pcts = [p for p in statkeys if p[:3] == 'pct']
                    for p in pcts:
                        pval = float(p[-2:])
                        stats[p].append(float(np.nanpercentile(allvals, pval)))
                                    
            # Save stats to csv, shapefile. Shapefiles truncate at 10? character limit, I think.
            for sk in stats.keys():
                if control['date_st'][r] is None:
                    csv_fieldname = str(statct) + '.' + rname + '.' 'nodate.' + sk + '.' + control['units'][r]
                else: # Must edit the feature name to include the date
                    yr = np.arange(control['date_st'][r], control['date_en'][r] + 1)[ib]
                    csv_fieldname = str(statct) + '.' + rname + '.' + str(yr) + '.' + sk + '.' + control['units'][r]
                if len(stats[sk]) == 0:
                    csv_df[csv_fieldname] = np.nan
                else:
                    csv_df[csv_fieldname] = stats[sk]
                statct = statct + 1
            
        # If we have a time-varying raster, add a column of the temporal average once we've computed each of the individual time steps
        timestats = ['tavg', 'tstd', 'trange']
        if len(control['bands'][r])> 1:
            totalcols = len(stats.keys()) * len(control['bands'][r])
            startcol = csv_df.shape[1] - totalcols
            yrst = control['date_st'][r]
            yren = control['date_en'][r]
            for isk, sk in enumerate(stats.keys()):
                # Compute time-averaged column
                cols = np.arange(startcol + isk, csv_df.shape[1]-isk, len(stats.keys()))
                for ts in timestats:
                    csv_fieldname = str(statct) + '.' + rname + '.' + ts + str(yrst) + '-' + str(yren) + '.' + sk + '.' + control['units'][r]
                    if ts == 'tavg':
                        if np.sum(pd.isna((csv_df.iloc[:, cols].values))) == len(cols):
                             csv_df[csv_fieldname] = np.nan
                        else:
                             csv_df[csv_fieldname] = np.nanmean(csv_df.iloc[:, cols], axis=1)
                    elif ts == 'tstd':
                        if np.sum(pd.isna((csv_df.iloc[:, cols].values))) == len(cols):
                             csv_df[csv_fieldname] = np.nan
                        else:
                             csv_df[csv_fieldname] = np.nanstd(csv_df.iloc[:, cols], axis=1)
                    elif ts == 'trange':
                        if  np.sum(pd.isna((csv_df.iloc[:, cols].values))) == len(cols):
                             csv_df[csv_fieldname] = np.nan
                        else:
                             csv_df[csv_fieldname] = np.nanmax(csv_df.iloc[:, cols], axis=1) - np.nanmin(csv_df.iloc[:, cols], axis=1)
                        
                    statct = statct + 1                

    return csv_df


def get_controls(DAmax, stYYYY, enYYYY):
    
    """ Prepare paths and parameters for computing subbasin raster stats """
    
    # Determine if we need to use the coarse rasters    
    DAthresh = 50000 # km^2, analyses with any basin DAs larger than this value will use the coarse DEM. Testing on the Colville showed that using the coarse resulted in differences in less than 1% for all stats (most were well under 1%)
    if DAmax > DAthresh:
        usecoarse = True
    else:
        usecoarse = False

    # Load raster metadata file   
    datapaths = ru.get_datapaths()
    rast_df = pd.read_csv(datapaths['metadata'])
    
    # Rid the nans (they load because of column formatting extending beyond the last row of data)
    rast_df = rast_df[~pd.isna(rast_df.dataID)]
    
    # Create a control dictionary for computing subbasin stats
    controlkeys = ['rastpath', 'bands', 'maskraster', 'stats', 'nodatavals', 'rID', 'rname', 'units', 'resolution', 'date_st', 'date_en', 'EPSG']
    control = {ck:[] for ck in controlkeys}
    for i, row in rast_df.iterrows():
        
        # Skip some of the data entries -- they are not meant to be zonal-statted
        rID = row['dataID']
        
        if row['is_raster?'] == 'no':
            continue
        
        if rID in ['watermask_coarse', 'DEM_coarse', '_slope_coarse']:
            continue
        
        if row['skip?'] == 'yes':
            continue
        
        control['rID'].append(rID)
        control['rname'].append(row['nominally'])
        
        # Path to raster
        if rID == 'DEM':
            if usecoarse is True:
                rp = datapaths['DEM_coarse']
            else:
                rp = datapaths['DEM']
        elif rID == 'slope':
            if usecoarse is True:
                rp = datapaths['slope_coarse']
            else:
                rp = datapaths['slope']
        elif rID == 'watermask':
            if usecoarse is True:
                rp = datapaths['watermask_coarse']
            else:
                rp = datapaths['watermask']
        else:
            rp = os.path.normpath(datapaths[rID])
        control['rastpath'].append(rp)
        
        # Dates and bands
        if pd.isna(row['start:yyyy']):
            control['bands'].append([1])
            control['date_st'].append(None)
            control['date_en'].append(None)
        else:
            # Start and end dates - if requested dates are beyond those available for the raster, simply crop the dates to the available ones
            if stYYYY < row['start:yyyy']:
                control['date_st'].append(int(row['start:yyyy']))
            else:  
                control['date_st'].append(int(stYYYY))
                
            if enYYYY >= row['end:yyyy']:
                control['date_en'].append(int(row['end:yyyy']))
            else:
                control['date_en'].append(int(enYYYY))
            
            # Which bands to pull from raster?
            startband = int(control['date_st'][-1] - row['start:yyyy'] + 1)
            nbands = int(control['date_en'][-1] - control['date_st'][-1]) + 1
            control['bands'].append([int(x/row['dt (yr)']) for x in np.arange(startband, startband + nbands, dtype=np.int)])
       
        # Should we also mask water pixels?
        if row['water mask?'] == 'yes': 
            control['maskraster'].append(datapaths['watermask']) 
        else:
            control['maskraster'].append(None)
        
        # Which stats to compute?
        stats_tmp = row['stats']
        if pd.isna(stats_tmp):
            stats_tmp = []
        else:
            stats_tmp = [st.strip() for st in stats_tmp.split(',')]
        # Count and mean are always returned
        stats_tmp = stats_tmp + ['count', 'mean']
        stats_tmp = list(set(stats_tmp)) # uniquify list
        control['stats'].append(stats_tmp)
        
        # nodatavals?
        if pd.isna(row['nodata']) or row['nodata'].lower() == 'none':
            control['nodatavals'].append(None)
        else:
            control['nodatavals'].append(row['nodata'])
            
        # units
        control['units'].append(row['units'])
        
        # resolution
        control['resolution'].append(row['resolution'])
                    
    return control   


def hyps_curve(evals, areavals, nbins=20, rm_outliers=True):
    """ 
    Give elevation values and their corresponding areas, returns the 
    hypsometric curve computed at nbins intervals. If the elevations have some
    strange noise, use rm_outliers to eliminate the largest and smallest 1% of 
    elevation data. Outliers can distort the appearance of thehyposmetric curve.
    """

    # nbins is the number of ~equal-probability bins to divide the data into
    qbreaks = np.linspace(0, 100, nbins)

    if rm_outliers:
        # Remove outliers by eliminating the 1% extreme max/min elevations
        qbreaks[0] = 1.
        qbreaks[-1] = 99.       
        
    # Create equal-probability elevation bins
    breaks = np.percentile(evals, qbreaks)

    # Loop through bins and compute area associated with each elevation interval
    area = []
    for b in range(len(breaks)-1):
        if b == 0:
            area.append(np.sum(areavals[np.logical_and(evals >= breaks[b], evals <= breaks[b+1])]))
        else:
            area.append(np.sum(areavals[np.logical_and(evals > breaks[b], evals <= breaks[b+1])]))
    
    # Get the CDF of areas
    Aout = np.cumsum(area)
    
    # Elevations (less than or equal to)
    Eout = breaks[1:][::-1]
                    
    return Eout, Aout



