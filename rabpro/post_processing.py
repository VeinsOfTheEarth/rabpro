# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:26:07 2018

@author: Jon
"""

import re
import pandas as pd
import numpy as np
from rabpro import utils as ru


def main(controlfile, rabpropath=r"X:\RaBPro"):

#    controlfile = r"X:\RaBPro\control_files\Indigirka.csv"
    
    patru = ru.prepare_paths(controlfile, clear_results=False)
    
    # Load CSVs that we'll draw from to compile results
    cf = pd.read_csv(controlfile)
    cl = pd.read_csv(patru['cl_results'])
    sb = pd.read_csv(patru['subbasin_results'])
    er = pd.read_csv(patru['erosion_file'])
    wd = pd.read_csv(patru['width_file'])
    
    # Get basin interpolation values for each SCREAM segment
    basin_interp = get_interpolations(cl, er, sb)
            
    # Get indices for averaging over centerline points (i.e. which basin is each point in?)
    cl_bins = get_cl_seg_bins(cl, er)
    
    # Create results csv to put results into
    res = pd.DataFrame(columns=['River'])
    
    # Pull/compute some SCREAM variables
    dt = cf['endYYYY'].values - cf['startYYYY'].values # years
    seglens = er['SegmentLength(m)']
    sinuosity = er['Sinuosity']
    
    """ Add dependent variables """
    # Non-normalized erosion (m/yr)
    erosion1 = er['Mean_Erosion(m/yr)'].values
    res['Mean Erosion.m/yr'] = erosion1
    # Width-normalized erosion (chW/yr)
    erosion2 = er['WnormErode(chW/yr)'].values
    res['Mean Wnorm Erosion.chW/yr'] = erosion2
    # Erosion rate per valley length (m/yr)
    erosion3 = er['Total_Erosion(m^2)'].values / (dt * (seglens/sinuosity)) 
    res['Eroded Area per Val_Len.m/yr'] = erosion3
    # Width
    width = wd['Mean_width(m)'].values
    res['Mean Width.m'] = width
    effwidth = wd['Total_Effective_Width(m)'].values
    res['Effective Width.m'] = effwidth
    # Sinuosity
    sinu = er['Sinuosity'].values
    res['Sinuosity'] = sinu
    
    # Now set the river name once the DF has been initialized
    res['River'] = np.matlib.repmat(cf['run_name'].values, res.shape[0],1)
    
    """ Prepare driving/controlling variables """
    # DA 
    data = sb['DA.km^2'].values   
    DA = interp_data(basin_interp, data)
    res['Drainage Area.km^2'] = DA
    
    # Slope
    slopes = cl['slope_smooth'].values
    slopesOC = cl['slope_linOC'].values
    # Average them according to the map we created in cl_seg_bins
    avg_slope = []
    avg_slope_OC = []
    for c in cl_bins:
        avg_slope.append(np.mean(slopes[c]))
        avg_slope_OC.append(np.mean(slopesOC[c]))
    res['Avg Stream Slope.m/m'] = avg_slope
    res['Avg Stream SlopeOC.m/m'] = avg_slope_OC
    
    # Lat/lon
    res['Latitude.degrees'] = er['Latitude'].values
    res['Longitude.degrees'] = er['Longitude'].values
    
    """ Now add variables from sb_results.csv file """
    
    # Get data of interest from subbasin stats output
    varnames, spatialstats, temporalstats = get_vars(sb)
    
    for i in range(0,len(varnames)):
        vn = varnames[i]
        allss = spatialstats[i]
        allts = temporalstats[i]
        for ss in allss:
            if ss == 'count' or ss == 'pct95' or ss=='hypA': # We won't keep counts or 95th percentiles
                continue
            
            elif ss == 'hypE': # Compute hypsometric integral and approximate hypsometric integral
                E = get_data(sb, name = vn, stat='hypE', date='nodate')[0][-1][0].split()
                E = np.array([float(e.replace('[','').replace(']','').replace(',','')) for e in E])
                A = get_data(sb, name = vn, stat='hypA', date='nodate')[0][-1][0].split()
                A = np.array([float(a.replace('[','').replace(']','').replace(',','')) for a in A])
                # Normalize
                E = (E - np.min(E)) / (np.max(E) - np.min(E))
                A = (A - np.min(A)) / (np.max(A) - np.min(A))
                # Integrate
                HI = np.sum((E[0:-1] + E[1:])/2 * np.diff(A))
                temp = np.empty((res.shape[0],1))
                temp[:] = np.nan
                temp[-1] = HI
                # Compute approximate Hypsometric Integral (Pike and Wilson 1971)
                # First, make sure all the variables are available
                Emean = get_data(sb, name='Elevation', stat='mean', date='nodate')[0]
                Emin = get_data(sb, name='Elevation', stat='min', date='nodate')[0]
                try:
                    Emax = get_data(sb, name='Elevation', stat='pct99', date='nodate')[0]
                except:
                    Emax = get_data(sb, name='Elevation', stat='max', date='nodate')[0]

                HIapprox = (Emean - Emin)/(Emax - Emin)
                # Save the HI and HI approximation
                res['Hypsometric Integral.nounits'] = HI
                res['Hypsometric Int_approx.nounits'] = interp_data(basin_interp, HIapprox)
                continue
                
            for ts in allts:
                data, units = get_data(sb, name = vn, stat=ss, date=ts)
                data_i = interp_data(basin_interp, data)
                
                try:
                    txtidx = re.search("\d", ts).start()
                    ts_text = ts[0:txtidx]
                except:
                    ts_text = ts
            
                colname = vn + '.' + ss + '.' + ts_text + '.' + units
                
                res[colname] = data_i
    
    res.to_csv(patru['results_pp'], index=False)


""" For pulling data for reservoir """
#import pickle
#yrs = np.arange(1984,2016)
#data=[]
#name = 'Precipitation Rate GD' #'Precip (mswep)' # 
#name = 'Precip (mswep)' # 
#gdprecip = []
#gdyrs = []
#mswepprecip= []
#mswepyrs = []
#wbmQ = []
#wbmQyrs = []
#yravail = []
#gdsnow = []
#
#for y in yrs:
#    
#    datatemp, units = get_data(sb, name='Precipitation Rate GD', stat='mean', date=str(y))
#    gdprecip.append(datatemp[-1][0])
#    gdyrs.append(y)
#    
#    datatemp, units = get_data(sb, name='Precip (mswep)', stat='mean', date=str(y))
#    mswepprecip.append(datatemp[-1][0])
#    mswepyrs.append(y)
#    
#    try:
#        datatemp, units = get_data(sb, name='Long Term Discharge WBM', stat='max', date=str(y))
#        wbmQ.append(datatemp[-1][0])
#        wbmQyrs.append(y)
#    except:
#        pass
#
#    datatemp, units = get_data(sb, name='Snowmelt GD', stat='mean', date=str(y))
#    gdsnow.append(datatemp[-1][0])
#
#savepath = r"C:\Users\Jon\Desktop\Powell_Reservoir_for_Todd\rabpro_data.pkl"
#with open(savepath, 'wb') as f:
#    pickle.dump([gdprecip, gdyrs, mswepprecip, mswepyrs, wbmQ, wbmQyrs, gdsnow], f)



def get_vars(df):
    
    columns = list(df) # ordered list
    
    # Pull out the unique names
    varnames = set()
    for k in df.keys():
        ks = k.split('.')
        if len(ks) > 2:
            varnames.add(ks[1].strip())
    varnames = list(varnames)
    varnames = sort_varnames(varnames)
    
    # Get the column indices for each name
    col_name_ids = []
    for vn in varnames:
        varcols = []
        for ic,c in enumerate(columns):
            if vn in c:
                varcols.append(ic)
        col_name_ids.append(varcols)
        
    # Get the available stats for each variable (spatial stats)
    spatialstats = []
    for cid in col_name_ids:
        statstmp = set()
        for c in cid:
            statstmp.add(columns[c].split('.')[3])
        spatialstats.append(list(statstmp))
        
    # Get the available stats for each variable (temporal stats)
    tempstats = []
    for cid in col_name_ids:
        statstmp = set()
        for c in cid:
            possstat = columns[c].split('.')[2]
            if not possstat[0].isdigit():
                statstmp.add(possstat)
        tempstats.append(list(statstmp))
        
    return varnames, spatialstats, tempstats
            

def get_data(df, name='', stat='', date=''):
    
    cols = [k for k in df.keys() if len(k.split('.')) > 1 and name.strip() == k.split('.')[1].strip()]
    cols = [c for c in cols if stat in c.split('.')[3]]
#    cols = [c for c in cols if date in c.split('.')[2] and '-' not in c.split('.')[2]]
    cols = [c for c in cols if date == c.split('.')[2]] 
    units = cols[0].split('.')[-1]
    
    return df[cols].values, units
    
    

def get_interpolations(cl, er, sb):
    # Map subbasins to SCREAM bins
    # Find centerline lengtru to the beginning of each subbasin 
    cldists = cl['Dist'].values
    clbasins = cl['Basin number'].values
    basinbreaks = np.where(np.diff(clbasins) > 0)[0] + 1
    basin_end_cl = cldists[basinbreaks]
    basin_seg_cl_dists = np.diff(np.insert(basin_end_cl, [0], 0))
    
    # Find cente rline lengtru to the beginning of each SCREAM segment
    seg_cent_cl = er['T1Distance(m)'].values
#    seg_lens = er['SegmentLength(m)'].values
    
    # Map scream segment centerline points to their locations within the basin;
    # values are interpolated to indicate fraction of basin that should be included
    flowdists = sb['Flow Length.km'] * 1000
    # Compute correction factor for HydroBasins flow lengtru versus SCREAM's
    cfact = np.nanmean(flowdists[1:-1] / basin_seg_cl_dists[1:]) # average of ratio of stream length in HB to stream length via SCREAM -- used only to interpolate values for most-downstream basin
    basin_interp = []
    for seg in seg_cent_cl:
        
        # Find basin containing the scream segement center point
        bid = np.sum(seg > basin_end_cl)
        
        if bid == 0: # We are in the most-upstream basin
            basin_interp.append(1) # No reliable way to interpolate here...so just push points to same basin
        elif bid == sb.shape[0] - 1: # We're in the most-downstream basin
            extra = (seg - basin_end_cl[bid-1]) * cfact
            frac = extra / (flowdists[bid]) 
            if frac > 1:
                basin_interp.append(bid + 1)
            else:
                basin_interp.append(bid + frac)
        else:        
            extra = seg - basin_end_cl[bid-1] 
            frac = extra / (basin_end_cl[bid] - basin_end_cl[bid-1])
            basin_interp.append(bid + frac)
    basin_interp = [bi - 1 for bi in basin_interp] # Adjust to 0-indexing
    
    return basin_interp


def get_cl_seg_bins(cl, er):
    
    cldists = cl['Dist'].values
    seg_cent_cl = er['T1Distance(m)'].values
    scream_seg_endpoints = (seg_cent_cl[0:-1] + seg_cent_cl[1:])/2
    scream_seg_endpoints = np.insert(scream_seg_endpoints, [0], 0)
    scream_seg_endpoints = np.append(scream_seg_endpoints, cldists[-1]+1)

    cl_seg_bins = []
    for i in range(len(scream_seg_endpoints)-1):
        cl_seg_bins.append(np.where(np.logical_and(cldists>scream_seg_endpoints[i], cldists<scream_seg_endpoints[i+1]))[0])
    
    return cl_seg_bins


def interp_data(basin_interp, data):
    # Interpolate the data according to the map determined earlier
    data_interp = []
    for bi in basin_interp:
        frac = bi - int(bi)   
        if frac == 0:
            data_interp.append(data[bi])
        else:
            data_interp.append(data[int(bi)] + (data[int(bi) + 1] - data[int(bi)])*frac)
    
    
    return np.array(data_interp)


# This is the order in which statistics will be written to final results file;
# in general, the "more important" variables come earlier
def sort_varnames(namelist):
    sortedvars = [
        'Elevation',
        'Topographic Slope',
        'Long Term Discharge WBM',
        'Total Runoff GD',
        'Sediment Flux WBM',
        'Precip (mswep)',
        'N Days Precip',
        'N Days to Account for 75% of Annual Precip',
        'Number of Peaks Required to Account for 75% of Precip',
        'Precipitation Rate GD',
        'Rainfall Rate GD',
        'Fraction Precip as Snow GD',
        
        'Skin Temperature GD',
        'Freezethaw Days (skin) GD',
        'Degree Days (skin) GD',
        
        'Air Temperature GD',
        'Freezethaw Days (air) GD',
        'Degree Days (air) GD',
        'Permafrost',
        
        'NDVI',
        'Leaf Area Index',
        
        'Soil Thickness',
        'Clay, Subsurface',
        'Clay, Topsoil',
        'Silt, Subsurface',
        'Silt, Topsoil',
        'Sand, Subsurface',
        'Sand, Topsoil',
        'Gravel, Subsurface',
        'Gravel, Topsoil',
        
        'Storm Runoff GD',
        'Baseflow GD',
        'Snowmelt GD',
        'Snow Precipitation Rate GD',
        'Snow Rate GD',
        'Snow Depth Water Equivalent GD',
        
        'Soil Moisture Root Zone GD',
        'Soil Moisture 0-10cm GD',
        'Soil Moisture 10-40cm GD',
        'Soil Moisture 100-200cm GD',
        'Soil Moisture 40-100cm GD',
        
        'Direct Evap GD',
        'Transpiration GD',
        'Wind Speed GD',
        'Evapotranspiration GD',
        'Plant Canopy Surface Water GD',
        'Specific Humidity GD',
        'Canopy Evap GD',
        'Potential Evaporation GD',
          
        'Pressure GD',
        'Albedo GD',
        'Net Longwave Radiation GD',
        'Sensible Heat Flux GD',
        'Net Shortwave Radiation GD',
        'Downward Longwave Radiation GD',
        'Downward Shortwave Radiation GD',
        'Heat Flux (ground) GD',
        'Latent Heat Flux GD'
        ]
    # Remove accidental leading/trailing spaces
    sortedvars = [s.strip() for s in sortedvars]
    
    sortednamelist = []
    for s in sortedvars:
        if s in namelist:
            sortednamelist.append(s)
    
    if len(sortednamelist) != len(namelist):
        raise RuntimeError('The following variables cannot be sorted: {}.'.format(set(namelist)-set(sortednamelist)))
    
    return sortednamelist








    














## Compute temporal integrative variables (e.g. average across all years) - this is now done in subbasin stats
#allkeys = sb.keys()
#timestats = ['tavg', 'tstd', 'trange']
#timevars = pd.DataFrame()
#for v in allvars:
#    var = []
#    for k in allkeys:
#        if v in k:
#            var.append(k)
#    yr = sorted(list(set([vv.split('.')[2] for vv in var if vv.split('.')[2][0].isdigit()])))
#    stats = list(set([vv.split('.')[3] for vv in var]))
#    if len(yr) > 0:
#        for s in stats:
#            data, units = get_data(sb, name=v, stat=s, date=yr)
#            for ts in timestats:
#                savename = v + ' ' + s + ' ' + ts + str(yr[0]) + '-' + str(yr[-1]) + '.' + units[0]
#                if ts == 'tavg':
#                    timevars[savename] = np.nanmean(data, axis=1)
#                elif ts == 'tstd':
#                    timevars[savename] = np.nanstd(data, axis=1)
#                elif ts == 'trange':
#                    timevars[savename] = np.max(data, axis=1) - np.min(data, axis=1)





        





            

