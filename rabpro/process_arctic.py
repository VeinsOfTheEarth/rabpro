# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:01:24 2018

@author: Jon
"""
import pandas as pd
import numpy as np
import os
import re
import rabpro

""" For running the Arctic SCREAM rivers """


#rbppath = '/home/jschwenk/RaBPro'
rbppath= r"X:\RaBPro"

meta = pd.read_excel(os.path.join(rbppath, 'ArcticScreamOutput', 'SupplementaryTable1_HighLatitudeErosionRates.xlsx'))

# List of rivers
rivers = list(set(meta['River'].values))

# For each river, find the start and end years
riv = dict()
riv['Names'] = rivers
riv['Dates'] = []
for r in rivers:
    rdf = meta[meta['River'].values==r]
    yrprs = np.empty((rdf.shape[0],2))
    yrprs[:,0] = rdf['Time interval start'].values
    yrprs[:,1] = rdf['Time interval end'].values
    yrprs = np.unique(yrprs, axis=0)
    riv['Dates'].append(yrprs)

# Prepare control files
rivfolder = os.path.join(rbppath, 'ArcticScreamOutput')

# Get all folders in the rivfolder (exclude any singular files)
allrivfolders = [f for f in os.listdir(rivfolder) if os.path.isfile(os.path.join(rivfolder,f)) is False]

# Control file template path
cftemplate = os.path.join(rbppath, 'control_file.csv')
cfsavebase = os.path.join(rbppath, 'control_files')

# Get name, start and end years; create control files
allcfs = []
for r in allrivfolders:
    m = re.search("\d", r)
    if m is None:
        rivname = r
        dates = riv['Dates'][riv['Names'].index(r)]
        start = int(dates[0][0])
        end = int(dates[0][1])
        
    else:
        rivname = r[0:m.start()]
        datestr = r[m.start():]
        datesplit = datestr.split('_')
        
        if int(datesplit[0]) > 50:
            start = int('19' + datesplit[0])
        else:
            start = int('20' + datesplit[0])
        
        if int(datesplit[1]) > 50:
            end = int('19' + datesplit[1])
        else:
            end = int('20' + datesplit[1])
    
    # Get the centerline and width filenames
    fold = os.path.join(rivfolder,r)
    clfile = [f for f in os.listdir(fold) if 'cl' in f.lower()]
    widfile = [f for f in os.listdir(fold) if 'width' in f.lower()]
    
    # Load control file, edit, and re-save
    cf = pd.read_csv(cftemplate)
    cf['run_name'][0] = r
    cf['cl_file'][0] = os.path.join(fold, clfile[0])
    cf['width_file'][0] = os.path.join(fold, widfile[0])
    cf['startYYYY'][0] = int(start)
    cf['endYYYY'][0] = int(end)
    
    
    cfoutpath = os.path.join(cfsavebase, r + '.csv') 
    allcfs.append(cfoutpath)
    cf.to_csv(cfoutpath, index=False)
    
    
    
# Run RaBPro
def rbpar(cfi):
    print(cfi)
#    # Don't process if already been processed
#    resultfile = os.path.join(rbppath,'Results', cfi.split('.')[0].split(os.sep)[-1],'subbasin_results.csv')
#    if os.path.isfile(resultfile):
#        tempdf = pd.read_csv(resultfile)
#        if tempdf.shape[1] > 3:
#            return
#    if "yenisei"  not in cfi.lower():
#        return
#    if cfi.split('.')[0].split(os.sep)[-1] in ['Colville', 'Indigirka', 'Kolyma', 'Koyukuk']: #  'Lena73_13', 'Lena73_99', 'Lena99_13', 'Noatak00_12', 'Noatak78_00', 'Noatak78_12', 'Ob'
#        return
    rabpro.main(cfi, rabpropath=rbppath)

cfbase = [f for f in os.listdir(cfsavebase) if '_sb' not in f.lower()]

from joblib import Parallel, delayed
Parallel(n_jobs=10)(delayed(rbpar)(os.path.join(cfsavebase,cfi)) for cfi in cfbase)

#for cfi in cfbase:
#    rbpar(os.path.join(cfsavebase,cfi))


    
import post_processing as pp
results_folder = r"X:\RaBPro\Results"
cfbase = [f for f in os.listdir(cfsavebase) if '_sb' not in f.lower()]
done_results = os.listdir(results_folder)
do = [c for c in cfbase if c.split('.')[0] in done_results]
#for d in do:
#    pp.main(os.path.join(cfsavebase,d))
    
# Combine all PP into single file
import pandas as pd
folders = [d.split('.')[0] for d in do]
ppfile = 'results_pp.csv'


for i,f in enumerate(folders):
    filepath = os.path.join(results_folder, f, ppfile)
    if i == 0:
        allresults = pd.read_csv(filepath)
    else:
        tmp_df = pd.read_csv(filepath)
        allresults = concat_ordered_columns([allresults, tmp_df])
        
allresults.to_csv(os.path.join(results_folder,'all_rivers.csv'), index=False)


def concat_ordered_columns(frames):
    columns_ordered = []
    for frame in frames:
        columns_ordered.extend(x for x in frame.columns if x not in columns_ordered)
    final_df = pd.concat(frames)    
    return final_df[columns_ordered]       


from matplotlib import pyplot as plt
plt.close('all')
plt.figure()
plt.loglog(allresults['Effective Width.m'], allresults['Long Term Discharge WBM.max.tavg.m^3/s'], '.')
plt.xlabel('Effective Width.m')
plt.ylabel('Long Term Discharge WBM.mean.tavg.m^3/s')


# Look for simple r2 between different variables
from scipy import stats
import numpy as np

xvar = 'Eroded Area per Val_Len.m/yr'
xlog = True

ylog = True

allvars = list(set(allresults.keys()))

vname = []
r2 = []
skipvars = ['River', 'Mean Erosion.m/yr', 'Mean Wnorm Erosion.chW/yr', 'Eroded Area per Val_Len.m/yr']
for v in allvars:
    if v in skipvars:
        continue
    
    x = allresults[xvar].values
    y = allresults[v].values

    if xlog is True:
        x = np.log10(x)
    if ylog is True: 
        y = np.log10(y)    

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    r2.append(r_value**2)
    vname.append(v)

plt.close('all')
plt.plot(r2)

vname[78]




