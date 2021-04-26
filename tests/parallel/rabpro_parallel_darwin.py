# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:07:35 2020

@author: Jon
"""
# import sys
# sys.path.append('/projects/rabpro/Code/rabpro')

# import pandas as pd
# import rabpro as rp
# from joblib import Parallel, delayed
# import os
# import utils as ru


# def run_rabpro(coords, da, name, source, verbose=False):
    
#     print('Rabproing {}.'.format(name))
#     years = 'all'        

#     try:
#         rpo = rp.profiler(coords, da, name, force_merit=True, verbose=False)
        
#         if source in ['USGS', 'Hydat']:
#             search_radius = 500
#         else:
#             search_radius = 3000
#         rpo.delineate_basins(search_radius=search_radius)
#         rpo.elev_profile()
#         rpo.basin_stats(years=years)
        
#         # Set paths for export - delete files in each folder if they exist
#         basepath = os.path.normpath('/projects/rabpro/arctic_gage_analysis/Results')
#         namedpath = os.path.join(basepath, name)
#         if os.path.isdir(namedpath) is False:
#             os.makedirs(namedpath)
#         else:
#             for filename in os.listdir(namedpath):
#                 ru.delete_file(filename)
    
#         rpo.paths['stats'] = os.path.join(namedpath, 'subbasin_stats.csv')
#         rpo.paths['subbasins'] = os.path.join(namedpath, 'subbasins.json')
#         rpo.paths['centerline_results'] = os.path.join(namedpath, 'centerline.json')
#         rpo.paths['dem_results'] = os.path.join(namedpath, 'dem_flowpath.json')
#         rpo.export()
#     except:
#         print('Error in {}.'.format(name))

# path_gage_metadata = r"/projects/rabpro/arctic_gage_analysis/arctic_gage_metadata.csv"
# df = pd.read_csv(path_gage_metadata)
# df = df[0:4]
# coords = list(zip(df.provided_lat, df.provided_lon))
# das = df.provided_drainage_area.values
# names = df.index.values
# sources = df.source.values
# years = 'all'

# # i = 10
# # run_rabpro(coords=coords[i], da=das[i], name=str(names[i]), source=sources[i], verbose=True)

# Parallel(n_jobs=40, prefer="threads")(
#     delayed(run_rabpro)(coords=c, da=d, name=str(n), source=s, verbose=False) for (c, d, n, s) in zip(coords, das, names, sources))

# ### Local ###
import sys
sys.path.append(r'X:\RaBPro\Code\rabpro')

import pandas as pd
import rabpro as rp
# from joblib import Parallel, delayed
import os
import utils as ru

def run_rabpro_local(coords, da, name, source, verbose=False, force_merit=True, overwrite=False):
    
    print('Rabproing {}.'.format(name))
    years = 'all'        

    try:
        rpo = rp.profiler(coords, da, name, force_merit=force_merit, verbose=False)

        # Set paths for export - delete files in each folder if they exist
        basepath = os.path.normpath(r'X:\RaBPro\Results\Arctic Gages')
        namedpath = os.path.join(basepath, name)
        if os.path.isdir(namedpath) is False:
            os.makedirs(namedpath)
        else:
            for filename in os.listdir(namedpath):
                ru.delete_file(filename)

        rpo.paths['stats'] = os.path.join(namedpath, 'subbasin_stats.csv')
        rpo.paths['subbasins'] = os.path.join(namedpath, 'subbasins.json')
        rpo.paths['centerline_results'] = os.path.join(namedpath, 'centerline.json')
        rpo.paths['dem_results'] = os.path.join(namedpath, 'dem_flowpath.json')

        if overwrite is False:
            if os.path.isfile(rpo.paths['subbasins']) is True:
                print('{} already exists; skipping.'.format(name))
                return
            
        if source in ['USGS', 'Hydat']:
            search_radius = 500
        else:
            search_radius = 3000
        rpo.delineate_basins(search_radius=search_radius)
        rpo.elev_profile()
        rpo.basin_stats(years=years)
        rpo.export()
    except:
        print('Error in {}.'.format(name))

path_gage_metadata = r"C:\Users\Jon\Desktop\Research\InteRFACE\River Discharges\Mapping Gages\arctic_gage_metadata.csv"

df = pd.read_csv(path_gage_metadata)
# df = df[176:]
coords = list(zip(df.provided_lat, df.provided_lon))
das = df.provided_drainage_area.values
names = df.index.values
sources = df.source.values
years = 'all'

for i, row in df.iterrows():

    coords = (row['provided_lat'], row['provided_lon'])
    da = row['provided_drainage_area']
    name = str(i)
    source = row['source']
    years = 'all'
    
    if da > 10000:
        force_merit = False
    else:
        force_merit = True
    
    run_rabpro_local(coords, da, name, source, verbose=True, force_merit=force_merit)
    

# Parallel(n_jobs=4)(
#     delayed(run_rabpro_local)(coords=c, da=d, name=str(n), source=s, verbose=False) for (c, d, n, s) in zip(coords, das, names, sources))


# i = 67
# cds = coords[i]
# da = das[i]
# name = str(names[i])
# source = sources[i]

# rpo = rp.profiler(cds, da, name, force_merit=True, verbose=False)

# if source in ['USGS', 'Hydat']:
#     search_radius = 500
# else:
#     search_radius = 3000
# rpo.delineate_basins(search_radius=search_radius)
# rpo.elev_profile()
# rpo.basin_stats(years=years)

# Figure out how many errors there were
basepath = os.path.normpath(r'X:\RaBPro\Results\Arctic Gages')
files = os.listdir(basepath)
files = [f for f in files if f.isdigit() is True]
ct = 0
for f in files:
    folderfiles = os.listdir(os.path.join(basepath, f))
    if len(folderfiles) > 0:
        ct = ct + 1
        
        
""" Run centerlines provided by Feng """
def run_rabpro_centerlines(coords, da, name, source, verbose=False, force_merit=True, overwrite=False):
    
    print('Rabproing {}.'.format(name))
    years = 'all'        

    try:
        rpo = rp.profiler(coords, da, name, force_merit=force_merit, verbose=False)

        # Set paths for export - delete files in each folder if they exist
        basepath = os.path.normpath(r'X:\RaBPro\Results\Arctic Gages')
        namedpath = os.path.join(basepath, name)
        if os.path.isdir(namedpath) is False:
            os.makedirs(namedpath)
        else:
            for filename in os.listdir(namedpath):
                ru.delete_file(filename)

        rpo.paths['stats'] = os.path.join(namedpath, 'subbasin_stats.csv')
        rpo.paths['subbasins'] = os.path.join(namedpath, 'subbasins.json')
        rpo.paths['centerline_results'] = os.path.join(namedpath, 'centerline.json')
        rpo.paths['dem_results'] = os.path.join(namedpath, 'dem_flowpath.json')

        if overwrite is False:
            if os.path.isfile(rpo.paths['subbasins']) is True:
                print('{} already exists; skipping.'.format(name))
                return
            
        if source in ['USGS', 'Hydat']:
            search_radius = 500
        else:
            search_radius = 3000
        rpo.delineate_basins(search_radius=search_radius)
        rpo.elev_profile()
        rpo.export(what=['elevs'])
    except:
        print('Error in {}.'.format(name))


