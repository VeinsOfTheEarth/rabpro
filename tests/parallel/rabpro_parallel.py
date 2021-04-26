# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:07:35 2020

@author: Jon
"""
import sys
sys.path.append(r'X:\RaBPro\Code\rabpro')

import pandas as pd
import rabpro as rp
from joblib import Parallel, delayed


def run_rabpro(coords, da, name, source, verbose=False):
    
    years = 'all'        

    rpo = rp.profiler(coords, da, name, force_merit=True, verbose=False)
    
    if source in ['USGS', 'Hydat']:
        search_radius = 500
    else:
        search_radius = 3000
    rpo.delineate_basins(search_radius=search_radius)
    rpo.elev_profile()
    rpo.basin_stats(years=years)
    rpo.export()


path_gage_metadata = r"C:\Users\Jon\Desktop\Research\InteRFACE\River Discharges\Mapping Gages\arctic_gage_metadata.csv"

df = pd.read_csv(path_gage_metadata)
df = df[0:4]
coords = list(zip(df.provided_lat, df.provided_lon))
das = df.provided_drainage_area.values
names = df.index.values
sources = df.source.values
years = 'all'

run_rabpro(coords[0], das[0], 'testpar', verbose=True)

Parallel(n_jobs=4, prefer="threads")(
    delayed(run_rabpro)(coords=c, da=d, name=str(n), source=s, verbose=False) for (c, d, n, s) in zip(coords, das, names, sources))

