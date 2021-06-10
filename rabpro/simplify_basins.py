# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:13:34 2021

@author: Jon
"""
import os
import geopandas as gpd

path_basins = r'X:\RaBPro\Data\Gage Basins\Gage Data'

overwrite = False
simptol = 0.001 # tolerance for simplifying basin polygons

gage_ids = os.listdir(path_basins)
for gid in gage_ids:
    path_gage = os.path.join(path_basins, str(gid))
    path_gage_basin = os.path.join(path_gage, 'subbasins.json')
    path_gage_basin_simp = os.path.join(path_gage, 'subbasins_simp.json')
    
    # Check if subbasins file exists
    if os.path.isfile(path_gage_basin) is True:
        
        # Check if simplified version already exists
        if os.path.isfile(path_gage_basin_simp) is True:
            if overwrite is False:
                simplify = False
            elif overwrite is True:
                simplify = True
        else:
            simplify = True
            
        if simplify:
            gdf = gpd.read_file(path_gage_basin)
            gdf_s = gdf.simplify(simptol) 
            try:
                gdf_s.to_file(path_gage_basin_simp, driver='GeoJSON')
                print('simplified {}'.format(gid))
            except:
                print('Empty GDF for {}'.format(gid))
            