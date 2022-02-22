# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:23:03 2022

@author: 318596
"""

from rabpro.core import profiler
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point
from pyproj import CRS

os.environ['RABPRO_DATA'] = r'X:\Data'

coords = (-18.0931, 177.5461)
da = 1395

f = profiler(coords, da=1395)
f.delineate_basin(force_merit=True)
gdf_merit = f.watershed
f.delineate_basin(force_hydrobasins=True)
gdf_hb = f.watershed
f.elev_profile(dist_to_walk_km=25)

path_base = "docs/paper_fig/"

gdf_merit.to_file(path_base + 'basin_merit.gpkg', driver='GPKG')
gdf_hb.to_file(path_base + 'basin_hb.gpkg', driver='GPKG')
f.flowline.to_file(path_base + 'flowline.gpkg', driver='GPKG')
target_pt = gpd.GeoDataFrame(geometry=[Point(coords[::-1])], crs=CRS.from_epsg(4326))
target_pt.to_file(path_base + 'coordinate.gpkg', driver='GPKG')
