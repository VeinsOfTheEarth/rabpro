# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:26:14 2020

@author: Jon
"""
import os
os.chdir(r'X:\RaBPro\Code\rabpro')

import rabpro as rp
import geopandas as gpd
import pandas as pd
import shapely
from matplotlib import pyplot as plt

# Test case 1: well-resolved centerline
latlon = gpd.read_file(r"X:\RaBPro\Results\Indigirka\Indigirka_cl.geojson")
blah = rp.profiler(latlon, verbose=True)
blah.delineate_basins()
blah.elev_profile()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\test1_basin.shp')
blah.merit_gdf.to_file(r'X:\RaBPro\Code\tests\output\test1_merit.shp')
blah.gdf.to_file(r'X:\RaBPro\Code\tests\output\test1_cl.shp')

# Test case 2: point with drainage area
latlon = (68.1,	35.1)
da = 204
blah = rp.profiler(latlon, da, name='blah', verbose=True)
blah.delineate_basins()
blah.elev_profile()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\test2_basin.shp')
blah.merit_gdf.to_file(r'X:\RaBPro\Code\tests\output\test2_merit.shp')
blah.gdf.to_file(r'X:\RaBPro\Code\tests\output\test2_cl.shp')

# Test case 3: point without drainage area
latlon = (68.1,	35.1)
blah = rp.profiler(latlon, name='blah', verbose=True)
blah.delineate_basins()
blah.elev_profile()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\test3_basin.shp')
blah.merit_gdf.to_file(r'X:\RaBPro\Code\tests\output\test3_merit.shp')
blah.gdf.to_file(r'X:\RaBPro\Code\tests\output\test3_cl.shp')

# Test case 4: a Grill centerline
latlon = gpd.read_file(r"C:\Users\Jon\Desktop\Research\InteRFACE\Feng\Centerlines\clip_2_GeneratePoints_003.shp")
blah = rp.profiler(latlon, verbose=True)
blah.delineate_basins()
blah.elev_profile()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\test4_basin.shp')
blah.merit_gdf.to_file(r'X:\RaBPro\Code\tests\output\test4_merit.shp')
blah.gdf.to_file(r'X:\RaBPro\Code\tests\output\test4_cl.shp')

# Test case 5: a Grill centerline but forced with merit
latlon = gpd.read_file(r"C:\Users\Jon\Desktop\Research\InteRFACE\Feng\Centerlines\clip_2_GeneratePoints_003.shp")
blah = rp.profiler(latlon, force_merit=True, verbose=True)
blah.delineate_basins()
blah.elev_profile()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\test5_basin.shp')
blah.merit_gdf.to_file(r'X:\RaBPro\Code\tests\output\test5_merit.shp')
blah.gdf.to_file(r'X:\RaBPro\Code\tests\output\test5_cl.shp')

# Test case 6: test point from shapefile
latlon = gpd.read_file(r"X:\RaBPro\Code\tests\input\testpoint1.shp")
da = 204
blah = rp.profiler(latlon, da=da, force_merit=False, verbose=True)
blah.delineate_basins()
blah.elev_profile()
blah.basin_stats()
blah.export()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\test5_basin.shp')
blah.merit_gdf.to_file(r'X:\RaBPro\Code\tests\output\test5_merit.shp')
blah.gdf.to_file(r'X:\RaBPro\Code\tests\output\test5_cl.shp')

# Debug case 1: problematic point (Birch Creek, #246)
# Solved
latlon = (61.33333,	-122.094)
da = 542
blah = rp.profiler(latlon, da=da, force_merit=True, name='Birch', verbose=True)
blah.delineate_basins()
blah.elev_profile()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\Birch_basin.shp')

# Debug case 2: MERIT basin polygon casues problems in zonal stats
# Solved
latlon = r"X:\RaBPro\Code\tests\input\Gauge_143.shp"
da = 1110
blah = rp.profiler(latlon, da=da, force_merit=True, verbose=True)
blah.delineate_basins()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\143_basin.shp')
blah.basin_stats()

# Debug case 3: MERIT geometry appears empty - pixels are found but returned
# polygon is empty. 
latlon = r"X:\RaBPro\Code\tests\input\Gauge_162.shp"
da = 1300
blah = rp.profiler(latlon, da=da, force_merit=True, verbose=True)
blah.delineate_basins()
blah.basins.to_file(r'X:\RaBPro\Code\tests\output\162_basin.shp')
blah.basin_stats()
