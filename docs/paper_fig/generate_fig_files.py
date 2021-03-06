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
import seaborn as sns

os.environ['RABPRO_DATA'] = r'X:\Data'

coords = (-18.0931, 177.5461)
da = 1395

f = profiler(coords, da=da)
f.delineate_basin(force_merit=True)
gdf_merit = f.watershed
f.delineate_basin(force_hydrobasins=True)
gdf_hb = f.watershed
f.elev_profile(dist_to_walk_km=5000)

path_base = r"X:\Research\RaBPro\Code\docs\paper_fig"

gdf_merit.to_file(path_base + 'basin_merit.gpkg', driver='GPKG')
gdf_hb.to_file(path_base + 'basin_hb.gpkg', driver='GPKG')
f.flowline.to_file(os.path.join(path_base, 'flowline.gpkg'), driver='GPKG')
target_pt = gpd.GeoDataFrame(geometry=[Point(coords[::-1])], crs=CRS.from_epsg(4326))
target_pt.to_file(path_base + 'coordinate.gpkg', driver='GPKG')

# Plot elevation profile
dists = f.flowline['Distance (m)'].values/1000
elevs = f.flowline['Elevation (m)'].values
zero_idx = np.where(dists==0)[0][0]

plt.close()
sns.set(rc={'figure.figsize':(6,3.2)})
sns.set_style('darkgrid')
ax = sns.lineplot(dists, elevs)
plt.plot(dists[zero_idx], elevs[zero_idx], 'o', markersize=10, color='black', markerfacecolor='white')
ax.set(xlabel='Distance (km)', ylabel='Elevation (m)')
plt.tight_layout()



