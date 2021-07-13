# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:30:07 2021

@author: Jon

A simple example of RaBPro's functionality, except for computing basin
statistics.
"""
import sys, os
import rabpro

# Specify a point within our test DEM region
coords = (32.97287, -88.15829)

# We can also specify a drainage area for testing; looking at the test
# region MERIT_upa grid, a value of 111,548 should be close
da = 18680

# Boot up the profiler; note that providing the drainage area (da) is optional
# We also choose to set force_merit to True, which will use MERIT data to
# perform basin delineation.
rpo = rabpro.profiler.profiler(coords, name='basic_test', da=da, force_merit=True)
#path_data='/Users/talzussman/Documents/School Files/LANL/rabpro/Data'
# Compute the watershed for this point - this can take a few minutes since
# we've chosen a rather large basin
rpo.delineate_basins() # requires merit n30w090 [elv, fdr, upa, MERIT103]

# Compute the river elevation profile
rpo.elev_profile() # requires merit-dem (this ex. requires n30w090 [elv, fdr, upa, wth])

# The basin geometry is stored in a geopandas GeoDataframe
# We access it through the rpo object
basins = rpo.basins
# Can export it as well
rpo.basins.to_file(rpo.paths['subbasins'], driver='GeoJSON')

data = rabpro.subbasin_stats.Dataset("JRC/GSW1_3/GlobalSurfaceWater", "occurrence", stats=['min', 'max', 'range', 'std', 'sum', 'pct50', 'pct3'])

# Finally, we can compute basin statistics - this will not work without
# auxiliary datasets. We could compute topographic stats actually.
rpo.basin_stats([data]) # requires "mswep" Beck et al 2019?
