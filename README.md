# rabpro

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Anaconda badge](https://anaconda.org/jschwenk/rabpro/badges/version.svg)](https://anaconda.org/jschwenk/rabpro)

longitudinal river profiles, global watershed delineation, watershed stats

## Setup

### Software

```shell
conda env create -f environment.yml
source activate rabpro

# set use-feature to silence deprecation warning
# pip install --use-feature=in-tree-build . 
```

### Data

Locate the MERIT DEM "tile" of interest and run [Data/scripts/get_merit_dem.py](Data/scripts/get_merit_dem.py) with username and password arguments.

```shell
python Data/scripts/get_merit_dem.py n30w090 <username> <password>
```

Download Hydrobasins levels 1 and 12 using [Data/scripts/get_hydrobasins.py](Data/scripts/get_hydrobasins.py)

## Usage

### command line

```shell
python rabpro/run_rabpro.py
```

#### python

```python
import geopandas as gpd
import rabpro
from rabpro import utils
from rabpro.subbasin_stats import Dataset

# verify pathing setup "works"
# utils.get_rabpropath()
# utils.get_datapaths()

coords_file = gpd.read_file(r"tests/test_data/Big Blue River.geojson")
rpo = rabpro.profiler(coords_file)
rpo.delineate_basins() # requires hydrobasins levels 1 and 12
# name = "test"
# rpo.basins.to_file('Data/gaugebasin_shp/'+name+'.shp',driver='ESRI Shapefile')
rpo.elev_profile() # requires merit-dem (this ex. requires n30w090 [elv, fdr, upa, wth])
rpo.basin_stats([Dataset("JRC/GSW1_3/GlobalSurfaceWater", "occurrence")]) # requires Google Earth Engine access
```
