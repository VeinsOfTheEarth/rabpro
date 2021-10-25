# rabpro

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Anaconda badge](https://anaconda.org/jschwenk/rabpro/badges/version.svg)](https://anaconda.org/jschwenk/rabpro) [![build](https://github.com/jonschwenk/rabpro/actions/workflows/build.yaml/badge.svg)](https://github.com/jonschwenk/rabpro/actions/workflows/build.yaml)

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

Locate the MERIT DEM "tile" of interest and run the following command with username and password arguments:

```shell
rabpro download merit n30e150 <username> <password>
```

Download Hydrobasins levels 1 and 12:

```shell
rabpro download hydrobasins
```

## Usage

### command line

```shell
python rabpro/run_rabpro.py
```

#### python

```python
import geopandas as gpd
import rabpro
from rabpro.subbasin_stats import Dataset
import requests
import pandas as pd
import matplotlib.pyplot as plt

coords_file = gpd.read_file(r"tests/data/Big Blue River.geojson")

rpo = rabpro.profiler(coords_file)
rpo.delineate_basins()
rpo.basins.to_file('big_blue_river.gpkg',driver='GPKG')

url, task = rpo.basin_stats([Dataset("JRC/GSW1_3/GlobalSurfaceWater", "occurrence")])
r = requests.get(url)
with open("big_blue_river_gsw.csv", 'wb') as f:
    f.write(r.content)

bbr_gsw = pd.read_csv("big_blue_river_gsw.csv")
bbr_gdf = gpd.read_file('big_blue_river.gpkg')
bbr_gdf["mean"] = [float(x) for x in bbr_gsw["mean"]]

bbr_gdf.plot(column = "mean", legend = True)
plt.legend(title="GSW occurrence %", loc = (0.95, 1), frameon = False)
plt.show()
```

## Testing

```python
# file-based testing
python -m unittest tests/test.py

# object-based testing
python -m pytest
```

## Docs

```shell
cd docs && make html
```
