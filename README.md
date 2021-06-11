# rabpro

longitudinal river profiles, global watershed delineation, watershed stats

## Setup

```shell
conda env create -f environment.yml
source activate rp
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
import utils

# verify pathing setup "works"
# utils.get_rabpropath()
# utils.get_datapaths()

coords_file = gpd.read_file(r"../tests/input/Big Blue River.geojson")
rpo = rabpro.profiler(coords_file)
rpo.delineate_basins() # requires hydrobasins levels 1 and 12
# name = "test"
# rpo.basins.to_file('Data/gaugebasin_shp/'+name+'.shp',driver='ESRI Shapefile')
rpo.elev_profile() # requires merit dem
```