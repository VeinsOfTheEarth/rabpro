import geopandas as gpd
import rabpro
from rabpro.subbasin_stats import Dataset
from shapely.geometry import box
import pandas as pd


def test_basin_stats():
    coords_file = gpd.read_file(r"tests/data/Big Blue River.geojson")
    gdf = gpd.GeoDataFrame(
        {"idx": [1], "geometry": [box(*coords_file.total_bounds)]}, crs="EPSG:4326"
    )

    data, task = rabpro.subbasin_stats.main(
        gdf,
        [
            Dataset(
                "JRC/GSW1_3/GlobalSurfaceWater",
                "occurrence",
                stats=["min", "max", "range", "std", "sum", "pct50", "pct3"],
            )
        ],
        test=True,
    )

    res = pd.DataFrame(data["features"][0]["properties"], index=[0])

    assert float(res["mean"]) > 0
    assert res.shape[1] == 9
