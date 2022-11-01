import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import rabpro
from rabpro import utils
from rabpro.basin_stats import Dataset


# coords_file = gpd.read_file(r"tests/data/Big Blue River.geojson")
# total_bounds = coords_file.total_bounds
total_bounds = np.array([-85.91331249, 39.42609864, -85.88453019, 39.46429816])
gdf = gpd.GeoDataFrame({"idx": [1], "geometry": [box(*total_bounds)]}, crs="EPSG:4326")

# Ensure the gee metadata is updated
_ = utils.get_datapaths(update_gee_metadata=True)


def clean_res(feature):
    res = pd.DataFrame(feature["properties"], index=[0])
    res["id"] = feature["id"]
    return res


def test_customreducer():
    def asdf(feat):
        return feat.getNumber("max")

    urls, task = rabpro.basin_stats.compute(
        [Dataset("JRC/GSW1_4/YearlyHistory", "waterClass", stats=["max"])],
        gee_feature_path="users/jstacompute/basins",
        reducer_funcs=[asdf],
        verbose=True,
    )    
    res = rabpro.basin_stats.fetch_gee(urls, ["waterclass"])

    assert all(res["waterclass_asdf"] == res["waterclass_max"])


def test_categorical_imgcol():

    urls, task = rabpro.basin_stats.compute(
        [Dataset("MODIS/006/MCD12Q1", "LC_Type1", stats=["freqhist"])],
        gee_feature_path="users/jstacompute/basins",
    )
    res = rabpro.basin_stats.fetch_gee(urls, ["lulc"])

    assert res.shape[1] > 1


def test_timeindexed_imgcol():

    urls, tasks = rabpro.basin_stats.compute(
        [
            Dataset(
                "JRC/GSW1_4/YearlyHistory",
                "waterClass",
            )
        ],
        gee_feature_path="users/jstacompute/basins",
    )

    res = rabpro.basin_stats.fetch_gee(urls, ["waterclass"])

    assert res["waterclass_mean"].iloc[0] > 0
    assert res.shape[0] > 0


def test_timeindexedspecific_imgcol():

    urls, task = rabpro.basin_stats.compute(
        [
            Dataset(
                "JRC/GSW1_4/YearlyHistory",
                "waterClass",
                start="2017-01-01",
                end="2019-01-01",
            )
        ],
        gee_feature_path="users/jstacompute/basins",
    )

    res = rabpro.basin_stats.fetch_gee(urls, [""])

    assert res.shape[0] >= 2


def test_nontimeindexed_imgcol():

    data, task = rabpro.basin_stats.compute(
        [
            Dataset(
                "JRC/GSW1_4/MonthlyRecurrence",
                "monthly_recurrence",
            )
        ],
        gee_feature_path="users/jstacompute/basins",
        test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data[0]["features"]])

    assert res.shape[0] > 0


def test_img():

    data, task = rabpro.basin_stats.compute(
        [
            Dataset(
                "JRC/GSW1_4/GlobalSurfaceWater",
                "occurrence",
                stats=["min", "max", "range", "std", "sum", "pct50", "pct3"],
            )
        ],
        gee_feature_path="users/jstacompute/basins",
        test=True,
    )

    res = pd.DataFrame(data[0]["features"][0]["properties"], index=[0])

    assert float(res["mean"]) > 0
    assert res.shape[1] == 11
