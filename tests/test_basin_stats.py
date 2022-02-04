import ee
import copy
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rabpro.subbasin_stats import Dataset

import rabpro


# coords_file = gpd.read_file(r"tests/data/Big Blue River.geojson")
# total_bounds = coords_file.total_bounds
total_bounds = np.array([-85.91331249, 39.42609864, -85.88453019, 39.46429816])
gdf = gpd.GeoDataFrame({"idx": [1], "geometry": [box(*total_bounds)]}, crs="EPSG:4326")


def clean_res(feature):
    res = pd.DataFrame(feature["properties"], index=[0])
    res["id"] = feature["id"]
    return res


def clean_freqhist(feature, name_category):
    feature = copy.deepcopy(feature)
    res_hist = pd.DataFrame(feature["properties"]["histogram"], index=[0])
    res_hist.columns = [name_category + "_" + x for x in res_hist.columns]

    del feature["properties"]["histogram"]
    res = pd.DataFrame(feature["properties"], index=[0])
    res["id"] = feature["id"]

    res = pd.concat([res, res_hist], axis=1)

    return res


def test_customreducer():
    def asdf(feat):
        return feat.getNumber("max")

    data, task = rabpro.subbasin_stats.main(
        [Dataset("JRC/GSW1_3/YearlyHistory", "waterClass", stats=["max"])],
        sb_inc_gdf=gdf,
        reducer_funcs=[asdf],
        test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data[0]["features"]])

    assert all(res["asdf"] == res["max"])


def test_categorical_imgcol():
    data, task = rabpro.subbasin_stats.main(
        [Dataset("MODIS/006/MCD12Q1", "LC_Type1", stats=["freqhist"])],
        sb_inc_gdf=gdf,
        test=True,
    )

    res = pd.concat(
        [clean_freqhist(feature, "LC_Type1") for feature in data[0]["features"]]
    )

    assert res.shape[1] > 4


def test_timeindexed_imgcol():

    data, task = rabpro.subbasin_stats.main(
        [Dataset("JRC/GSW1_3/YearlyHistory", "waterClass",)], sb_inc_gdf=gdf, test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data[0]["features"]])

    assert res["mean"].iloc[0] > 0
    assert res.shape[0] > 0


def test_timeindexedspecific_imgcol():

    data, task = rabpro.subbasin_stats.main(
        [
            Dataset(
                "JRC/GSW1_3/YearlyHistory",
                "waterClass",
                start="2017-01-01",
                end="2019-01-01",
            )
        ],
        sb_inc_gdf=gdf,
        test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data[0]["features"]])

    assert res.shape[0] == 2


def test_nontimeindexed_imgcol():

    data, task = rabpro.subbasin_stats.main(
        [Dataset("JRC/GSW1_3/MonthlyRecurrence", "monthly_recurrence",)],
        sb_inc_gdf=gdf,
        test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data[0]["features"]])

    assert res.shape[0] > 0


def test_img():

    data, task = rabpro.subbasin_stats.main(
        [
            Dataset(
                "JRC/GSW1_3/GlobalSurfaceWater",
                "occurrence",
                stats=["min", "max", "range", "std", "sum", "pct50", "pct3"],
            )
        ],
        sb_inc_gdf=gdf,
        test=True,
    )

    res = pd.DataFrame(data[0]["features"][0]["properties"], index=[0])

    assert float(res["mean"]) > 0
    assert res.shape[1] == 9
