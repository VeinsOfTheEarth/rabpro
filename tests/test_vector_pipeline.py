import rabpro
from pyproj import CRS
import geopandas as gpd

from rabpro import utils as ru


def test_profiler():
    shp_path = r"tests/data/test_coords.shp"
    csv_path = r"tests/data/test_coords.csv"
    coords_file = gpd.read_file(shp_path)

    # read from csv path
    rpo_csv = rabpro.profiler(csv_path)
    assert str(type(rpo_csv)) == "<class 'rabpro.core.profiler'>"

    # read from shp path
    rpo_shp = rabpro.profiler(shp_path)
    assert str(type(rpo_shp)) == "<class 'rabpro.core.profiler'>"

    # read from tuple
    rpo_tuple = rabpro.profiler(
        (coords_file["latitude"][0], coords_file["longitude"][0])
    )
    assert str(type(rpo_tuple)) == "<class 'rabpro.core.profiler'>"

    # read from gdf
    rpo_gdf = rabpro.profiler(coords_file)
    assert str(type(rpo_gdf)) == "<class 'rabpro.core.profiler'>"

    # read from non-4326 gdf
    rpo_gdf_non4326 = rabpro.profiler(coords_file.to_crs(CRS.from_epsg(3857)))
    assert str(type(rpo_gdf_non4326)) == "<class 'rabpro.core.profiler'>"

    # setting da
    rpo_da = rabpro.profiler((56.22659, -130.87974), da=1994)
    assert str(type(rpo_da)) == "<class 'rabpro.core.profiler'>"


def test_delineate_basin():
    csv_path = r"tests/data/test_coords.csv"

    rpo_hydrobasins = rabpro.profiler(csv_path)
    rpo_hydrobasins.delineate_basin()
    assert (
        str(type(rpo_hydrobasins.watershed))
        == "<class 'geopandas.geodataframe.GeoDataFrame'>"
    )

    rpo_merit = rabpro.profiler(csv_path)
    rpo_merit.delineate_basin(force_merit=True)
    assert (
        str(type(rpo_merit.watershed))
        == "<class 'geopandas.geodataframe.GeoDataFrame'>"
    )


def test_elev_profile():
    dps = ru.get_datapaths(force=True)
    ru.build_virtual_rasters(dps)
    csv_path = r"tests/data/test_coords.csv"

    rpo = rabpro.profiler(csv_path)
    rpo.elev_profile(dist_to_walk_km=5)
    assert str(type(rpo.flowline)) == "<class 'geopandas.geodataframe.GeoDataFrame'>"

