import rabpro
import geopandas as gpd
import contextlib, io


def test_rebuild_vrts():
    coords_file = gpd.read_file(r"tests/data/Big Blue River.geojson")

    # Ensure profiler calls to get_datapaths prints rebuild_vrts progress
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        rpo = rabpro.profiler(coords_file)
    output = f.getvalue()
    # # jschwenk turned off all rebuild_vrt printing
    # assert len(output) > 0

    # Ensure delineate_basins calls to get_datapaths DOES NOT trigger rebuild_vrts
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        rpo.delineate_basins()
    output = f.getvalue()
    assert len(output) < 600
