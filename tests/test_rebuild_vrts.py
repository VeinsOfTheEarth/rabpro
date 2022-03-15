import contextlib, io

from rabpro import utils as ru


def test_rebuild_vrts():

    # initial building
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dps = ru.get_datapaths()
        ru.build_virtual_rasters(dps, skip_if_exists=False, verbose=True)
    output = f.getvalue()

    # ensure caching has occurred
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dps = ru.get_datapaths()
        ru.build_virtual_rasters(dps, skip_if_exists=False, verbose=False)
    output = f.getvalue()
    assert len(output) == 0

    # ensure cache override works
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dps = ru.get_datapaths(force=True)
        ru.build_virtual_rasters(dps, skip_if_exists=False, verbose=True)
    output = f.getvalue()
    assert len(output) > 0
