import contextlib, io

from rabpro import utils as ru


def test_rebuild_vrts():

    # initial building
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dps = ru.get_datapaths(rebuild_vrts=True, quiet=False)
    output = f.getvalue()
    assert len(output) > 0

    # ensure caching has occurred
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dps = ru.get_datapaths(rebuild_vrts=True, quiet=False)
    output = f.getvalue()
    assert len(output) == 0

    # ensure cache override works
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dps = ru.get_datapaths(rebuild_vrts=True, quiet=False, force=True)
    output = f.getvalue()
    assert len(output) > 0
