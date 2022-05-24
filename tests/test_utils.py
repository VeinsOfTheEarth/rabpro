import pytest
from rabpro import utils


def test_negativelat():
    assert utils.coords_to_merit_tile(178, -17) == "s30e150"


def test_negativelon():
    assert utils.coords_to_merit_tile(-118, 32) == "n30w120"
    assert utils.coords_to_merit_tile(-97.355, 45.8358) == "n30w120"


def test_nodata():
    with pytest.raises(ValueError):
        utils.coords_to_merit_tile(-140, 10)

    with pytest.raises(ValueError):
        utils.coords_to_merit_tile(200, 200)
