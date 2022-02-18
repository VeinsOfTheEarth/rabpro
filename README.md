<a href='https:///VeinsOfTheEarth.github.io/rabpro/'><img src="docs/_static/logo_banner.png" align="center" height=100/></a>

[![PyPI Latest Release](https://img.shields.io/pypi/v/rabpro.svg)](https://pypi.org/project/rabpro/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Anaconda badge](https://anaconda.org/jschwenk/rabpro/badges/version.svg)](https://anaconda.org/jschwenk/rabpro) [![build](https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml/badge.svg)](https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml)

> Package to delineate watershed subbasins and compute attribute statistics using [Google Earth Engine](https://developers.google.com/earth-engine/).

## Setup

|[Software installation](https://veinsoftheearth.github.io/rabpro/install/index.html)|[Data configuration](https://veinsoftheearth.github.io/rabpro/configure/index.html#data)|[Software configuration](https://veinsoftheearth.github.io/rabpro/configure/index.html#software)|
|--|--|--|

## Usage

> See Example notebooks:

|[Basic workflow](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basic_example.html)|[Full workflow](https://veinsoftheearth.github.io/rabpro/examples/notebooks/full_example.html)|[Basin stats examples](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basin_stats.html)|
|--|--|--|

## Citation

The following text is the current citation for rabpro:

> Zussman, T., J. Schwenk, and J. Rowland. River and Basin Profiler: A Module for Extracting Watershed Boundaries, River Centerlines, and Catchment Statistics. Other. Hydrology, December 30, 2021. <https://doi.org/10.1002/essoar.10509912.1>.

## Development

### Testing

```python
python -m pytest
```

### Local docs build

```shell
cd docs && make html
```
