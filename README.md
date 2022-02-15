# rabpro - river and basin profiler <a href='https:///VeinsOfTheEarth.github.io/rabpro/'><img src="docs/_static/logo.png" align="right" height=140/></a>

[![PyPI Latest Release](https://img.shields.io/pypi/v/rabpro.svg)](https://pypi.org/project/rabpro/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Anaconda badge](https://anaconda.org/jschwenk/rabpro/badges/version.svg)](https://anaconda.org/jschwenk/rabpro) [![build](https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml/badge.svg)](https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml)

> Package to delineate watershed subbasins and compute attribute statistics using [Google Earth Engine](https://developers.google.com/earth-engine/).

## Setup

|[Software installation](https://veinsoftheearth.github.io/rabpro/install/index.html)|[Data configuration](https://veinsoftheearth.github.io/rabpro/configure/index.html#data)|[Software configuration](https://veinsoftheearth.github.io/rabpro/configure/index.html#software)|
|--|--|--|

## Usage

> See Example notebooks:

|[Basic workflow](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basic_example.html)|[Full workflow](https://veinsoftheearth.github.io/rabpro/examples/notebooks/full_example.html)|[Basin stats examples](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basin_stats.html)|
|--|--|--|

## Development

### Testing

```python
# file-based testing
python -m unittest tests/test.py

# object-based testing
python -m pytest
```

### Local docs build

```shell
cd docs && make html
```
