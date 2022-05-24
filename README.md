
<p align="center">
<a href='https:///VeinsOfTheEarth.github.io/rabpro/'><img src="docs/_static/logo_banner.png" height=100/></a>
</p>

<p align="center">  
  <a href=https://github.com/psf/black><img src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
  <a href=https://anaconda.org/conda-forge/rabpro><img src=https://anaconda.org/conda-forge/rabpro/badges/version.svg></a>
  <a href=https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml><img src=https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml/badge.svg></a>
</p>

> Package to delineate watershed basins and compute attribute statistics using [Google Earth Engine](https://developers.google.com/earth-engine/).

## Setup

|[Software installation](https://veinsoftheearth.github.io/rabpro/install/index.html)|[Data configuration](https://veinsoftheearth.github.io/rabpro/configure/index.html#data)|[Software configuration](https://veinsoftheearth.github.io/rabpro/configure/index.html#software)|
|--|--|--|

## Usage

> See Example notebooks:

|[Data configuration](https://veinsoftheearth.github.io/rabpro/examples/notebooks/downloading_data.html)|[Basic workflow](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basic_example.html)|[Multiple basins workflow](https://veinsoftheearth.github.io/rabpro/examples/notebooks/multiple_basins.html)|[Basin stats examples](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basin_stats.html)|
|--|--|--|--|

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

## Contributing

We welcome all forms of user contributions including feature requests, bug reports, code, and documentation requests - simply open an [issue](https://github.com/VeinsOfTheEarth/rabpro/issues).

Note that *rabpro* adheres to [Black code style](https://black.readthedocs.io/en/stable/) and [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) for documentation. We ask that contributions adhere to these standards as much as possible. For code development contributions, please contact us via email (rabpro at lanl [dot] gov) to be added to our slack channel where we can hash out a plan for your contribution.
