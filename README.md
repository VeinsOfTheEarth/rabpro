
<p align="center">
<a href='https:///VeinsOfTheEarth.github.io/rabpro/'><img src="docs/_static/logo_banner.png" height=100/></a>
</p>

<p align="center">  
  <a href=https://github.com/psf/black><img src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
  <a href=https://anaconda.org/conda-forge/rabpro><img src=https://anaconda.org/conda-forge/rabpro/badges/version.svg></a>
  <a href=https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml><img src=https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml/badge.svg></a>
  <a style="border-width:0" href="https://doi.org/10.21105/joss.04237">
  <img src="https://joss.theoj.org/papers/10.21105/joss.04237/status.svg" alt="DOI badge" >
</a>
    <a href=https://doi.org/10.5281/zenodo.6600732><img src=https://zenodo.org/badge/DOI/10.5281/zenodo.6600732.svg></a>
</p>

Package to delineate watershed basins and compute attribute statistics using [Google Earth Engine](https://developers.google.com/earth-engine/).

## Setup

|[Software installation](https://veinsoftheearth.github.io/rabpro/install/index.html)|[Data configuration](https://veinsoftheearth.github.io/rabpro/configure/index.html#data)|[Software configuration](https://veinsoftheearth.github.io/rabpro/configure/index.html#software)|
|--|--|--|

## Usage

See Example notebooks:

|[Data configuration](https://veinsoftheearth.github.io/rabpro/examples/notebooks/downloading_data.html)|[Basic workflow](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basic_example.html)|[Multiple basins workflow](https://veinsoftheearth.github.io/rabpro/examples/notebooks/multiple_basins.html)|[Basin stats examples](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basin_stats.html)|
|--|--|--|--|

## Citation

The following text is the current citation for rabpro:

> Schwenk, J., T. Zussman, J. Stachelek, and J. Rowland. (2022). rabpro: global watershed boundaries, river elevation profiles, and catchment statistics.  Journal of Open Source Software, 7(73), 4237, <https://doi.org/10.21105/joss.04237>.

If you delineate watersheds, you should cite either or both (depending on your method) of HydroBasins:

> Lehner, B., Grill G. (2013). Global river hydrography and network routing: baseline data and new approaches to study the world’s large river systems. Hydrological Processes, 27(15): 2171–2186. <https://doi.org/10.1002/hyp.9740>

or MERIT-Hydro:

> Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., & Pavelsky, T. M. (2019). MERIT Hydro: A high‐resolution global hydrography map based on latest topography dataset. *Water Resources Research*, *55*(6), 5053-5073. <https://doi.org/10.1029/2019WR024873>

## Development

### Testing

```python
python -m pytest
python -m pytest -k "test_img"
```

### Local docs build

```shell
cd docs && make html
```

## Contributing

We welcome all forms of user contributions including feature requests, bug reports, code, and documentation requests - simply open an [issue](https://github.com/VeinsOfTheEarth/rabpro/issues).

Note that *rabpro* adheres to [Black code style](https://black.readthedocs.io/en/stable/) and [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) for documentation. We ask that contributions adhere to these standards as much as possible. For code development contributions, please contact us via email (rabpro at lanl [dot] gov) to be added to our slack channel where we can hash out a plan for your contribution.
