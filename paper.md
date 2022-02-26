---
title: 'rabpro: global watershed boundaries, river elevation profiles, and catchment statistics'
tags:
  - watersheds
  - basins
  - rivers
  - DEM
  - Google Earth Engine
authors:
  - name: Jon Schwenk^[Corresponding author]
    orcid: 0000-0001-5803-9686
    affiliation: 1
  - name: Tal Zussman
    orcid: 0000-0003-3087-8511
    affiliation: 2
  - name: Jemma Stachelek
    orcid: 0000-0002-5924-2464
    affiliation: 1
affiliations:
 - name: Los Alamos National Laboratory, Division of Earth and Environmental Sciences
   index: 1
 - name: Columbia University, Department of Computer Science
   index: 2
date: 25 February 2022
bibliography: paper.bib
---


## Summary

River and Basin Profiler (`rabpro`) is a Python package to delineate watersheds, extract river flowlines and elevation profiles, and compute watershed statistics for any location on the Earth’s surface. As fundamental hydrologically-relevant units of surface area, watersheds are areas of land that drain via aboveground pathways to the same location. Delineations of watershed boundaries are typically performed on DEMs that represent surface elevations as gridded rasters. Depending on the resolution of the DEM and the size of the watershed, delineation may be very computationally expensive. With this in mind, we designed `rabpro` to provide user-friendly workflows to manage the complexity and computational expense of watershed calculations given an arbitrary coordinate pair. In addition to basic watershed delineation, `rabpro` will extract the elevation profile for a watershed’s main-channel flowline. This enables the computation of river slope, which is a critical parameter in many hydrologic and geomorphologic models. Finally, `rabpro` provides a user-friendly wrapper around Google Earth Engine’s (GEE) Python API to enable cloud-computing of zonal watershed statistics and/or time-varying forcing data from hundreds of available datasets. Altogether, `rabpro` provides the ability to automate or semi-automate complex watershed analysis workflows across broad spatial extents.

```
![The core functionality of rabpro demonstrated on the Sigatoka River. (A) Study site with both MERIT and HydroBASINS delineations and river flowline extraction for a hypothetical gage station. Bing VirtualEarth base image. (B) MERIT-Hydro delineation with MERIT-Hydro flowlines underneath. (C) HydroBASINS delineation with level-12 HydroBASINS polygons as white outlines. (D) Extracted elevation profile with gage location denoted by white circle at Distance = 0. (E) Examples of time-averaged (where appropriate) basin characteristics retrieved by rabpro from Google Earth Engine. Sources: population [@center_for_international_earth_science_information_network-ciesin-columbia_university_gridded_2017], NDVI [@didan__kamel_mod13a2_2015], topo slope [@amatulli_geomorpho90m_2020], precipitation [precipitation_processing_system_pps_at_nasa_gsfc_gpm_2019], soil moisture [@oneill__peggy_e_smap_2018], and temperature [@copernicus_climate_change_service_era5_2017]. (F, G) Time-series data fetched by rabpro for the temperature and precipitation datasets in (E).](./docs/paper_fig/figure_1.PNG)
```

## Statement of Need

Watersheds play a central and vital role in many scientific, engineering, and environmental management applications (See @brooks_hydrology_2003 for a comprehensive overview). While `rabpro` can benefit any watershed-based research or analysis, it was originally designed to satisfy the needs of data-driven rainfall-runoff models. These models aim to predict a streamflow (runoff) time series as a function of precipitation over the upstream land area (i.e. the watershed). In addition to watershed delineations and precipitation estimates, they typically require data on both time-varying parameters (or forcing data) like temperature, humidity, soil moisture, and vegetation as well as static watershed properties like topography, soil type, or land use/land cover [@kratzert_toward_2019; @gauch_rainfallrunoff_2021; @nearing_data_2021; @kratzert_note_2021]. The `rabpro` API enables users to manage the complete data pipeline necessary to drive such a model starting from the initial watershed delineation through the calculation of static and time-varying parameters. Some hydrologic and hydraulic models also require channel slope for routing streamflow [@boyle_toward_2001; @piccolroaz_hyperstream_2016; @wilson_water_2008], developing rating curves [@fenton_calculation_2001; @colby_relationship_1956], or modeling local hydraulics [@schwenk_life_2015; @schwenk_high_2017; @schwenk_meander_2016]. 

The need for watershed-based data analysis tools is exemplified by the growing collection of published datasets that provide watershed boundaries, forcing data, and/or watershed attributes in precomputed form, including CAMELS [@addor_camels_2017], CAMELS-CL, -AUS, and -BR [@alvarez-garreton_camels-cl_2018; @fowler_camels-aus_2021; @chagas_camels-br_2020], Hysets [@arsenault_comprehensive_2020], and HydroAtlas [@linke_global_2019]. These datasets provide off-the-shelf options for building streamflow models, but they suffer from a degree of inflexibility. For example, someone desiring to add a watershed attribute, to use a new remotely-sensed data product, or to update the forcing data time-series to include the most recently available data must go through the arduous process of sampling it themselves. `rabpro` was designed to provide flexibility for both building a watershed dataset from scratch or appending to an existing one.

While we point to streamflow modeling as an example, many other applications exist. `rabpro` is currently being used to contextualize streamflow trends, build a data-driven model of riverbank erosion, and generate forcing data for a mosquito population model. `rabpro`'s focus is primarily on watersheds, but some users may also find `rabpro`'s Google Earth Engine wrapper convenient for sampling raster data over any geopolygon(s).


## State of the field

The importance of watersheds, availability of DEMs, and growing computational power has led to the development of many excellent open-source terrain (DEM) analysis packages that provide watershed delineation tools, including [TauDEM](https://hydrology.usu.edu/taudem/taudem5/) [@tarboton_terrain_2005], [pysheds](https://github.com/mdbartos/pysheds) [@bartos_pysheds_2020], [Whitebox Tools ](https://github.com/jblindsay/whitebox-tools)[@lindsay_whitebox_2016], [SAGA](https://sagatutorials.wordpress.com/terrain-analysis/) [@conrad_system_2015], among many others. Computing statistics and forcing data from geospatial rasters also has a rich history of development, and Google Earth Engine [@gorelick_google_2017] has played an important role. Almost a decade has passed since Google Earth Engine has been available to developers, and the community has in-turn developed open-source packages to interface with its Python API in user-friendlier ways, including [gee_tools](https://github.com/gee-community/gee_tools) [@principe_gee_tools_2021], [geemap](https://geemap.org/) [@wu_geemap_2020], [eemont](https://github.com/davemlz/eemont) [@montero_eemont_2021], and [restee](https://github.com/KMarkert/restee) [@markert_restee_2021]–each of which provides support for sampling zonal statistics and time series from geospatial polygons.

However, to our knowledge, `rabpro` is the only available package that provides efficient end-to-end delineation and characterization of watershed basins at scale. While a combination of the cited terrain analysis packages and GEE toolboxes can achieve `rabpro`’s functionality, `rabpro`’s blending of them enables simpler, less error-prone, and faster results. 

One unique `rabpro` innovation is its automation of “hydrologically addressing” input coordinates. DEM watershed delineations require that the outlet pixel be precisely specified; in many `rabpro` use cases, this is simply a (latitude, longitude) coordinate that may not align with the underlying DEM. `rabpro` will attempt to “snap” the provided coordinate to a nearby flowline while minimizing the snapping distance and the difference in upstream drainage area (if provided by the user). Another unique `rabpro` feature provides the ability to optimize the watershed delineation method according to basin size such that pixel-based (from MERIT-Hydro [@yamazaki_merit_2019]) delineations can be used for more accurate estimates and/or smaller basins, and coarser subbasin-based (from HydroBASINS [@lehner_hydrobasins_2014]) delineations can be used for rapid estimates of larger basins. 


## Functionality

`rabpro` executes watershed delineation based on either the MERIT-Hydro dataset, which provides a global, ~90 meter per pixel, hydrologically-processed DEM suite, or the HydroBASINS data product, which provides pre-delineated subbasins at approximately ~230 km^2 per subbasin. Conceptually, basin delineation is identical for both. The user-provided coordinate is hydrologically addressed by finding the downstream-most pixel (MERIT-Hydro) or subbasin (HydroBASINS). The watershed is then delineated by finding all upstream pixels or subbasins that drain into the downstream pixel/subbasin and unioning these pixels/subbasins into a single polygon. A user must therefore download either the MERIT-Hydro tiles covering their study watershed or the appropriate HydroBASINS product; `rabpro` provides tooling to automate these downloads and create its expected data structure (See the Downloading data [notebook](https://github.com/VeinsOfTheEarth/rabpro/blob/main/docs/source/examples/notebooks/downloading_data.ipynb)).

There are three primary operations supported by `rabpro`: 1) basin delineation, 2) elevation profiling, and 3) subbasin (zonal) statistics. If operating on a single coordinate pair, the cleanest workflow would be instantiating an object of the `profiler` class and calling (in order) the `delineate_basins()`, `elev_profile()`, and `basin_stats()` methods (See the [Basic Example](https://veinsoftheearth.github.io/rabpro/examples/notebooks/basic_example.html) notebook). If operating on multiple coordinate pairs, the workflow would loop through each coordinate pair while delineating each watershed (optionally calculating its elevation profile). As the loop runs, the user collects each basin polygon in a list, concatenates the list, and directly calls  `subbasin_stats.compute()` on the resulting GeoDataFrame (See the [Full Example](https://veinsoftheearth.github.io/rabpro/examples/notebooks/full_example.html) notebook). More details on package functionality can be found in [the documentation](https://VeinsOfTheEarth.github.io/rabpro/).


![alt_text](images/image1.png "image_tooltip")


Figure 1: Example output from the [Full Example](https://veinsoftheearth.github.io/rabpro/examples/notebooks/full_example.html) notebook where dam-associated [@prior_vote-dams_2022] watersheds in Sri Lanka are delineated and zonal statistics are run for water occurrence, temperature, and precipitation.


## Dependencies

`rabpro` relies on functionality from the following Python packages: GDAL [@gdalogr_contributors_gdalogr_2020], NumPy [@harris_array_2020], GeoPandas [@jordahl_geopandasgeopandas_2020], Shapely [@gillies_shapely_2007], pyproj [@snow_pyproj4pyproj_2021], scikit-image [@van_der_walt_scikit-image_2014], scipy [@virtanen_scipy_2020], and earthengine-api [@gorelick_google_2017]. Use of the watershed statistics methods requires a free Google Earth Engine account. Required MERIT-Hydro and HydroBASINS data are freely available for download by visiting their websites or using `rabpro`’s download scripts; MERIT-Hydro requires users to first register to receive a username and password for access to downloads.


## Acknowledgements

Jordan Muss, Joel Rowland, and Eiten Shelef envisioned and created a predecessor to `rabpro` and helped guide its early development. `rabpro` was developed with support from the Laboratory Directed Research and Development program of Los Alamos National Laboratory under project number 20210213ER and as part of the Interdisciplinary Research for Arctic Coastal Environments (InteRFACE) project through the Department of Energy, Office of Science, Biological and Environmental Research Earth and Environment Systems Sciences Division RGMA program, awarded under contract grant #9233218CNA000001 to Triad National Security, LLC (“Triad”). TZ was supported by funding from the Columbia Undergraduate Scholars Program Summer Enhancement Fellowship.


## References
