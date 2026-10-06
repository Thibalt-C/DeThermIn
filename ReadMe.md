# DeThermIn: a complete workflow towards spatio-temporally refined Land Surface Temperature

**Date**: September 2026 \
**Author**: Thibault CHARDON (IPGP, Université Paris Cité) \
\

***CC-BY 4.0 licence** (https://creativecommons.org/licenses/by/4.0/deed.fr).*

The aim of this workflow is to derive Dense Thermal Information (DeThermIn) for urban areas. More precisely, it refines Land Surface Temperature (LST) from space-borne thermal imagery at a nearly-continuous temporal resolution and up to 0.5 meters of spatial resolution.

This repository aims to make easily reproducible the methodology and tools used in the following:
> Chardon, T. (2026): *Use of 3D Point Clouds with Advanced Deep Learning Techniques for Satellite Thermal Imagery Refinement in Urban Environments*. TU Delft [internship report], supervisor: Lindenbergh, R.C.

Please cite it if you make use of part or totality of this work.

## Rationale

With anthropic climate change, urban communities are exposed to multiple Slow Onset Events[^https://unfccc.int/process/bodies/constituted-bodies/WIMExCom/SOEs]. Among them, the increase in temperature raises concerns, as it will augment the heat stress in cities and subsequently cause more serious health issues among the population. Furthermore, extreme heat events are foreseen — and yet observed — to be more intense, longer and more numerous in this context, adding up to the pressure applied on public health systems and causing visible downfalls for urban societies in general.

Such increase in heat stress is not spatially homogeneous, and some areas, denoted as Urban Heat Islands (UHIs), present higher perceived temperatures. Micro-climate studies intend to understand those disparities, and ultimately highlight possible adaptations of the urban infrastructures, that allow to mitigate the UHI effect. While some solutions are already identified for a general case (e.g. the use of more reflective materials, more vegetated and pervious covers), there is a lack in specific solutions based on local urban context. Looking for such solutions would also help to avoid mis-adaptations, by providing quantified scenarios to local experts. It thus requires observations, using adequate metrics. In this scope, Land Surface Temperature (LST), which is derived from thermal remote sensing, can help to derive spatially dense representations of heat stress, although it needs to be combined with other thermal indicators to avoid biased representations.

Airborne remote sensing of LST has the capability of producing maps with a ground sampling distance — spatial resolution — of less than a meter. However, it is associated with subsequent costs and is hardly ever done more than once on a same area. Several space missions produce LST maps with different combinations of spatial and temporal resolutions. A major goal would then consist in using those multiple sources of data to produce spatially and temporally enhanced LST. Such data fusion technique is often divided into two steps: (1) spatio-temporal fusion, that blends multi-source LST data to obtain the best spatial and temporal resolutions; (2) super-resolution, that uses ancillary data of higher spatial resolution, to estimate LST at a scale similar to airborne capabilities — that are then providing ground truth in this optic.

## Principle

The DeThermIn workflow, via its associated python module, proposes an approach based on 3 different space-borne missions, and novel dynamical features derived from airborne LiDAR data. It can be subdivided into two parts similarly to upper:
1. Spatio-temporal fusion using a Diurnal Temperature Cycle (DTC) model (Göttsche & Olesen, 2001)
2. Super-resolution using high resolution ancillary data with a deep-learning based regression

The DTC model enables continuous estimation of LST, as long as it is assumed to follow a Sun-dependent and clear-sky behaviour. Thus, we derive an hourly LST from this model. The super-resolution model is trained and validated using ground truth LST from Toulouse airborne campaign CAMCATT-AI4GEO (2021) that is resampled at 2 meters of resolution (initial ground sampling distance being 1.95 m). This 2 m resolution can however be downsampled up to 0.5 meters, corresponding to the resolution of the used ancillary data. Hence, by default, the DeThermIn workflow delivers a 1 hour / 2 meters refined LST.

## Requirements

The DeThermIn workflow has been developed and tested using python version 3.12. A *conda* environment was used, containing the following packages:

| Package | Version | Goal |
| --- | --- | --- |
| ``cartopy`` | 0.25.0 | geographic tools for `matplotlib` |
| ``cmcrameri`` | 1.9 | colormaps |
| ``folium`` | 0.0.20 | interactive maps generation |
| ``geojson`` | 3.2.0 | convert geometries to geojson instances |
| ``geopandas`` | 1.1.3 | geographic database handling |
| ``graphviz`` | 0.21 | displaying heuristic maps (*facultative*) |
| ``h5py`` | 3.16.0 | hdf5 file reader |
| ``hda`` | 2.39 | API client for EUMETSAT data |
| ``ipython`` | 8.37.0 | displaying animated plots |
| ``joblib`` | 1.5.3 | parallel computations |
| ``lightning`` | 2.6.5 | GPU acceleration for `pytorch` |
| ``matplotlib`` | 3.10.8 | visualisation tools |
| ``numpy`` | 2.2.3 | general computations |
| ``openeo`` | 0.46.0 | API client for Copernicus data |
| ``owslib`` | 0.35.0 | connection to WMS/WMTS services |
| ``pandas`` | 3.0.2 | database handling and datetime conversions |
| ``pillow`` | 12.2.0 | geotiff import |
| ``pvlib`` | 0.15.2 | solar position and irradiance computations |
| ``pyproj`` | 3.7.2 | coordinate system reprojections |
| ``pytorch`` | 2.10.0 | deep learning architectures |
| ``pytorch-msssim`` | 1.0.0 | multi-scale SSIM metric for `pytorch` models |
| ``rasterio`` | 1.5.0 | handling of geotiff files |
| ``requests`` | 2.32.5 | makes HTTP requests |
| ``rioxarray`` | 0.21.0 | `rasterio` applied to `xarray` objects |
| ``rvt-py`` | 2.2.3 | relief visualisation tools |
| ``scikit-learn`` | 1.8.0 | machine learning algorithms |
| ``scipy`` | 1.18.0 | least squares regression and specific computations |
| ``shapely`` | 2.1.2 | creation of polygons and geometry collections |
| ``tqdm`` | 4.67.3 | progression bars |
| ``xarray`` | 2025.1.1 | handling of time series |

It is recommended to create an environment dedicated to this workflow beforehand. 

## Get started

### *For a first use*

When using for the first time the DeThermIn workflow, you need to prior download the geostationary data. This can be done by executing the following on a terminal:

```
python pre-download geostationary data.py
```

It will download the geostationary data for all the available areas (i.e. France and the Netherlands) from the 1st of June 2026 to the 31th of August 2026. This date can be edited in the script. Depending on the used date range, the data can come from Meteosat Third Generation (MTG) or Meteosat Second Generation (MSG).

Another thing to do is to train the deep neural network used for the super-resolution. Use the following to launch the training:

```
python train ResUNet.py
```

Note that you are able to choose a spatial resolution and to select which features you want to use. You just need to edit the script.



### Select an area

The DeThermIn workflow has been implemented for both France and the Netherlands (mainland). The following French ultramarine territories are also included: Réunion, Mayotte, Guadeloupe, Martinique.

Please mention the geographic coordinates of the area you are looking for. Then, select the size of the square centred on those coordinates, corresponding to the area of interest you want to obtain.


### Choose a specific day

You have to choose a single day for which you want to process the high resolution LST.

> Note: you can only choose a day between the 1st of June 2026 and the 31th of August 2026. If you want to select another date, please change the acquired dates in the following script: `pre-download geostationary data.py `