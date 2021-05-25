# TDA analysis of periodic dynamic networks

## Introduction

This package implements a pipeline to detect periodicity and quasiperiodicity in time series of networks, which may have node and edge weights. The method is based on a topological approach which first summarizes the time series of networks using a time series of sublevel set persistence diagrams; these are informative representations of the connectivity information and node weights of the network. Motivated by a result from dynamical systems theory, Takens' theorem, which states that a time-delay embedding of a time series can capture its geometry, we analyze the time delay embedding of this persistence diagram time series. In particular, a circular or toroidal geometry reflects periodic and quasiperiodic behavior in the time series, respectively.

## Dependencies
Standard computing python libraries: numpy, scipy, matplotlib.

TDA python libraries (can be installed via pip):

- Ripser 
- persim
- gudhi

See `requirements.txt` for a full list of dependencies. You can install the package via pip as

`pip3 install tdadynamicnetworks`.

## Code Description and Pipeline usage

See the notebooks for a usage of the full pipeline. At a high level, we apply the following functions in order, supposing we have as input a dynamic network.

- `persistence_fns.get_filtration`, applied to each graph in the dynamic network. 
- Pass the output of the previous step to `persistence_fns.get_rips_complex`, which calculates the $0$-dimensional sublevel set filtration of the dynamic network.
- Calculate the distance matrix on the resulting time series of barcodes, using `persistence_fns.get_bottleneck_dist_matrix`.
- Then run `sliding_window_fns.sliding_window` and `sliding_window_fns.sw_distance_matrix` to create the sliding window embedding of the barcode time series. Note that `sliding_window_fns.sliding_window` generates the indices of the time series needed to create the sliding window; this is then passed to the second function to subset the whole time series distance matrix in order to create the distance matrix of the sliding window.
- Calculate the persistence of the sliding window point cloud via the `ripser` package, using its distance matrix.

If you have pip installed the package, call each of the above functions using the module name `tdadynamicnetworks`.

## Notebooks / Experiments

The experiments on NOAA weather data utilize datasets which are too unwieldy to include in git -- these need to be downloaded from the [NOAA](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00516/html) in order for the relevant notebooks to be run.

Simulated experiments in the `notebooks` folder rely on functions in the submodule `tdadynamicnetworks/examples/`.

## References

[1] Timothy Sudijono, Christopher Tralie, and Jose A. Perea. *Periodicity Detection of Sampled Functions and Dynamic Networks.* In preparation.