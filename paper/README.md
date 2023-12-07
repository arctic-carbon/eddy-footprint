## Ludwig et al. Biogeosciences footprint ensembles

This portion of the repository contains code used in:

Ludwig, S. M., L. Schiferl, J. Hung, S. M. Natali, R. Commane. Biogeosciences. In review.
Resolving heterogeneous fluxes from tundra halves the growing season carbon budget. https://bg.copernicus.org/preprints/bg-2023-119/

The jupyter notebook here uses an early version of footprint generation that is site-specific for this study.
While the footprint models are the same as those implmented in eddy-footprint, the rotation, interpolation and projecting steps are different.
This notebook relies on gdal for these steps and creates temporary raster (.tif) files.
The output is numerous Geotiffs rather than an xarray object.

We maintain this dataset and notebook as an archive for the associated paper.
The eddy-footprint package should be preferred to use as it is faster, more versatile, and integrated with scientific Python libraries.
