##########
Flood Tool
##########

This package implements a flood risk prediction and visualization tool.

Installation Instructions
-------------------------

To use our model for predicting the flood risk for a specific postcode, first use git to clone the repository containing the material for the model. 
Navigate to/create a folder where you want to store the model. When in the desired location users could run:

.. math::
    git clone https://github.com/ese-msc-2022/ads-deluge-Trent.git

This repository contains an environment.yml file for building the conda environment. Navigate into the repository (e.g. cd ads-deluge-Trent). 
Users then run

.. math::
    conda env create -f environment.yml
    conda activate deluge

You should now see the environment name (deluge) displayed on the left hand side of the terminal, done!
You have successfully accessed our model now!


Usage guide
-----------

Our model contains the following .py documents, 

Risk Tool-
         |
         - tool.py ()
         - geo.py (Convertions. Converting between degrees and radians, between input latitude/longitude and 
           Cartesian (x, y, z) coordinates, or between OSGB36 easting/northing and GPS latitude and longitude pairs)
         - gridsearch.py
         - live.py (gets the live station data from the website)
         - transformer.py
        
Visualization-
             |
             - analysis.py (plots the postcode density, risk map and heatmap)
             - mapping.py (plots circles on the map)

Risk tool
-----------

Usage:

.. math::
    tool = flood_tool.Tool()        
    tool.train()        
    tool.get_xxx_method()        

Initialization - load data, preprocess by dropping duplicates and spliting into train and test sets, set default sub-models for different tasks.        

Model training - process data, train different models (do grid search, try different hyperparameters to find the best model).        

Make predictions - 

"get_flood_class_from_postcodes_methods()" :
    Gets and returns a dictionary of available flood probablity classification methods for respective postcodes. 
    There are 3 classification methods in our model: Linear Regression, Decision Tree Regressor and Random Forest Regressor. 
    
    This function is prepared for "get_flood_class_from_postcodes()", which takes string "postcodes" and "methods" as inputs, generates series predicting flood probability classification for a collection of postcodes, and returns series of predicted flood risk classification labels indexed by postcodes. 
    
    The function "get_flood_class_from_locations_methods()" does the same for Easting/Northing coordinates.

"get_house_price_methods()" :
    Gets and returns a dictionary of available flood house price regression methods. 
    
    "get_median_house_price_estimate" function takes string "postcodes" and "methods" as inputs, generates a series predicting median house prices 
    for a collection of postcodes, and returns series of predicted median house prices indexed by postcodes.

"get_local_authority_methods()" :
    Gets and returns a dictionary of available local authorithy classification methods.
    
    "get_local_authority_estimate()" function then takes sequences of floats "eastings" and "northings", and "method" as inputs, generates series predicting local authorities for a sequence of OSGB36 locations, and returns series of predicted authorities indexed by postcodes.

Live Data
-----------
"get_station_data_from_csv()" & "get_live_station_data()" :
    Get precipitation value of specified station.

"get_all_livedata()" :
    Get a dataframe of all live data and save a csv file.

Data Visualization
-----------

In analysis.py,

"plot_risk_map()" :
    Takes a data frame "risk_data" with columns "{'easting','northing','risk'}" as input.

"plot_EN_map()" :
    Takes float arrays "easting", "northing" and "risk", and a string "filename" as inputs, and returns a map.

"plot_84_map()" :
    Takes float arrays "latitude", "longitude" and "risk", and a string "filename" as inputs, and returns a map.

"plot_annual_flood_risk_from_postcodes()" :
    Takes sequence of strs of postcodes as inputs, and returns a map.  

In mapping.py,

"plot_circle()" :
    Takes floats "lat", "lon", "radius" and a folium.Map "map" as inputs, and returns a Folium map object.

Geodetic Transformations
------------------------

For historical reasons, multiple coordinate systems exist in British mapping.
The Ordnance Survey has been mapping the British Isles since the 18th Century
and the last major retriangulation from 1936-1962 produced the Ordance Survey
National Grid (or **OSGB36**), which defined latitude and longitude across the
island of Great Britain [1]_. For convenience, a standard Transverse Mercator
projection [2]_ was also defined, producing a notionally flat gridded surface,
with gradations called eastings and westings. The scale for these gradations
was identified with metres.


The OSGB36 datum is based on the Airy Ellipsoid of 1830, which defines
semimajor axes for its model of the earth, :math:`a` and :math:`b`, a scaling
factor :math:`F_0` and ellipsoid height, :math:`H`.

.. math::
    a &= 6377563.396, \\
    b &= 6356256.910, \\
    F_0 &= 0.9996012717, \\
    H &= 24.7.

The point of origin for the transverse Mercator projection is defined in the
Ordnance Survey longitude-latitude and easting-northing coordinates as

.. math::
    \phi^{OS}_0 &= 49^\circ \mbox{ north}, \\
    \lambda^{OS}_0 &= 2^\circ \mbox{ west}, \\
    E^{OS}_0 &= 400000 m, \\
    N^{OS}_0 &= -100000 m.

More recently, the world has gravitated towards the use of Satellite based GPS
equipment, which uses the (globally more appropriate) World Geodetic System
1984 (or **WGS84**). This datum uses a different ellipsoid, which offers a
better fit for a global coordinate system. Its key properties are:

.. math::
    a_{WGS} &= 6378137,, \\
    b_{WGS} &= 6356752.314, \\
    F_0 &= 0.9996.

For a given point on the WGS84 ellipsoid, an approximate mapping to the
OSGB36 datum can be found using a Helmert transformation [3]_,

.. math::
    \mathbf{x}^{OS} = \mathbf{t}+\mathbf{M}\mathbf{x}^{WGS}.


Here :math:`\mathbf{x}` denotes a coordinate in Cartesian space (i.e in 3D)
as given by the (invertible) transformation

.. math::
    \nu &= \frac{aF_0}{\sqrt{1-e^2\sin^2(\phi^{OS})}} \\
    x &= (\nu+H) \sin(\lambda)\cos(\phi) \\
    y &= (\nu+H) \cos(\lambda)\cos(\phi) \\
    z &= ((1-e^2)\nu+H)\sin(\phi)

and the transformation parameters are

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathbf{t} &= \left(\begin{array}{c}
    -446.448\\ 125.157\\ -542.060
    \end{array}\right),\\
    \mathbf{M} &= \left[\begin{array}{ c c c }
    1+s& -r_3& r_2\\
    r_3 & 1+s & -r_1 \\
    -r_2 & r_1 & 1+s
    \end{array}\right], \\
    s &= 20.4894\times 10^{-6}, \\
    \mathbf{r} &= [0.1502'', 0.2470'', 0.8421''].
    \end{eqnarray*}

Given a latitude, :math:`\phi^{OS}` and longitude, :math:`\lambda^{OS}` in the
OSGB36 datum, easting and northing coordinates, :math:`E^{OS}` & :math:`N^{OS}`
can then be calculated using the following formulae:

.. math::
    \rho &= \frac{aF_0(1-e^2)}{\left(1-e^2\sin^2(\phi^{OS})\right)^{\frac{3}{2}}} \\
    \eta &= \sqrt{\frac{\nu}{\rho}-1} \\
    M &= bF_0\left[\left(1+n+\frac{5}{4}n^2+\frac{5}{4}n^3\right)(\phi^{OS}-\phi^{OS}_0)\right. \\
    &\quad-\left(3n+3n^2+\frac{21}{8}n^3\right)\sin(\phi-\phi_0)\cos(\phi^{OS}+\phi^{OS}_0) \\
    &\quad+\left(\frac{15}{8}n^2+\frac{15}{8}n^3\right)\sin(2(\phi^{OS}-\phi^{OS}_0))\cos(2(\phi^{OS}+\phi^{OS}_0)) \\
    &\left.\quad-\frac{35}{24}n^3\sin(3(\phi-\phi_0))\cos(3(\phi^{OS}+\phi^{OS}_0))\right] \\
    I &= M + N^{OS}_0 \\
    II &= \frac{\nu}{2}\sin(\phi^{OS})\cos(\phi^{OS}) \\
    III &= \frac{\nu}{24}\sin(\phi^{OS})cos^3(\phi^{OS})(5-\tan^2(phi^{OS})+9\eta^2) \\
    IIIA &= \frac{\nu}{720}\sin(\phi^{OS})cos^5(\phi^{OS})(61-58\tan^2(\phi^{OS})+\tan^4(\phi^{OS})) \\
    IV &= \nu\cos(\phi^{OS}) \\
    V &= \frac{\nu}{6}\cos^3(\phi^{OS})\left(\frac{\nu}{\rho}-\tan^2(\phi^{OS})\right) \\
    VI &= \frac{\nu}{120}\cos^5(\phi^{OS})(5-18\tan^2(\phi^{OS})+\tan^4(\phi^{OS}) \\
    &\quad+14\eta^2-58\tan^2(\phi^{OS})\eta^2) \\
    E^{OS} &= E^{OS}_0+IV(\lambda^{OS}-\lambda^{OS}_0)+V(\lambda-\lambda^{OS}_0)^3+VI(\lambda^{OS}-\lambda^{OS}_0)^5 \\
    N^{OS} &= I + II(\lambda^{OS}-\lambda^{OS}_0)^2+III(\lambda-\lambda^{OS}_0)^4+IIIA(\lambda^{OS}-\lambda^{OS}_0)^6



Function APIs
-------------

.. automodule:: flood_tool
  :members:
  :imported-members:


.. rubric:: References

.. [1] A guide to coordinate systems in Great Britain, Ordnance Survey
.. [2] Map projections - A Working Manual, John P. Snyder, https://doi.org/10.3133/pp1395
.. [3] Computing Helmert transformations, G Watson, http://www.maths.dundee.ac.uk/gawatson/helmertrev.pdf
