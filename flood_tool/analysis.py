"""Analysis tools."""

import os
import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from .geo import GeoTransformer
import flood_tool

import branca.colormap as cm
from folium.plugins import HeatMap, MarkerCluster
from folium import Marker
import folium
from folium.plugins import MousePosition


__all__ = ['plot_risk_map']

def plot_postcode_density(postcode_file=(os.path.dirname(__file__)
                                      +'/resources/postcodes_unlabelled.csv'),
                          coordinate=['easting','northing'], dx=1000):
    """Plot a postcode density map from a postcode file."""

    
    pdb = pd.read_csv(postcode_file)

    bbox = (pdb[coordinate[0]].min()-0.5*dx, pdb[coordinate[0]].max()+0.5*dx,
            pdb[coordinate[1]].min()-0.5*dx, pdb[coordinate[1]].max()+0.5*dx)

    
    nx = (math.ceil((bbox[1] - bbox[0])/dx),
          math.ceil((bbox[3] - bbox[2])/dx))

    x = np.linspace(bbox[0]+0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2]+0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y in pdb[coordinate].values:
        Z[math.floor((x-bbox[0])/dx), math.floor((y-bbox[2])/dx)] += 1

    plt.pcolormesh(X, Y, np.where(Z>0, Z, np.nan).T,
                   norm=matplotlib.colors.LogNorm())
    plt.axis('equal')
    plt.colorbar()

def plot_risk_map(risk_data, coordinate=['easting','northing'], dx=1000):
    """Plot a risk map."""

    bbox = (risk_data[coordinate[0]].min()-0.5*dx, risk_data[coordinate[0]].max()+0.5*dx,
            risk_data[coordinate[1]].min()-0.5*dx, risk_data[coordinate[1]].max()+0.5*dx)

    
    nx = (math.ceil((bbox[1] - bbox[0])/dx),
          math.ceil((bbox[3] - bbox[2])/dx))

    x = np.linspace(bbox[0]+0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2]+0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y, val in risk_data[coordinate+['risk']].values:
        Z[math.floor((x-bbox[0])/dx), math.floor((y-bbox[2])/dx)] += val

    plt.pcolormesh(X, Y, np.where(Z>0, Z, np.nan).T,
                   norm=matplotlib.colors.LogNorm())
    plt.axis('equal')
    plt.colorbar()
        
def plot_EN_map(easting,northing,risk,filename):
    '''
    plot a heatmap
            
    Parameters
    ----------
    easting: array
    northing: array
    risk: array 
        dtype should be float
    filename: str
        the name of map you plot

    Returns
    -------
    a map
    '''
    lat,lon = GeoTransformer.get_gps_lat_long_from_easting_northing(easting,northing)
    m = folium.Map([53., -1],
               zoom_start=5.5,
               control_scale=True
              )

    data1 = [[lat[i], lon[i], risk[i]] for i in range(len(risk))]
    min_risk = risk.min()
    max_risk = risk.max()

    # add a heatmap
    HeatMap(data1).add_to(m)

    #add a colorbar
    cm.LinearColormap(['purple','c',"green", "yellow", 'orange',"red"],vmin=min_risk,vmax=max_risk).add_to(m)
    
    # add coordinate
    formatter = "function(num) {return L.Util.formatNum(num, 4) + ' º ';};"
    plugin_hover = MousePosition(
        position='topright',
        separator=' | ',
        empty_string='Mouse swipe to display latitude and longitude',
        lng_first=False,
        num_digits=20,
        prefix='coordinate：',
        lat_formatter=formatter,
        lng_formatter=formatter,
    )
    m.add_child(plugin_hover)
    
    # add markercluster
    mc = MarkerCluster()# initialization
    for i in range(len(risk)):
        # Add the latitude and longitude of each data to point clustering
        mc.add_child(Marker([lat[i], lon[i]]))
    m.add_child(mc)

    # add a minimap
    minimap = folium.plugins.MiniMap(toggle_display=True,
                         tile_layer='stamenwatercolor',
                         position='bottomleft',
                         width=300, height=200,
                         zoom_level_offset=-4)
    m.add_child(minimap)

    # add latlngpopup
    m.add_child(folium.LatLngPopup())

    # save as a html
    m.save(f"{filename}.html")
    return m

def plot_84_map(latitude,longitude,risk,filename):
    '''
    plot a heatmap
            
    Parameters
    ----------
    latitude: array
    longitude: array
    risk: array 
        dtype should be float
    filename: str
        the name of map you plot

    Returns
    -------
    a map
    '''
    m = folium.Map([53., -1],
               zoom_start=5.5,
               control_scale=True
              )

    data1 = [[latitude[i], longitude[i], risk[i]] for i in range(len(risk))]
    min_risk = risk.min()
    max_risk = risk.max()

    # add a heatmap
    HeatMap(data1).add_to(m)

    #add a colorbar
    cm.LinearColormap(['purple','c',"green", "yellow", 'orange',"red"],vmin=min_risk,vmax=max_risk).add_to(m)
    
    # add coordinate
    formatter = "function(num) {return L.Util.formatNum(num, 4) + ' º ';};"
    plugin_hover = MousePosition(
        position='topright',
        separator=' | ',
        empty_string='Mouse swipe to display latitude and longitude',
        lng_first=False,
        num_digits=20,
        prefix='coordinate：',
        lat_formatter=formatter,
        lng_formatter=formatter,
    )
    m.add_child(plugin_hover)
    
    # add markercluster
    mc = MarkerCluster()# initialization
    for i in range(len(risk)):
        # Add the latitude and longitude of each data to point clustering
        mc.add_child(Marker([latitude[i], longitude[i]]))
    m.add_child(mc)

    # add a minimap
    minimap = folium.plugins.MiniMap(toggle_display=True,
                         tile_layer='stamenwatercolor',
                         position='bottomleft',
                         width=300, height=200,
                         zoom_level_offset=-4)
    m.add_child(minimap)

    # add latlngpopup
    m.add_child(folium.LatLngPopup())

    # save as a html
    m.save(f"{filename}.html")
    return m

def plot_annual_flood_risk_from_postcodes(postcodes):
    """
    Return a map

    ana.plot_annual_flood_risk_from_postcodes(postcodes)
    
    Parameters
    ----------

    postcodes : sequence of strs
        Sequence of postcodes.

    Returns
    -------
    a map and a html file
    """
    new_tool = flood_tool.Tool(sample_labels=['riskLabel','medianPrice'])   
    methods = new_tool.get_default_methods()
    new_tool.train(method=methods)  
    pred_risklabel = new_tool.get_annual_flood_risk(risk_labels=True)
    risk = np.array(pred_risklabel.values,dtype=float)
    coordinates = new_tool.get_easting_northing(postcodes)
    m = plot_EN_map(coordinates['easting'].values,coordinates['northing'].values,risk,'pred_risklabel')
    return m


'''
usage(an example):
import flood_tool.analysis as ana

df = pd.read_csv('flood_tool/resources/postcodes_sampled.csv')
easting = df['easting'].values
northing = df['northing'].values
risk = np.array(df['riskLabel'],dtype=float)
risk_data = pd.DataFrame({'easting':easting,'northing':northing,'risk':risk})

ana.plot_risk_map(risk_data=risk_data,coordinate=['easting','northing'])
ana.plot_EN_map(easting,northing,risk,'example')
ana.plot_84_map(latitude,longitude,risk,'example')

POSTCODES = ['TN6 3AW','BN7 2HP','BN1 5PF']
ana.plot_annual_flood_risk_from_postcodes(POSTCODES)
'''
    
