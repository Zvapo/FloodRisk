"""Interactions with rainfall and river data."""

import numpy as np
import pandas as pd

import urllib.request
import json

__all__ = ["get_station_data_from_csv"]


def get_station_data_from_csv(filename, station_reference):
    """Return readings for a specified recording station from .csv file.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference
        station_reference to return.

    >>> data = get_station_data_from_csv('resources/wet_day.csv')
    """
    frame = pd.read_csv(filename)
    frame = frame.loc[frame.stationReference == station_reference]

    return pd.to_numeric(frame.value.values)

def get_live_station_data(station_reference):
    """Return readings for a specified recording station from live API.

    Parameters
    ----------

    station_reference
        station_reference to return.

    Returns
    ----------
    value in certain station
    
    >>> data = get_live_station_data('0184TH')
    """
    url='https://environment.data.gov.uk/flood-monitoring/id/stations/'+station_reference
    url=urllib.request.urlopen(url)
    content=url.read()
    content=json.loads(content)
    try:
        if content['items']['label'] == 'Rainfall station':
            value = np.array(content['items']['measures']['latestReading']['value'])
        else:
            try:
                value = np.array(content['items']['measures'][1]['latestReading']['value'])
            except KeyError:
                value = np.array(content['items']['measures']['latestReading']['value'])   
    except TypeError:
        value = None
    except KeyError:
        value = None
    return pd.to_numeric(value)

def get_all_livedata():
    '''
    this function return a dataframe which contains station_reference and values
    At the same time you will get a csv file of real-time data
    '''
    stations = pd.read_csv('flood_tool/resources/stations.csv')
    stations.dropna(axis=0, how='any', thresh=None, subset='latitude', inplace=True)
    station_reference = np.array(stations['stationReference'])
    values = []
    for i in station_reference:
        value = get_live_station_data(i)
        values.append(value)   
    df = pd.DataFrame({'stationReference':station_reference,'value':values})

    # get live time
    url='https://environment.data.gov.uk/flood-monitoring/id/stations/'+'000008'
    url=urllib.request.urlopen(url)
    content=url.read()
    content=json.loads(content)
    time = content['items']['measures']['latestReading']['dateTime']
    
    # get livedata with coordinates()
    livedata = stations[stations['stationReference'].isin(df.stationReference.values)]
    livedata = livedata.set_index('stationReference')
    df = df.join(livedata, on='stationReference')
    df.to_csv(f'{time}.csv',index=False)
    return df




'''
usage:
import flood_tool.live as live

live.get_station_data_from_csv(filename='flood_tool/resources/wet_day.csv',station_reference='2841TH')
live.get_live_station_data(station_reference='2841TH')
live.get_all_livedata()
'''