# -*- coding: UTF-8 -*-
import geopandas as gpd
from shapely.geometry import Point, Polygon


shapefile = '/home/ts809/Documents/Code/github/building_resilience/extra_resources/data_resources/shapes/ne_110m_admin_0_countries.shp'
#Read shapefile using Geopandas
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
#Rename columns.
gdf.columns = ['country', 'country_code', 'geometry']

polygons = list(gdf['geometry'])
countries = gdf['country_code'].values

def get_polygon_index(point, polygons=polygons):
    for i, polygon in enumerate(polygons):
        if Point(point).within(polygon):
            return i
    return None

def whichCountry(point, polygons=polygons):
    '''
    Finding the country from coordinates
    Parameters:
    ----------
    coords (tuple):
        (latitude,longitude)
    Returns:
    --------
    UN_code (string):
        Three letter UN country code
    '''
    point = (point[1],point[0])  # the function is actually asking for (lon,lat)
    i = get_polygon_index(point, polygons)
    if i == None:
        return None
    else:
        return countries[i]
