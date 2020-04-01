import numpy as np
import json
import overpy
import pyproj
from shapely.geometry import shape
from shapely.ops import transform
from functools import partial

class Building():
    '''
    Class for open street map building
    '''

    def __init__(self, way, building_usage, hdd, cdd, levels):
        '''
        Parameters:
        -----------
        way: overpy.Way
            coordinates for building perimeter
        building_usage: int
            building usage code
        hdd: float
            heating degree days
        cdd: float
            cooling degree days
        levels: int
            building levels
        '''
        self.way = way
        self.building_usage = building_usage
        self.hdd = hdd
        self.cdd = cdd
        self.levels = levels
    
    def calc_energy_usage(self, trained_classifier, sc):
        '''
        Calculate the energy usage
        '''
        pass

    def estimate_area(self):
        '''
        Estimate the area of the building using OSM coordinates
        Return:
        -------
        area_ft2: float
            building area
        '''
        
        coords = [[float(node.lat), float(node.lon)] for 
                        node in self.way.nodes]
        footprint = {'type': 'Polygon',
                    'coordinates': [coords]}
        footprint_s = shape(footprint)

        # Transform from WGS84 to Mercator
        proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
               pyproj.Proj(init='epsg:3857'))
        footprint_area = transform(proj, footprint_s).area
        
        # Convert to ft^2
        area_ft2 = footprint_area*10.764

        print(area_ft2)

        return area_ft2

class ResidentialBuilding(Building):
    '''
    Residential open street map building
    '''

    def __init__(self, way, building_usage, hdd, cdd, levels):
        super().__init__(way, building_usage, hdd, cdd, levels)

    def convert_level_data(self):
        '''
        fix the level data
        Return:
        --------
        level_data: int
           number of building levels
        '''

        levels_dict = {1: 10, 2: 20, 3:31}
        level_data = levels_dict.get(self.levels, 32)

        return level_data

    def calc_energy_usage(self, trained_model, sc):
        '''
        Calculate the energy usage
        Parameters:
        -----------
        trained_classifier:
            A trained model with a .predict() method
        sc: StandardScalar
            Scalar fitted on training data
        Returns:
        -------
        energy_usage: float
            energy usage predicted by the model
        '''

        # Get data and scale
        sqft = self.estimate_area()*self.levels
        stories = self.convert_level_data()
        building_type = self.building_usage
        data = np.array([stories, building_type, sqft, self.hdd, self.cdd]).reshape(1, -1)
        data = sc.transform(data)

        # Predict energy usage with model
        energy_usage = trained_model.predict(data)

        return energy_usage
    
class CommercialBuilding(Building):
    '''
    Commercial open street map building
    '''

    def __init__(self, way, building_usage, hdd, cdd, levels):
        super().__init__(way, building_usage, hdd, cdd, levels) 

    def calc_energy_usage(self, trained_classifier, sc):
        '''
        Calculate the energy usage
        Parameters:
        -----------
        trained_classifier:
            A trained model with a .predict() method
        sc: StandardScalar
            Scalar fitted on training data
        Returns:
        -------
        energy_usage: float
            energy usage predicted by the model
        '''

        # Get data and scale
        sqft = self.estimate_area()*self.levels
        building_type = self.building_usage
        data = np.array([sqft, building_type, self.hdd, self.cdd]).reshape(1, -1)
        data = sc.transform(data)

        # Predict energy usage with model
        energy_usage = trained_classifier.predict(data)

        return energy_usage*sqft