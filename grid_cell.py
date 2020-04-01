import numpy as np
import json
import overpy
from tag_dicts import *
from buildings import *

class GridCell():
    '''
    Object representing one grid cell in the global energy maps
    '''

    def __init__(self, lat_b, lat_t, lon_l, lon_r, land_use, hdd, cdd):
        '''
        Parameters:
        -----------
        lat_b, lat_t, lon_l, lon_r: float
            lat lon coords of the grid cell
        land_use: int
            CORINE land use code
        hdd: float
            heating degree days
        cdd: float
            cooling degree days
        '''
        self.land_use = land_use
        self.api = overpy.Overpass()
        self.query = """[out:json][timeout:25];
            (
            way[building]({},{},{},{});
            node(w);
            );
            out body;
            >;
            """.format(lat_b, lon_l, lat_t, lon_r)
        self.buildings = self.api.query(self.query)
        self.way_ids = self.buildings.get_way_ids()
        self.hdd = hdd
        self.cdd = cdd
        self.res_buildings, self.com_buildings = self.make_building_list()

    def get_building_usage(self, tags):
        '''
        get the usage, residential/commercial classification
        and height for input into the model
        Parameters:
        -----------
        tags: dict
            building info tags from API query
        Returns:
        --------
        building_usage: int
            CBECS/RECS building usage code
        res_com: String
            residential or commercial
        default_height: int
            number of levels in building
        '''

        tag = tags['building']

        if tag == 'yes':
            building_usage, res_com, default_height = land_use_conversion.get(self.land_use, [2, 'residential', 1])

        else:
            # Check the weird place of worship thing
            if tags.get('amenity') == 'place_of_worship':
                building_usage = 12
                res_com = 'commercial'
                default_height = 1
            else:
                building_usage, res_com, default_height = osm_tag_conversion.get(tag, [2, 'residential', 1])

        # Override levels if exact values available
        exact_levels = self.override_height(tags)
        if exact_levels:
            default_height = exact_levels

        return building_usage, res_com, default_height

    def override_height(self, tags):
        '''
        override the height for buildings with levels specified
        Parameters:
        -----------
        tags: dict
            building info tags from API query
        Returns:
        --------
        levels: int
            number of building levels
        '''

        building_levels = tags.get('building:levels', False)
        levels = tags.get('levels', False)
        height = tags.get('height', False)
        try:
            if building_levels:
                return int(building_levels)
            elif levels:
                return int(levels)
            elif height:
                return float(height)/4.5
        except ValueError:
            return None
        return None

    def make_building_list(self):
        ''' 
        create a list of all the buildings associated with a grid cell
        Returns:
        --------
        res_buildings: list
            list of ResidentialBuilding objects
        com_buildings: list
            list of CommercialBuilding objects
        '''

        res_buildings = []
        com_buildings = []

        print("Retrieving building lists")

        # Iterate over each way 
        for way_id in self.way_ids:

            print("Retrieving building {}".format(way_id))

            way = self.buildings.get_way(way_id)
            tags = self.buildings.get_way(way_id).tags

            # Handle the different cases
            building_usage, res_com, levels = self.get_building_usage(tags)

            # Add building to appropriate list
            if res_com == 'residential':
                new_building = ResidentialBuilding(way, building_usage, self.hdd, self.cdd, levels)
                res_buildings.append(new_building)
            else:
                new_building = CommercialBuilding(way, building_usage, self.hdd, self.cdd, levels)
                com_buildings.append(new_building)

        return res_buildings, com_buildings

    def get_res_energy(self, hc_classifier, sc):
        '''
        retrieve the total energy usage of all residential buildings
        in the cell
        Parameters:
        -----------
        hc_classifier: 
            trained model with .predict() method
        sc: StandardScalar
            scalar fitted to training data
        Returns:
        --------
        res_energy: float
            total residential energy usage by buildings in cell
        '''

        res_energy = 0

        for building in self.res_buildings:
            energy = building.calc_energy_usage(hc_classifier, sc)
            res_energy += energy

        return res_energy

    def get_com_energy(self, hc_classifier, sc):
        '''
        retrieve the total energy usage of all commercial buildings
        in the cell
        Parameters:
        -----------
        hc_classifier: 
            trained model with .predict() method
        sc: StandardScalar
            scalar fitted to training data
        Returns:
        --------
        com_energy: float
            total residential energy usage by buildings in cell
        '''


        com_energy = 0

        for building in self.com_buildings:
            energy = building.calc_energy_usage(hc_classifier, sc)
            com_energy += energy

        return com_energy
