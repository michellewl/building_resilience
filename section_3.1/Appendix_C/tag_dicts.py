'''
Conversion of OSM tags to CBECS/RECS tags for input into trained models
'''
import numpy as np

land_use_conversion = {
    111: [2, 'commercial', 2],
    112: [2, 'residential', 1],
    121: [5, 'commercial', 1],
    123: [5, 'commercial', 1],
    124: [5, 'commercial', 1],
    142: [13, 'commercial', 2]
}

osm_tag_conversion = {
    'commercial': [2, 'commercial', 3],
    'industrial': [5, 'commercial', 1],
    'kiosk': [6, 'commercial', 1],
    'office': [2, 'commercial', 3],
    'retail': [25, 'commercial', 2],
    'supermarket': [6, 'commercial', 1],
    'warehouse': [5, 'commercial', 1],
    'civic': [13, 'commercial', 2],
    'fire_station': [7, 'commercial', 3],
    'government': [2, 'commercial', 3],
    'hospital': [16, 'commercial', 7],
    'public': [13, 'commercial', 2],
    'school': [14, 'commercial', 2],
    'kindergarten': [14, 'commercial', 2],
    'transportation': [13, 'commercial', 2],
    'university': [14, 'commercial', 2],
    'apartments': [5, 'residential', 10],
    'bungalow': [2, 'residential', 1],
    'cabin': [2, 'residential', 1],
    'detached': [2, 'residential', 1],
    'farm': [2, 'residential', 1],
    'hotel': [5, 'residential', 10],
    'house': [2, 'residential', 1],
    'residential': [2, 'residential', 1],
    'semidetached_house': [3, 'residential', 2],
    'terrace': [3, 'residential', 2],
    'static_caravan': [1, 'residential', 1]
    }

# Zero energy for
zero_usage_tags = ['shed', 'carport', 'garage', 'parking', 'ruins']