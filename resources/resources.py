# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import reverse_geocoder as rg


def whichCountry(coords):
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
    country_codes = pd.read_csv('data_resources/country_codes_2_3_letters.csv')
    
    result = rg.search(coords)
    
    country = list(result[0].values())[-1]

    index_of_interest = list(country_codes['2-letter code']).index(country)
    
    if index_of_interest >=54:
        UN_code = list(country_codes['3-letter code'])[index_of_interest]
    else: 
        UN_code = 'non-independent territory'
    
    return UN_code
