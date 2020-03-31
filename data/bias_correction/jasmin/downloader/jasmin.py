# -*- coding: UTF-8 -*-
from jasmin.downloader import dataprocessing as dp
from baspy._xarray.util import extract_region
import pandas as pd
import baspy as bp

## _cm ending means climate model
## tas means surface temp

cat_model = bp.catalogue(dataset='cmip5', Model='HadGEM2-CC', Frequency='day',
                  Experiment='rcp45', RunID='r1i1p1', Var='tas').reset_index(drop=True)

for index, row in cat_model.iterrows():
	cm = bp.open_dataset(row)

tas_cm = cm.tas

lon_cor_cm = dp.roll_lon(tas_cm)

## Extract a specific region 
extr_reg_cm = extract_region(lon_cor_cm, kabul)


reg_time_sliced_cm = dp.slice_time(extr_reg_cm, 1979, 2050)

