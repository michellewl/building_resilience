* This folder contains all scripts and documentation needed to obtain the data used in this project 

Our project contains multiple datasets, namely:
- ASHRAE dataset:
The American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE) released via Kaggle 1448 buildings across 16 global sites, with hourly meter reading data for the year 2016

- Eia buildings
- 

## _Getting the data_

#### To download eia data run from terminal the command:
```
sh eia_buildings/download_eia_data.sh
```
* This will download the data as a csv file in your Downloads folder.

#### To download ASHRAE kaggle data: 
1. Ensure you run python3, then:
```
pip3 install kaggle (on windows), pip3 install --user kaggle (mac)

```
2. To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com
3. Then go to the Account tab of your user profile (https://www.kaggle.com/<username>/account) and select Create API Token.
4. Run
  ```
   mv ~/Downloads/kaggle.json .kaggle/kaggle.json
   chmod 600 /Users/user_name/.kaggle/kaggle.json
  ```
 5. Go to data tab, accept competition terms, then:
  ```
    kaggle competitions download ashrae-energy-prediction
  ```

## _Exploring and cleaning_
- see ASHRAE dedicated [folder](https://github.com/michellewl/building_resilience/tree/omer/data/ashrae) 

#### Data exploration scripts
These scripts print information about the ASHRAE dataset which may be useful when writing further code. The first port of call should be the "quick look" script, which describes the features in the weather and building datasets.

The IEA data exploration script is specific to the IEA dataset and will not work with the ASHRAE dataset.

Plotting scripts produce graphs which may be useful when exploring the ASHRAE dataset.

  ##### [Example exploration notebook](https://github.com/michellewl/building_resilience/blob/omer/data/ashrae/exploration/notebooks/Exploration_ASHRAE.ipynb)  
  
  
 
#### General data links:
- [Downscaled climate projections by earth engine](https://developers.google.com/earth-engine/datasets/catalog/NASA_NEX-GDDP)

