* This folder contains all scripts and documentation needed to obtain the data used in this project 

## _What's in here?_

- Datasets' short description 
- How to get these datasets
- Exploration & cleaning procedures to datasets


## _Datasets_

Our project contains multiple datasets, namely:

#### a. [ASHRAE dataset](https://www.kaggle.com/c/ashrae-energy-prediction/data):<br/>
The American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE) released via Kaggle 1448 buildings across 16 global sites, with hourly meter reading data for the year 2016

#### b. [Eia dataset](https://www.eia.gov/consumption/commercial/data/):<br/>
"The Commercial Buildings Energy Consumption Survey (CBECS) is a national sample survey that collects information on the stock of U.S. commercial buildings, including their energy-related building characteristics and energy usage data (consumption and expenditures)"

#### c. Top to bottom model datasets:<br/>
  - WorldBank's [World Development Indicators](https://databank.worldbank.org/source/world-development-indicators):
  
  - [UN's population dynamics and projections](https://population.un.org/wpp/Download/Standard/Population/) datasets:
  
  - [IEA](https://www.eia.gov/energyexplained/) (International Cooling Agency) electricity usage datasets:
  

#### d. bias correction datasets (available on JASMIN):<br/>
  - ERA (reanalysis data)
  - Climate model HadGEM-CC2-piControl-r1

## _Getting the data_

#### a.
#### To download eia data run from terminal the command:
```
sh eia_buildings/download_eia_data.sh
```
* This will download the data as a csv file in your Downloads folder.

#### b.
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
#### d.
#### To use the ERA/climate model data please see the [jasmin folder](https://github.com/michellewl/building_resilience/tree/omer/data/bias_correction/jasmin)

## _Exploring and cleaning_
- see ASHRAE dedicated [folder](https://github.com/michellewl/building_resilience/tree/omer/data/ashrae) 
- see EIA dedicated [folder](https://github.com/michellewl/building_resilience/tree/omer/other/anna)
  ##### [Example: ASHRAE exploration notebook](https://github.com/michellewl/building_resilience/blob/omer/data/ashrae/exploration/notebooks/Exploration_ASHRAE.ipynb)  
 
  
 
#### General data links:
- [Downscaled climate projections by earth engine](https://developers.google.com/earth-engine/datasets/catalog/NASA_NEX-GDDP)

