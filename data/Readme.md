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



#### Interesting data links:
- [Downscaled climate projections by earth engine](https://developers.google.com/earth-engine/datasets/catalog/NASA_NEX-GDDP)


#### Data exploration scripts
These scripts print information about the ASHRAE dataset which may be useful when writing further code. The first port of call should be the "quick look" script, which describes the features in the weather and building datasets.

The IEA data exploration script is specific to the IEA dataset and will not work with the ASHRAE dataset.

Plotting scripts produce graphs which may be useful when exploring the ASHRAE dataset.

  ##### [Example exploration notebook](https://github.com/michellewl/building_resilience/blob/omer/data/ashrae/exploration/notebooks/Exploration_ASHRAE.ipynb)  
