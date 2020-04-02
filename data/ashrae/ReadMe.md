## _What's in here?_

This folder contains scripts for exploring and cleaning the ASHRAE [dataset](https://www.kaggle.com/c/ashrae-energy-prediction/data);</br> 
in the exploration folder you'll find exploration both on a daily level and an hourly level. 


| Pre-processing step                                                                                                                                                          | Justification |---|---|Missing values (NaN) in the raw weather and energy hourly data were interpolated using the mean of the values in the previous and following hour| This would account for any occasional instances of corrupted/missing data without changing the overall behaviour.Extreme values in energy readings were removed by retaining the middle 99.9% values of the total raw dataset. These were replaced by NaN and then filled in as above.|                                                                                               

### Daily data cleaning step

| Pre-processing step                                                                                                                                                             | Justification      
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|---
| Missing values (NaN) in the raw weather and energy hourly data were interpolated using the mean of the values in the previous and following hour| This would account for any occasional instances of corrupted/missing data without changing the overall behaviour.                                                      |
| Extreme values in energy readings were removed by retaining the middle 99.9% values of the total raw dataset. These were replaced by NaN and then filled in as above.                                                                                               | The presence of extreme values (multiple magnitudes greater than the rest of the data) was identified during data exploration and perceived as possible errors at the point of data collection.
| Some weather features were removed from the analysis entirely | Many sites lacked data for these weather features. 
| Meter readings provided as kBTU were converted to kWh per square foot.                                 | Energy was converted to more commonly used units and calculated per unit of area, to account for buildings of different size. Note that building area (square feet) was retained as an input feature.               
| Two sites (7, 9) with unusual energy reading behaviour were visually identified during data exploration and removed from further analysis.                                                                                               | Possible explanations for the behaviour of these sites have yet to be resolved.          
| Long time periods (2-6 months) with missing data were visually identified during data exploration and removed from further analysis (some buildings at sites 0, 15).                                                                                              | It was perceived that these may have been instances of disused buildings.                 |
| The dewpoint temperature input feature was converted to units of relative humidity (RH).                                                                                             | This step was taken to enable the application of neural network models to global climate model outputs, which use RH.|   

### Hourly data  - cleaning & exploration

An exploration notebook of the hourly data and the consequent actions for cleaning can be found [here](https://github.com/michellewl/building_resilience/blob/michelle/branch2/data/ashrae/exploration/notebooks/Exploration_ASHRAE.ipynb)



