Only the stacking scripts provided here are required to process the raw ASHRAE data. The scripts for fixing time gaps and getting buildings were written to help design functions.

In the building-stacking script, the include_meta_data variable refers to building information about the year built and square feet. If made to equal False, the produced dataset will only include weather variables for each building.
This script converts wind direction (degrees) to two features - sine of the wind direction and cosine of the wind direction. Cloud coverage, sea level pressure and precipitation depth are removed because these contain too many NaNs. Any NaNs in the remaining features are interpolated using the mean of the values either side.
For the energy (target) array, high and low extremes are removed and then any missing values are mean-interpolated as above.

Both arrays are saved as .csv files.
