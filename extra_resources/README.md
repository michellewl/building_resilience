# Resources: common functions used in the project

prerequisites/packages needed:

pandas, reverse_geocoder

+ to install reverse_geocoder: *pip install reverse_geocoder*, or use a condas environment containing the package;

1. whichCountry(coords):

returns the 3-letter UN country code from the lat/lon coordinates;
tuple required for the operation: coord = (lat, lon);
+ **don't forget to change the path to the required datafile**