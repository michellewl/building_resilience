'top-to-bottom' approach
========================

[here]: https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_TotalPopulationBySex.csv

# 1. Introduction


(to be added from IEA cooling and other papers)


# 2. Data

The complete data folder used in this section of the project can be found on [this GoogleDrive link](https://drive.google.com/drive/folders/1cA-fF1VtLZZ0a7SzFtCLNodLaiOOyvmn?usp=sharing). It has the exact data that goes into the further analysis and mainly contains on those datasets:

+ UN population predictions: [here][];

+ !!!!!!!!UN household data: [downloadable here](https://population.un.org/household/exceldata/population_division_UN_Houseshold_Size_and_Composition_2019.xlsx);

+ WorldBank's World Development Indicators: [available here](https://databank.worldbank.org/source/world-development-indicators#). Selected metrics are: 'Access to electricity (% of population)', 'Electric power consumption (kWh per capita)', 'Population growth (annual %)', 'Population, total';

+ IEA (International Energy Agency) data: [indirectly available here](https://www.iea.org/data-and-statistics?country=WORLD&fuel=Energy%20supply&indicator=Coal%20production%20by%20type). Metrics are available yearly only by buying a license from the IEA. Free data contains the metrics with a 5-year resolution and the last year of data availability (i.e. 2000, 2005, 2010, 2015, 2017), but it has to be downloaded individually for each country and aggregated via python scripts. For the sake of simplicity, the aggregated data for the available years is readily available in the GoogleDrive folder.

+ 2 'extra datasets' containing miscellaneous information like country names written in different formats and thier short country codes.

+ (to add, final details about the climate models used for the CDD - cooling degree days)


# 3. Method


1. Calculate the electricity usage, for each sector, for cooling, from the available data as being equal to: population X access to electricity X electricity usage per capita X percentage of electricity usage for each sector X percentage of electricity usage for cooling.
2. Calculate from ERA5 reanalysis data and from bias corrected climate models the CDD in each 1x1 degrees cell ()
3. Use the electricity output from 1, the CDD data from 2 and the population data from UN to make a simple linear regression model that can predict the electricity usage in the future.


# 4. Future Work

- get more accuracy by using climate models outputs with greater granularity;
- use Gaussian Processes instead linear regression to predict the future electricity usages;
- IEA has an open data policy that constrains us to using data sampled every 5 years and the last year with data;
- 20% is the estimated use of electricity for cooling across all sectors in, all countries; this is based on estimations from the IPCC report but it needs more attention;

