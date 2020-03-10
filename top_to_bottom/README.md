'top-to-bottom' approach
========================

# 1. Introduction

# 2. Data

+ UN population predictions: [here](https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_TotalPopulationBySex.csv)

+ WorldBank's World Development Indicators: [available here](https://databank.worldbank.org/source/world-development-indicators#). Selected metrics are: 'Access to electricity (% of population)', 'Electric power consumption (kWh per capita)', 'Population growth (annual %)', 'Population, total'.

+ IEA (International Energy Agency) data: [indirectly available here](https://www.iea.org/data-and-statistics?country=WORLD&fuel=Energy%20supply&indicator=Total%20primary%20energy%20supply%20(TPES)%20by%20source). Metrics are available yearly only by buying a license from the IEA. Free data contains the metrics with a 5-year resolution and the last year of data availability (i.e. 2000, 2005, 2010, 2015, 2017), but it has to be downloaded individually for each country and aggregated via python scripts. For the sake of simplicity, the aggregated data is avilable in the 'data' folder.

# 3. Method