# Building Resilience
AI4ER Guided Team Challenge: Team 1 (Future demand for cooling &amp; heating of buildings in future climate scenarios)

## Background
It is generally accepted that the current climate crisis will result in an increased future global average temperature. However, at a higher spatial resolution, some parts of the globe may experience higher or lower temperatures to greater extremes. This project aims to investigate how the changing future climate will impact the energy demand of buildings, attributed to cooling and/or heating.

#### Project aims
- To make a prediction about global future energy demand of building heating and cooling, using weather data from bias-corrected global climate models (GCMs).
- To develop suitable machine learning model(s) that can be applied to individual buildings to make energy demand predictions.
- To compare performance of different prediction models, using common industry evaluation metrics.

#### Hypotheses
- A heuristic model predicts energy demand with lower error than a machine learning model.
- In future climate scenarios, the energy demand for space cooling in buildings increases (assuming current space cooling technology).

## Methodology
Ideal data would have high spatio-temporal resolution, enabling the mapping of weather data to building energy usage. Features would include information about the buildings, location and population, which are also interlinked.

Building variables might include:
- Height & size (structure)
- Materials
- Contents & use 
- Insulation (efficency and ongoing improvements)
- Infrastructure by which energy is consumed e.g. electricity, gas

Location-dependent variables including local regulations and most notably, local weather, have a large influence on building energy consumption for heating/cooling. However, the more nuanced human decision to use energy for building heating and cooling may also depend on occupation density of a building, the energy consumption culture and the availability of resources. For example, developing countries with poor access to air conditioning units may exhibit lower rates of cooling demand with increased temperature, compared to developed countries. A globally generalisable model could be applied if all of the above factors could be taken into account.

Unfortunately, such data is not widely available (and may not exist). With this in mind, we approach this problem from two key directions: top-down and bottom-up.

#### Top-down approach

This approach was used to create a simple heuristic model for predicting the energy usages for multiple countries. The datasets used are open datasets from global organisations and we use a linear regression model for each country to predict the energy usage for cooling with respect to the population of the country and the number of cooling degree days (CDDs). The linear regression is performed 3 times for each country with available data: once for the industrial sector, once for the commercial and once for the residential buildings. In the model, we assumed that those two variables (population and CDDs) are the most important ones for the problem, but other parameters could be included in future iterations, such as the country's gross domestc product.

#### Bottom-up approach
This approach uses available building energy data for a number of anonymised sites in developed countries. These datasets have good temporal resolution e.g. hourly or daily, and are good candidates for development of machine learning models. Here, we investigate a number of different ML approaches:
- Linear regression (baseline model)
- Neural networks
- XG Boost
