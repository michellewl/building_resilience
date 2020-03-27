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

#### Bottom-up approach
This approach uses available building energy data for a number of anonymised sites in developed countries. These datasets have good temporal resolution e.g. hourly or daily, and are good candidates for development of machine learning models. Here, we investigate a number of different ML approaches:
- Linear regression (baseline model)
- Neural networks
- XG Boost

## to edit

What is our probability model? 


In order to make progress, stepping away from the ideal case is inevitable.

We are building two products for individual building evaluation. 
The use case being, I don't know and I'm not sure??? 
 - comparing among building types (see MIT paper usign GP's) -> can we check it using KL divergence (that would imply belief in )
 - Predicting individual 
The use case being, I don't know and I'm not sure??? building s

When we refer to climate variables. We would like to have many variables, but most importantly it seems that we want dry- bulb temperature, dew point temperature, relative humidity, global solar radiation, wind speed, wind direction, degree of cloudiness, pressure, rainfall amount, and evaporation (see paper #3 from models section). 

Limitations: 
GDP predictions are problematic also might be scenario based
Population growth is based on scenarios 
Also energy usage is heavily affected by policy/regulations (scenarios - not limited to climate scenarios)
Data available only in the US and/or UK - seems a big leap to apply it in different locations. 

There are issues with CDD/HDD (see [link](https://www.energylens.com/articles/degree-days)):
+ "In the UK, the most readily available heating degree days come with a base temperature of 15.5°C"
+ "Fahrenheit-based degree days are 1.8 times bigger than their equivalent Celsius-based degree days"
+ "So, for a heated building, it is assumed that the energy consumption required to heat that building for any particular period is proportional to (or driven by) the number of heating degree days over that period"
+ "baseload" is often used to describe the total kW power of all the equipment that is on constantly
#### Different buildings have different base temperatures.
#### "people and equipment in the building. These sources contribute to an "average internal heat gain" that is typically worth around 3.5°C."
#### "the "default" base temperature in the UK is 15.5°C, whilst, in the US, it's 18°C"
####  "lighting energy consumption typically depends on the level of daylight, which varies seasonally and from day to day" - which means for example it matters if it's 18 degrees with daylight or not.
+ Heating degree days might disappear if we had to heat at night and then cool in the daytime
+ "When the outside temperature is close to the base temperature of the building, the building will often require little or no heating. Degree-day-based calculations are particularly inaccurate under such circumstance"

Things we need to check: 

- What is the effect of dropping all the varibale to fit to openstreet map data availability

* What is a baseline model in our case? I mean, can we do better than an area split to population density for example? 

* What metrics do we want to use!? 

Assumptions: 

model I (XGBoost):
 -- i.i.d data
Question: what led us to choose this model?
What variables were choosen and why? what variabels are feature engineered? 
 - If CDD/HDD is the same in year I and year 7, then energy usage is the same (time independent). 
   - this means also if any improvement in building efficiency won't be captured if it is not part of the considered variables  
 - Spatially constant (different areas with same characteristics will produce same results).
 
How do we test it (which metrics)? why?
 
model II (NN):
Question: what led us to choose this model? 
What variables were choosen and why?
- Model is not a great fit for the time series nature 
- Spatially constant (different areas with same characteristics will produce same results)
- Concentrated on education type buildings.
How do we test it (which metrics)? why?

model III (currently heuristic): 
- If CDD/HDD is the same in year I and year 7, then energy usage is the same (time independent).
-  More concretely, CDD/HDD is the only thing that is and will change future energy usage. 
How do we test it (which metrics)? why?


Idea:
- interesting to try to understand what uncertainty comes from climate, what uncertainty comes from scenarios
- Produce a geology like map to visualize all countries in different years in one plot.
- Can we calculate the baseload for buildings - i.e. when we don't cross the thresholds for CDD/HDD
   One idea would be to take the intrecept of the linear regression model. Or look at months where no heating or cooling is not needed - pay attention that different degree days will yield differernt baselines. How will we do it for unknown locations?!
 - Show that the heating and cooling for different building types is different. 
 How well does 15.5 degrees sits with the energy consumption - we have the hourly data in Michelle's data -- can we check it?
 - Look at proportional changes in consumption --> not absolute!
 - Can we comapre similar buildings in different locations to assess behavioural impact? 
 - Let's plot the energy usage conditioned on HDD/CDD marginalizing all the other parameters.
 - Let's add a simple table of correlations of varibales to energy usage

Challenges/open questions: 
- How do we combine the different climate models?

ML questions to check: 
- Leo breiman says that random forests get F in explanation - is it reliable to use the different importance measures? 



Our current project includes 2 products for individual buildings and one product for global energy usage projection.




