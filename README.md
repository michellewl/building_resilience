# building_resilience
AI4ER Guided Team Challenge: Team 1 (Future demand for cooling &amp; heating of buildings in future climate scenarios)

Suspicion: a hand wavy model will perform as well or even better than a sophisticated one.

The main goal will be to infer the effect of climate model variables on the target (in our case heating/cooling energy usage). 
What is our probability model? 

In an ideal world we would have all the data we want; more concrretely, we can think of building specifics and area? specifics and population specifics. Building variables that would be of use are its height & size (structure), materials, contents, its insulation (in general efficency and ongoing temporal improvements) levels, the infrastructure by which it consumes electricity or other sources (e.g. gas). A Population (also occupancy rate) that resides in a building will matter - how many people are in a houshold? How rich are they? What is their energy consumption culture (according also to age). All of the aforementioned will relate to the location of the building through regulations in the area, the climate -> this needs to be broken down to the conditions experienced per day and per hour, the richness of the region and the culture (e.g. a location in india might be hot with poor population so even if air condition is wanted it can't be accessed). 
 - Dependency among regions seems to be not really important if all of the above variables exist in our dataset. 
 
In order to make progress, stepping away from the ideal case is inevitable.

We are building two products for individual building evaluation. 
The use case being, I don't know and I'm not sure??? 
 - comparing among building types (see MIT paper usign GP's) -> can we check it using KL divergence (that would imply belief in )
 - Predicting individual 
The use case being, I don't know and I'm not sure??? building s

When we refer to climate variables. We would like to have many variables, but most importantly it seems that we want humidity, wind, rainfall and temperature. 

Limitations: 
GDP predictions are problematic also might be scenario based
Population growth is based on scenarios 
Also energy usage is heavily affected by policy/regulations (scenarios - not limited to climate scenarios)
Data available only in the US and/or UK - seems a big leap to apply it in different locations. 

There are issues with CDD/HDD (see [link](https://www.energylens.com/articles/degree-days)):
+ "In the UK, the most readily available heating degree days come with a base temperature of 15.5°C"
+ "Fahrenheit-based degree days are 1.8 times bigger than their equivalent Celsius-based degree days"
+ "So, for a heated building, it is assumed that the energy consumption required to heat that building for any particular period is proportional to (or driven by) the number of heating degree days over that period"



Things we need to check: 
model I: 
- What is the effect of dropping all the varibale to fit to openstreet map data availability

* What is a baseline model in our case? I mean, can we do better than an area split to population density for example? 

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
How do we test it (which metrics)? why?

model III (currently heuristic): 
- If CDD/HDD is the same in year I and year 7, then energy usage is the same (time independent).
-  More concretely, CDD/HDD is the only thing that is and will change future energy usage. 
How do we test it (which metrics)? why?


Idea:
- interesting to try to understand what uncertainty comes from climate, what uncertainty comes from scenarios
- Produce a geology like map to visualize all countries in different years in one plot.
- Can we calculate the baseload for buildings - i.e. when we don't cross the thresholds for CDD/HDD

Challenges/open questions: 
- How do we combine the different climate models?



Our current project includes 2 products for individual buildings and one product for global energy usage projection.





