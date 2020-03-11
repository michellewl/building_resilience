### Assumptions 



### ML papers on the same topic 

| Papers                                                                                                                                                             | Method            | Who                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|---------------------------------------------------------|
| [paper](https://core.ac.uk/download/pdf/10128665.pdf)                                                                                                                       | Gausian process   | MIT                                                     |
| [paper](https://www.sciencedirect.com/science/article/pii/S1877343513000468)                                                                                                | Surveying non ML  | Centre for Climate Change and Sustainable Energy Policy |
| [paper](https://reader.elsevier.com/reader/sd/pii/S1364032117306093?token=20A8196FF849A681699CBD97B6D0C6F48C6700A68E5A46D9D0F6941B668B8DB5DA6A880CF32089258CC70361E52B82E4) | Surveying of ML   | Illonios university                                     |
| [paper](https://asmedigitalcollection.asme.org/solarenergyengineering/article/117/3/161/440937/Building-Energy-Use-Prediction-and-System)                                   | RNN               | Colorado university                                     |
| [paper](https://www.sciencedirect.com/science/article/pii/S0378778815302735)                                                                                                | NN/SVR            | Texas university                                        |
| [paper](https://www.sciencedirect.com/science/article/pii/S0360544218302081)                                                                                                | NN + ensembles    | South China University of Technology                    |
| [paper](https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_full.pdf)                                                                                              | IPCC report       | IPCC                                                    |



### GP paper notes:
- The paper aims to provide two products:

	1. Individual building consumption to identify outliers! - Can we at least be correct on outliers?? even if distribution is incorrect?


	2. Enter your own electricity usage --> get predictions

+	"simply providing people with feedback about their energy use can itself produce behavior changes that significantly reduce energy consumption (Darby 2006; Neenan and Robinson 2009)"

+ "focusing on the value of using logarithmic scaling for energy usage, and we use feature selection methods to determine the most relevant features for predicting consumption"..."total energy consumption roughly follows a log-normal distribution" - do we see similar patterns and also want to use log scaling? How do we produce a read out of the relevance of features from the different models we use?

+ The most important variable they found is building value! - Do we have any proxy for that?

+ They use T process (GP) - do we also have heavy tailed dist? 


- Cialdini and Schultz 2004; Allcott 2010 : show importance of normative feedback for energy efficency 	


### Surveying of ML notes (paper III):
- "the building sector represents 39% and 40% of the energy consumption and 38% and 36% of the CO2 emissions in the U.S. [1] and Europe [2], respectively"
- "CO2 emissions increased rapidly due to the increases in population and comfort demands of people"
- "Li et al. [8] reviewed the methods for building energy benchmarking and proposed a flowchart that intends to assist users in choosing the proper prediction method."
- "47% of the models focused on predicting overall energy consumption, with 31% and 20% focusing on cooling and heating energy consumption"
##### "The most commonly-used evaluation measures of energy consumption prediction models are the coefficient of variation (CV), mean absolute percentage error (MAPE), and root mean square error (RMSE)" - provides a unitless measure that is more convenient for comparison purpose.
- Occupant behavior is the greatest uncertainty in building energy consumption prediction 
- "In general, one-third of the cooling energy consumption can be saved if a good balance between natural light and solar heat can be achieved"
- "using a data-driven approach for exploratory analysis of what-if-scenarios outside of the training range may be unsuitable or may be used with caution."


