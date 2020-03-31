
#### What can be found here? 
This folder offers code for applying bias correction (currently empirical quantile mapping) for your chosen climate model and getting back the corrected .nc files. It also provides code to get the Cooling Degree Days (CDD) at each grid point from both the reanalysis data ([ERA](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-interim)) and the corrected climate model data.

##### Example output: 


#### What is bias correction and why do we do it? 
Because climate models are an imperfect representation of reality, they are biased. Thus, climate model outputs will have to be matched with observational data. This procedure is called bias correction (BC). 
However, as those biases between climate models and reality are complex, the currently available correction methods have their own limitations. In the current project, we use seasonally detrended  Empirical Quantile Mapping (ECDF)[Ho et al](https://journals.ametsoc.org/doi/pdf/10.1175/2011BAMS3110.1) to correct the data on air temperatures as its assumptions are capped to a minimum. The code contains possible extensions to other correction methods such as change factor (CF) and bias correction (BC).


#### Execution
In order to execute the following bias correction code, one first has to be connected to [JASMIN](http://www.jasmin.ac.uk). 
After you sign up to JASMIN refer to the jasmin folder for further details





##### Suggested papers:
- [Ed Hawkins et al](https://www.sciencedirect.com/science/article/pii/S0168192312001372)
- [Ho et al](https://journals.ametsoc.org/doi/pdf/10.1175/2011BAMS3110.1)
- [ALESSANDRI ET AL.](https://journals.ametsoc.org/doi/pdf/10.1175/2010MWR3417.1)


