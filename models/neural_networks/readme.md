## Neural networks for predictive modelling of time series building energy data
Neural networks have been established as the best ML methods for predictions with commercial buildings at high temporal resolutions (Edwards et al., 2012). A feed forward neural network (FFNN) is coded here with scripts for training and evaluation (assessment). Common industry metrics (Amasyali & El-Gohary, 2018) are reported in the assessment script.

### ML Python libraries
Both Sci-Kit Learn and PyTorch examples are included here.
##### Sci-Kit Learn
The sklearn library enables beginner-level coding of some machine learning models, including FFNNs. It was used at the beginning of this project and is a recommended first port of call. The scripts using sklearn learn are indicated by their filenames (use the training script first, then the assessment script). All other scripts are for use with PyTorch, which was used in this project after neural network familiarisation using sklearn.
##### PyTorch
The PyTorch library is widely used in research and enables full customisation of machine learning models. It requires definition of the network and dataset classes, hence the multilayer_perceptron and building_dataset scripts. The training and assessment scripts are analogous to those written using sklearn, except for the inclusion of validation set in the PyTorch scripts. Updated model parameters are regularly saved during training, which enables the plotting of loss history to determine whether it can be stopped early.

### Model evaluation
The following metrics were used for evaluation of the PyTorch models:
- Root mean squared error (RMSE)
- Coefficient of variation
- Symmetric mean absolute percentage error (SMAPE)
Throughout the project, these were recorded manually in a table saved as a .csv file. The plot_model_evaluation script takes the manually compiled .csv file and generates a scatter plot of the SMAPE metric against model complexity. The classes.py file contains a class definition which is used to calculate model complexity in terms of number of parameters.
