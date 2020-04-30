### PyTorch
The PyTorch library is widely used in research and enables full customisation of machine learning models. It requires definition of the network and dataset classes, hence the multilayer_perceptron and building_dataset scripts. The training and assessment scripts are analogous to those written using sklearn, except for the explicit inclusion of validation set in the PyTorch scripts. Updated model parameters are regularly saved during training, which enables the plotting of loss history to determine whether it can be stopped early.

#### Model evaluation
The following metrics were calculated for evaluation of the PyTorch models:
- Root mean squared error (RMSE)
- Coefficient of variation
- Symmetric mean absolute percentage error (SMAPE)
