# Differentially private Bayesian learning on distributed data

Code for running the tests in the paper "Differentially private Bayesian learning on distributed data" (arXiv:1703.01106).


## Requirements

The code uses Python3 with Numpy (tested with 1.11.1), Scipy (0.17.1), and Matplotlib (1.5.3).


## Running the tests

To run the tests using UCI data, get the Abalone and Wine Quality datasets (https://archive.ics.uci.edu/ml/datasets.html),
set the data location in UCI_data_getter.py and
use eps_data_test.py. The results can be plotted using combine_prediction_erros.py.

For the GDSC data, set the options in tensor.py and use clippingomega.py followed by run_tensor_tests.py in the drugsens_code-folder. To plot the results, run tensorresults.py followed by plot_tensor_results.py.

See the paper "Efficient differentially private learning improves drug sensitivity prediction" (arXiv:1606.02109) for more information on the GDSC data pre-processing.

For running the Spark tests, see the separate [readme at the `probic-decrypt-server` folder](../probic-decrypt-server).
