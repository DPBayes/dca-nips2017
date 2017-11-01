'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

Modified from the original code:
  Differentially private Bayesian linear regression 
  Arttu Nieminen 2016-2017
  University of Helsinki Department of Computer Science
  Helsinki Institute of Information Technology HIIT

GDSC/drug sensitivity data

Script for combining results produced by tensor.py.

run_tensor_tests.py / tensor.py should be run before this.

Run: python3 tensorresults
'''

import numpy as np
import csv
import os.path
import pickle
import sys
from collections import OrderedDict

drug_nbo = 264 # 264 in the paper
cv_rounds = 25 # 25 in the paper

pv_size = [840] # [840] in the paper

#privacy budget as lists of same length
eps = [1.0,3.0,5.0,7.5,10.0]
delta_list = np.zeros(shape=len(eps))+10e-4

# Set folders
inpath = 'res/'   # set path for individual files from different drugs and folds
outpath = 'resultsdata/'     # set path for computed final results
inprefix = 'cliptest-drugsens-'   # set input file prefix
outprefix = 'tensor-'       # set output file prefix

indatapath = inpath+inprefix
outdatapath = outpath+outprefix
datapath = indatapath

all_means = OrderedDict()
methods = ['true', 'clipped','noisy','cl_noisy','noisy_ind','cl_noisy_ind','scaling','cl_scaling','cl_true_TA','cl_true_TA','cl_true_TA_DP']

for m in methods:
  all_means[m] = OrderedDict()
  
  means = np.zeros((cv_rounds, drug_nbo, len(eps)))
  print('array shape (cv, drugs, eps): '+str(means.shape))
  
  for k_cv in range(cv_rounds):
    for k_drug in range(drug_nbo):
      filename = datapath+str(k_drug)+'-'+str(k_cv)+'.pickle'
      with open(filename, 'rb') as f:
        apu = pickle.load(f)
      
      means[k_cv, k_drug, :] = apu[m]
      
  all_means[m]['mean'] = np.mean(means,1)
  all_means[m]['std'] = np.std(means,1)

  # Save data
with open(outpath+'drugsens_test.pickle', 'wb') as f:
  pickle.dump(all_means,f,pickle.HIGHEST_PROTOCOL)
