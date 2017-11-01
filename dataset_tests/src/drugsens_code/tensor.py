'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

Modified from the original code:
  Differentially private Bayesian linear regression 
  Arttu Nieminen 2016-2017
  University of Helsinki Department of Computer Science
  Helsinki Institute of Information Technology HIIT

GDSC/drug sensitivity data

clippingomega.py should be run before this code.

Run: python3 tensor.py drugid seed
where 
- drugid is an integer in [0,1,...,264] (specifies drug)
- seed is an integer (specifies cv fold)
This program does 1-fold cv for given drug for one test tensor.
The cv split is defined by the given random seed.
run_tensor_tests.py is a helper script for running several drugs and CVs as in the paper.
'''

import sys
import os

import diffpri as dp
import numpy as np
import pickle
import csv
from collections import OrderedDict

# Import data
datapath = '' # add path for input and output data files
f = open(datapath+'GeneExpressionReducted.csv','rt')
reader = csv.reader(f,delimiter=',')
x = np.array(list(reader)).astype(float)
f.close()
f = open(datapath+'DrugResponse.csv','rt')
reader = csv.reader(f,delimiter=',')
y = np.array(list(reader)).astype(float)
f.close()
# For more information on the data pre-processing, see the paper "Efficient differentially private learning improves drug sensitivity prediction" (arXiv:1606.02109).

if len(sys.argv) > 1:
  drugid = int(sys.argv[1])
  seed = int(sys.argv[2])
else:
  drugid = 226
  seed = 0

# Number of samples to use
pv_size = [840] # [840] in the paper
pv_max = max(pv_size)

#privacy budget as lists of same length
eps = [1.0,3.0,5.0,7.5,10.0]
delta_list = np.zeros(shape=len(eps))+10e-4

#test set size
n_test = 100 # 100 in the paper

print('Running tensor test: drugid='+str(drugid)+', seed='+str(seed))

# Setup some parameters; see eps_data_test.py for more info
pars = {'assumed_data_range' : [1,7.5], #[1,7.5] in the paper
        #'feedback' : 0,
        'dim': 10, # 10 in the paper
        'tmp_folder' : 'tmp/',
        'add_noise' : 3,
        'scaling_comparison' : 0,
        'enforce_pos_def' : True,
        'privacy_for_marg_var' : .3, # NOTE: this should match the value in clippingomega.py; .3 in the paper
        'small_const_for_std' : .5, # .5 in the paper
        'drugsens_data' : True,
        'use_spark' : False,
        # Note: Spark version not tested with drugsens data
        'spark_filename' : 'tmp/sparktest.csv',
        'n_spark_messages' : 10,
        'spark_noise_range' : 10e13,
        'fixed_point_int' : 10e6
        }

csvpath = ''
# Fetch clipping threshold
f = open(csvpath+'C-WX.csv','rt')
reader = csv.reader(f,delimiter=',')
WX = np.array(list(reader)).astype(float)
f.close()
f = open(csvpath+'C-WY.csv','rt')
reader = csv.reader(f,delimiter=',')
WY = np.array(list(reader)).astype(float)
f.close()

#check number of missing values
inds = ~np.isnan(y[:,drugid])
n_data = np.sum(inds)
print('drugid '+str(drugid)+', has ' +str(n_data) +' target values (out of '+str(y.shape[0])+')')
y = y[inds,:]
x = x[inds,:]

res_all = OrderedDict()
models = ['true', 'clipped','noisy','cl_noisy','noisy_ind','cl_noisy_ind','scaling','cl_scaling','cl_true_TA','cl_true_TA_DP']
for m in models:
  res_all[m] = np.zeros((len(pv_size),len(eps)),dtype=np.float64)

for i in range(len(pv_size)):
  
  n_pv = pv_size[i]
  d = pars['dim']
  for j in range(len(eps)):
    pars['epsilon'] = eps[j]
    pars['delta'] = delta_list[j]
    
    w_x = WX[i,j]
    w_y = WY[i,j]
    
    # check amount of data, use maximum amount if too few samples
    if n_data < n_pv+n_test: #n_npv+n_test:
      print('Not enough non-missing data! Continuing with maximum amount of private data: ' + str(n_data-n_test))
      n_pv = n_data-n_test
    
    # Process data
    suff_stats_all,sigma_all,added_noise_dict,x_test,y_test,B_x,B_y,n_train = dp.processData(x,y,d,n_test,n_pv,pv_max,w_x,w_y,drugid,seed, pars)
    
    # calculate predictions
    for m in suff_stats_all:
      pred = dp.predictL(suff_stats_all[m][0],suff_stats_all[m][1],x_test)
      res_all[m][i,j] = dp.precision(pred,y_test)
      

with open('res/cliptest-drugsens-'+str(drugid)+'-'+str(seed)+'.pickle', 'wb') as f:
  pickle.dump(res_all, f, pickle.HIGHEST_PROTOCOL)
  
print('Done.')
