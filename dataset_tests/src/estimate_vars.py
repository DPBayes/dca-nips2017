'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data, GDSC/drug sensitivity data

Function for estimating data & target marginal variances for optimal clipping when not assuming an auxiliary open dataset.
'''

import numpy as np
import sys

import sufficient_stats

def get_estimates(data, pars, small_pos=.5):
  #Note: uses 1 clip for data and 1 for target; both scaled according to individual dim std
  
  N_train = data.shape[0]
  dim = pars['dim']
  
  #clip data to the assumed data range
  data[:,0:-1] = np.sign(data[:,0:-1]) * np.minimum( np.absolute(data[:,0:-1]), pars['assumed_data_range'][0] )
  data[:,-1] = np.sign(data[:,-1]) * np.minimum( np.absolute(data[:,-1]), pars['assumed_data_range'][1] )
  
  
  eps=pars['privacy_for_marg_var']*pars['epsilon']
  delta=pars['privacy_for_marg_var']*pars['delta']
  
  sigma = np.sqrt( 1/(N_train-1) * 2*np.log(1.25/delta)) * (np.sqrt(dim*(pars['assumed_data_range'][0]**2)+pars['assumed_data_range'][1]**2) / eps)
  
  #add noise
  products = np.add(data**2, np.random.normal(0,sigma,[N_train,dim+1]) ) 
  
  vars = np.nansum(products,0)/N_train
  ind = vars <= 0
  
  #set vars to small positive numbers if negative
  if sum(ind) > 0:
    vars[ind] = small_pos
  return vars
  
