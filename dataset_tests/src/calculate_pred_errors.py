'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data

Function for calculating predictive errors.
'''

import numpy as np
from matplotlib import pyplot as plt

import data_reader

def calculate_errors(data, dim, filename_data, model_coeff):
  regr_coeff_mu = model_coeff['mean']
  regr_coeff_std = model_coeff['prec']
  if regr_coeff_mu is None:
    return None, None, None, None, None
  
  #read data to numpy array (where target = last column)
  if filename_data is not '':
    data = np.zeros((data[0], dim+1))
    apu = dataReader.read_data(filename_data)
    for i in range(len(apu)):
      data[i,:] = apu[i]
    #center data
    data = data - np.mean(data, axis = 0)
  
  #calculate predictions (MAP)
  preds = np.dot(regr_coeff_mu, np.transpose(data[:,:-1]) )
  
  #calculate errors
  MAE = np.mean( np.absolute(data[:,-1] - preds) )
  MSE = np.mean( (data[:,-1] - preds)**2 )
  
  return MAE, MSE, np.mean(preds), np.std(preds), np.amax(preds)-np.amin(preds)