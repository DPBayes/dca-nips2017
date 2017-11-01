'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

Modified from the original code:
  Differentially private Bayesian linear regression 
  Arttu Nieminen 2016-2017
  University of Helsinki Department of Computer Science
  Helsinki Institute of Information Technology HIIT

Choose parameters for clipping using auxiliary data.

Run: python3 clippingomega.py
'''

import sys
import os

import diffpri as dp
import numpy as np
import csv

# average number of non-missing data ~ 400
pv_size = [400] # 400 in the paper

## NOTE: set these to match the values in tensor.py
#privacy budget: lists of similar length
eps = [1.0,3.0,5.0,7.5,10.0]
delta_list = np.zeros(shape=len(eps))+10e-4
np.random.seed(1)
ny = len(pv_size)
csvpath = ''   # path for output csv files
privacy_for_marg_var = .3 # .3 in the paper

nx = len(eps)
WX = np.zeros((ny,nx),dtype=np.float)
WY = np.zeros((ny,nx),dtype=np.float)
print('Finding optimal projection threshold...')
for i in range(len(pv_size)):
  for j in range(len(eps)):
    n= pv_size[i]
    d = 10
    
    e = eps[j]*(1-privacy_for_marg_var)
    delta = delta_list[j]*(1-privacy_for_marg_var)
    
    w_x,w_y = dp.omega(n,d,e,delta,method='corr',ln=10)
    WX[i,j] = w_x
    WY[i,j] = w_y
    
print('WX:\n'+str(WX))
print('WY:\n'+str(WY))
print('done!')
np.savetxt(csvpath+'C-WX.csv',WX,delimiter=',')
np.savetxt(csvpath+'C-WY.csv',WY,delimiter=',')
