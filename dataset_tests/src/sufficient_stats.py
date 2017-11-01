'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data, GDSC/drug sensitivity data

Function for calculating sufficient statistics with perturbations added by  individual clients.
'''

import csv
import numpy as np
import sys

def ss_individually(data, add_noise=False, sigma=None, use_spark = False, filename = None, n_spark_messages=None, spark_noise_range=None, fixed_point_int=None):
  
  k_clients = data.shape[0]
  dim = data.shape[1]-1 #dimensions without target
  
  added_noise = np.zeros((k_clients, dim*(dim+1)//2+dim))
  
  #construct the products in X'X and X'y individually for each client, with or without added noise
  products = np.zeros([k_clients, dim*(dim+1)//2 + dim] )
  ind = 0
  added_noise = np.zeros((k_clients, dim*(dim+1)//2+dim))
  added_noise = np.random.normal(0,sigma, (k_clients, dim*(dim+1)//2+dim))
  
  #suff_stat1
  for i in range(dim):
    for ii in range(i+1):
      products[:,ind] = (data[0:k_clients,i] * data[0:k_clients,ii])
      ind += 1
  #suff_stat2
  for i in range(dim):
    products[:,ind] = (data[0:k_clients,i] * data[0:k_clients,-1])
    ind += 1
  
  if not add_noise:
    added_noise = 0
  
  products += added_noise
  
  # save test data for Spark
  if use_spark and filename is not None:
    # use fixed-point representation in the noisy messages
    products = np.floor(products*fixed_point_int).astype('int64')
    # save as a matrix with n_clients rows and n_messages*suff_stats-dim columns, s.t. first n_messages cols correspond to the first element in the sufficient stats
    noisy_matrix = np.zeros((products.shape[0],products.shape[1]*n_spark_messages),dtype='int64')
    for i in range(products.shape[1]):
      noise = np.random.randint(-spark_noise_range,spark_noise_range, (products.shape[0],n_spark_messages-1) ,dtype='int64')
      noisy_matrix[:,n_spark_messages*i] = products[:,i]
      noisy_matrix[:,n_spark_messages*i:(n_spark_messages*(i+1)-1)] += noise
      noisy_matrix[:,n_spark_messages*(i+1)-1] = -np.sum(noise,1)
    np.savetxt(filename, noisy_matrix, delimiter=';')
    print('Saved sufficient statistics to file for Spark:\n {}'.format(filename))
    sys.exit()
  
  noisy_sum = np.sum(products, axis=0)
  #suff stats for X'X
  suff_stat1 = np.zeros([dim,dim])
  suff_stat1[np.tril_indices(dim,0)] = noisy_sum[0:dim*(dim+1)//2]
  suff_stat1 = suff_stat1 + np.triu(np.transpose(suff_stat1),k=1)
  #suff stat for X'y
  suff_stat2 = np.zeros([dim,1])
  suff_stat2 = noisy_sum[(dim*(dim+1)//2):]
  
  return suff_stat1, suff_stat2, added_noise

