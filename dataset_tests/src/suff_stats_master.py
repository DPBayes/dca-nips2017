'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data, GDSC/drug sensitivity data

Script for calculating std:s for noise for various models.
'''

import numpy as np
import sys
from collections import OrderedDict

import sufficient_stats

def get_suff_stats(data, data_clipped, n_train, k_repeat, clip_threshold, pars, data_clipped_true=None, clip_threshold_true=None, data_clipped_true_DP=None, clip_threshold_true_DP=None):

  dim = pars['dim']
############################################################
  added_noise_dict = OrderedDict()
  
  #use fixed sensitivities for UCI data; allow drugsens baseline methods to cheat a bit by using a bound calculated from the data
  if not pars['drugsens_data']:
    data_sensitivity = np.zeros(data.shape[1]-1) + pars['scale_to_range']
    target_sensitivity = pars['scale_to_range']
  else:
    data_sensitivity = np.zeros(dim) + pars['assumed_data_range'][0]
    target_sensitivity = np.ceil(np.amax(np.abs(data[:,-1])))
  
  #calculate clip products & range products between dimensions, includes factor of 2 for x_i*x_j sensitivities when i != j
  clip_prods = np.zeros((dim*(dim+1)//2) + dim)
  range_prods = np.zeros(len(clip_prods))
  if data_clipped_true is not None:
    clip_prods_true = np.zeros(len(clip_prods))
  if data_clipped_true_DP is not None:
    clip_prods_true_DP = np.zeros(len(clip_prods))
  ind = 0
  #for suff_stats X'X
  for i in range(dim):
    for ii in range(i+1):
      clip_prods[ind] = clip_threshold[i] * clip_threshold[ii]
      range_prods[ind] = data_sensitivity[i] * data_sensitivity[ii]
      if data_clipped_true is not None:
        clip_prods_true[ind] = clip_threshold_true[i] * clip_threshold_true[ii]
      if data_clipped_true_DP is not None:
        clip_prods_true_DP[ind] = clip_threshold_true_DP[i] * clip_threshold_true_DP[ii]
      #include factor of 2 from sensitivity for non-diagonal terms
      if i != ii:
        clip_prods[ind] *= 2
        range_prods[ind] *= 2
        if data_clipped_true is not None:
          clip_prods_true[ind] *= 2
        if data_clipped_true_DP is not None:
          clip_prods_true_DP[ind] *= 2
      ind = ind + 1
  #for suff stats X'y
  for i in range(dim):
    clip_prods[ind] = 2*clip_threshold[i] * clip_threshold[-1]
    range_prods[ind] = 2*data_sensitivity[i] * target_sensitivity
    if data_clipped_true is not None:
      clip_prods_true[ind] = 2*clip_threshold_true[i] * clip_threshold_true[-1]
    if data_clipped_true_DP is not None:
      clip_prods_true_DP[ind] = 2*clip_threshold_true_DP[i] * clip_threshold_true_DP[-1]
    ind = ind + 1
  
  #total l2-sensitivities for noise std calculations
  clip_sensitivity = np.sqrt( np.sum(clip_prods[0:(dim*(dim+1)//2)]**2) + np.sum(clip_prods[(dim*(dim+1)//2):]**2) )
  
  range_sensitivity = np.sqrt( np.sum(range_prods[0:(dim*(dim+1)//2)]**2) + np.sum(range_prods[(dim*(dim+1)//2):]**2) )
  
  if data_clipped_true is not None:
    clip_sensitivity_true = np.sqrt( np.sum(clip_prods_true[0:(dim*(dim+1)//2)]**2) + np.sum(clip_prods_true[(dim*(dim+1)//2):]**2) )
  
  if data_clipped_true_DP is not None:
    clip_sensitivity_true_DP = np.sqrt( np.sum(clip_prods_true_DP[0:(dim*(dim+1)//2)]**2) + np.sum(clip_prods_true_DP[(dim*(dim+1)//2):]**2) )
  
  sigma_all = OrderedDict()
  suff_stats_all = OrderedDict()
  
  eps=(1-pars['privacy_for_marg_var'])*pars['epsilon']
  delta=(1-pars['privacy_for_marg_var'])*pars['delta']
  eps_no_clip = pars['epsilon']
  delta_no_clip = pars['delta']
  
  if pars['add_noise'] in [1,3]:
#trusted aggregator noise
############################################################
    
    #clipped
    sigma_all['cl_noisy'] = np.sqrt( 1/n_train * 2*np.log(1.25/delta) ) * (clip_sensitivity/eps)
    #clipped true TA (non DP, i.e., doesn't spend privacy on clipping bounds)
    if data_clipped_true is not None:
      sigma_all['cl_true_TA'] = np.sqrt( 1/n_train * 2*np.log(1.25/delta_no_clip) ) * (clip_sensitivity_true/eps_no_clip)
    #clipped true TA (DP)
    if data_clipped_true_DP is not None:
      sigma_all['cl_true_TA_DP'] = np.sqrt( 1/n_train * 2*np.log(1.25/delta) ) * (clip_sensitivity_true_DP/eps)
    #no clipping
    sigma_all['noisy'] = np.sqrt( 1/n_train * 2*np.log(1.25/delta_no_clip) ) * (range_sensitivity/eps_no_clip)
    
    #calculate sufficient stats for clipped & unclipped data
    ss1, ss2, ss_cl1, ss_cl2, noise, noise_cl = None, None, None, None, None, None
    ss1, ss2, noise = sufficient_stats.ss_individually(data, add_noise=True, sigma=sigma_all['noisy'], use_spark=False)
    ss_cl1, ss_cl2, noise_cl = sufficient_stats.ss_individually(data_clipped, add_noise=True, sigma=sigma_all['cl_noisy'], use_spark=False)
    
    #cl true TA (not DP)
    if data_clipped_true is not None:
      ss_cl_true1, ss_cl_true2, noise_cl_true = sufficient_stats.ss_individually(data_clipped_true, add_noise=True, sigma=sigma_all['cl_true_TA'], use_spark=False)
    #cl true TA (DP)
    if data_clipped_true_DP is not None:
      ss_cl_true_DP1, ss_cl_true_DP2, noise_cl_true_DP = sufficient_stats.ss_individually(data_clipped_true_DP, add_noise=True, sigma=sigma_all['cl_true_TA_DP'], use_spark=False)
    
    suff_stats_all['noisy'] = [ss1, ss2]
    suff_stats_all['cl_noisy'] = [ss_cl1, ss_cl2]
    added_noise_dict['noisy'] = noise
    added_noise_dict['cl_noisy'] = noise_cl
    #cl true TA (not DP)
    if data_clipped_true is not None:
      suff_stats_all['cl_true_TA'] = [ss_cl_true1, ss_cl_true2]
      added_noise_dict['cl_true_TA'] = noise_cl_true
    #cl true TA (DP)
    if data_clipped_true_DP is not None:
      suff_stats_all['cl_true_TA_DP'] = [ss_cl_true_DP1, ss_cl_true_DP2]
      added_noise_dict['cl_true_TA_DP'] = noise_cl_true_DP
    
#with extra scaling factor for percentage honest clients
############################################################
    #calculate scaling factor
    if pars['scaling_comparison'] == 0:
      scaling = 1
    else:
      scaling = 1/(np.ceil( pars['scaling_comparison']*n_train) )
    #add noise in pieces separately by each client
    #noise std for X'X
    sigma_all['cl_scaling'] = np.sqrt( scaling * 2*np.log(1.25/delta) ) * (clip_sensitivity/eps)
    #noise std for X'y
    sigma_all['scaling'] = np.sqrt( scaling * 2*np.log(1.25/delta_no_clip) ) * (range_sensitivity/eps_no_clip)
    
    #calculate sufficient stats for clipped & unclipped data with extra scaling
    ss1, ss2, ss_cl1, ss_cl2, noise, noise_cl = None, None, None, None, None, None
    ss1, ss2, noise = sufficient_stats.ss_individually(data, add_noise=True, sigma=sigma_all['scaling'], use_spark=False)

    ss_cl1, ss_cl2, noise_cl = sufficient_stats.ss_individually(data_clipped, add_noise=True, sigma=sigma_all['cl_scaling'], use_spark=False)
    
    
    suff_stats_all['scaling'] = [ss1, ss2]
    suff_stats_all['cl_scaling'] = [ss_cl1, ss_cl2]
    added_noise_dict['scaling'] = noise
    added_noise_dict['cl_scaling'] = noise_cl

#individual noise i.e. n/(n-1) factor with clipped and unclipped data
############################################################
  if pars['add_noise'] in [2,3]:
    #clipped data
    sigma_all['cl_noisy_ind'] = np.sqrt( 1/(n_train-1) * 2*np.log(1.25/delta) ) * (clip_sensitivity/eps)
    
    #unclipped data
    sigma_all['noisy_ind'] = np.sqrt( 1/(n_train-1) * 2*np.log(1.25/delta_no_clip) ) * (range_sensitivity/eps_no_clip)
    
    #calculate sufficient stats for clipped & unclipped data
    # Note: unclipped used for Spark testing
    ss1, ss2, ss_cl1, ss_cl2, noise, noise_cl = None, None, None, None, None, None
    ss1, ss2, noise = sufficient_stats.ss_individually(data, add_noise=pars['add_noise'] > 0, sigma=sigma_all['noisy_ind'], use_spark=pars['use_spark'], filename=pars['spark_filename'], n_spark_messages=pars['n_spark_messages'], spark_noise_range=pars['spark_noise_range'], fixed_point_int=pars['fixed_point_int'])
    
    ss_cl1, ss_cl2, noise_cl = sufficient_stats.ss_individually(data_clipped, add_noise=pars['add_noise'] > 0, sigma=sigma_all['cl_noisy_ind'], use_spark=False)
    
    suff_stats_all['noisy_ind'] = [ss1, ss2]
    suff_stats_all['cl_noisy_ind'] = [ss_cl1, ss_cl2]
    added_noise_dict['noisy_ind'] = noise
    added_noise_dict['cl_noisy_ind'] = noise_cl
    
############################################################
  #calculate noiseless sufficient statistics for comparison
  #X'X
  suff_stats_all['true'] = list()
  suff_stats_all['true'].append(np.dot(np.transpose(data[:,0:-1]), data[:,0:-1]))
  #X'y
  suff_stats_all['true'].append(np.dot(np.transpose(data[:,0:-1]), data[:,-1]))

  #suff.stats for the noiseless clipped data
  suff_stats_all['clipped'] = list()
  #X'X
  suff_stats_all['clipped'].append(np.dot(np.transpose(data_clipped[:,0:-1]), data_clipped[:,0:-1]))
  #X'y
  suff_stats_all['clipped'].append(np.dot(np.transpose(data_clipped[:,0:-1]), data_clipped[:,-1]))
  
  return suff_stats_all, sigma_all, added_noise_dict
