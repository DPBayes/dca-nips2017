'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data

Script for calculating Bayesian linear regression from sufficient stats
'''

import numpy as np
import sys

def get_regression_est(suff_stats, pars):
  #assume suff_stats is a dictionary containing suff stats  as [X'X, X'y]
  #return dict with [prec, mean]
  
  #prior precisions
  l = 1
  l0 = 1
  
  palautettava = {}
  for k_stats in suff_stats.keys():
    apu = {}
    try:
      apu['prec'] = l*(suff_stats[k_stats][0]) + l0*np.identity(pars['dim'])
      apu['mean'] = np.linalg.solve(apu['prec'],l*(suff_stats[k_stats][1]))
    except:
      apu['prec'] = None
      apu['mean'] = None
    palautettava[k_stats] = apu
  return palautettava