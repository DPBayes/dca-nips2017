'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data, GDSC/drug sensitivity data

Function for checking & fixing matrix positive definiteness. Works by eigendecomposing, and re-composing with absolute values of the original eigenvalues.
'''

import numpy as np

def check(suff_stats, pars):
  
  if pars['enforce_pos_def'] == False:
    #simply flag non-pos.def matrices, no correction
    if pars['feedback'] > 0:
      for m in suff_stats:
        D, V = np.linalg.eig(suff_stats[m][0])
        if np.sum(D < 0) > 0:
          print('Non-positive definite Cov matrix for {}'.format(m))
    return suff_stats
  
  else:
    #eigendecompose, set eigenvalues to their absolute values & multiply back
    for m in suff_stats:
      apu = suff_stats[m][0]
      D, V = np.linalg.eig(apu)
      D = np.absolute(D)
      suff_stats[m][0] = np.dot( np.dot(V,np.diag(D)) ,np.linalg.inv(V))
  return suff_stats
