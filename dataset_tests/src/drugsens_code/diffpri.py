'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

Modified from the original code:
  Differentially private Bayesian linear regression 
  Arttu Nieminen 2016-2017
  University of Helsinki Department of Computer Science
  Helsinki Institute of Information Technology HIIT
  
GDSC/drug sensitivity data

Various functions and data processing steps used in the tests.
'''

import sys, os, copy
import numpy as np
from scipy.stats import spearmanr
import warnings

# NOTE on normalisation in distributed setting:
#   assume centered data (so remove column means)
#   row-wise L2-normalization is ok, since doesn't depend on other rows

# Centers and L2-normalises x-data (removes columnwise mean, normalises rows to norm 1)
def xnormalise(x):
  n = x.shape[0]
  d = x.shape[1]
  if n == 0:
    return x
  else:
    z = x-np.dot(np.ones((n,1),dtype=np.float),np.nanmean(x,0).reshape(1,d))
    return np.divide(z,np.dot(np.sqrt(np.nansum(np.power(z,2.0),1)).reshape(n,1),np.ones((1,d),dtype=np.float)))


# Centers y-data (removes columnwise mean, except for columns where all samples have / all but one sample has missing drug response(s))
def ynormalise(y):
  n = y.shape[0]
  d = y.shape[1]
  if n == 0:
    return y
  else:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      m = np.nanmean(y,0)
    ind = np.where(np.sum(~np.isnan(y),0)<=1)[0]
    m[ind] = 0.0 # don't center samples of size <= 1
    return y-np.dot(np.ones((n,1),dtype=np.float),m.reshape(1,d))


# Clip data
def clip(x,y,B_x,B_y):
  C = np.multiply(np.sign(x),np.minimum(np.abs(x),B_x))
  with np.errstate(invalid='ignore'):
    D = np.multiply(np.sign(y),np.minimum(np.abs(y),B_y))
  return C,D


# Selects drug based on drugid, removes cell lines with missing drug response
def ignoreNaN(xx,yy,drugid):
  ind = np.where(np.isnan(yy[:,drugid]))
  y = np.delete(yy[:,drugid],ind,axis=0)
  x = np.delete(xx,ind,axis=0)
  return x,y


# Non-private sufficient statistics
def nxx(x):
  return np.dot(x.T,x)
def nxy(x,y):
  return np.dot(x.T,y)
def nyy(y):
  return np.dot(y.T,y)


# Precision measure: Spearman's rank correlation coefficient
def precision(y_pred,y_real):
  r = spearmanr(y_pred,y_real)[0]
  if np.isnan(r):
    return 0.0
  else:
    return r


# Prediction errors (MAE, MSE) helper script
def pred_errors(pred, y, method):
  if method == 'mae':
    return np.mean(np.absolute(pred-y))
  elif method =='mse':
    return np.mean((pred-y)**2)


# Choose optimal w_x,w_y for clipping thresholds
def omega(n,d,eps,delta, method='corr',ln=20):
  
  # Precision parameters (correspond to the means of the gamma hyperpriors)
  l = 1.0
  l0 = 1.0
  
  l1 = ln
  l2 = ln
  
  st = np.arange(0.1,2.1,0.1)
  lenC1 = len(st)
  lenC2 = lenC1
  err = np.zeros((lenC1,lenC2),dtype=np.float64)
  
  for i in range(l1):

    # Create synthetic data
    x = np.random.normal(0.0,1.0,(n,d))
    x = xnormalise(x)
    sx = np.std(x,ddof=1)
    b = np.random.normal(0.0,1.0/np.sqrt(l0),d)
    y = np.random.normal(np.dot(x,b),1.0/np.sqrt(l)).reshape(n,1)
    y = ynormalise(y)
    sy = np.std(y,ddof=1)
    
    # Thresholds to be tested
    cs1 = st*sx
    cs2 = st*sy
    
    for j in range(l2):
      
      apu2 = np.random.normal(loc=0,
      scale=np.sqrt(n/(n-1)*2*np.log(1.25/delta)) * 1/eps,
      size=d*(d+1)//2+d)
      
      U = np.zeros((d,d))
      U[np.tril_indices(d,0)] = apu2[:d*(d+1)//2]
      U =  U + np.triu(np.transpose(U),k=1)
      V = apu2[d*(d+1)//2:].reshape((d,1))
      
      for ci1 in range(lenC1):
        c1 = cs1[ci1]
        for ci2 in range(lenC2):
          c2 = cs2[ci2]
          
          # Clip data
          xc,yc = clip(x,y,c1,c2)
          sensitivity = d*c1**4 + d*(d-1)*2*c1**4 + d*(2*c1*c2)**2
          
          # Perturbed suff.stats
          xx = nxx(xc) + U*(sensitivity**2)
          xy = nxy(xc,yc) + V*(sensitivity**2)
          
          # Prediction
          prec = l0*np.identity(d) + l*xx
          mean = np.linalg.solve(prec,l*xy)
          pred = np.dot(x,mean)
          
          # Errors
          if method == 'corr':
            rho = precision(pred,y)
            err[ci1,ci2] = err[ci1,ci2] + rho
          elif method == 'mae':
            MAE = pred_errors(pred,y,'mae')
            err[ci1,ci2] = err[ci1,ci2] - MAE
          elif method == 'mse':
            MSE = pred_errors(pred,y,'mse')
            err[ci1,ci2] = err[ci1,ci2] - MSE
          else:
            print('Unknown method in optimal clip!')
            sys.exit()

  # Average
  err = err/float(l1*l2)
  # Choose best
  ind = np.unravel_index(err.argmax(),err.shape)
  w_x = st[ind[0]]
  w_y = st[ind[1]]

  return w_x,w_y


# Prediction on test data
def predictL(nxx_pv,nxy_pv,x_test):
  l = 1.0
  l0 = 1.0
  d = nxx_pv.shape[0]
  # Posterior for Gaussian
  prec = l*(nxx_pv) + l0*np.identity(d)
  mean = np.linalg.solve(prec,l*(nxy_pv))
  # Compute prediction
  return np.dot(x_test,mean)


def estimate_stds(data,pars):
  PACKAGE_PARENT = '..'
  SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
  sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
  from estimate_vars import get_estimates
  return np.sqrt(get_estimates(data, pars, pars['small_const_for_std']))

def suff_stats_crypto(data, data_clipped, n_train, k_repeat, clip_threshold, pars, data_clipped_true=None, clip_threshold_true=None, data_clipped_true_DP=None, clip_threshold_true_DP=None):
  from suff_stats_master import get_suff_stats
  return get_suff_stats(data, data_clipped, n_train, k_repeat, clip_threshold, pars, data_clipped_true, clip_threshold_true, data_clipped_true_DP, clip_threshold_true_DP)

def enforce_pos_def(suff_stats, pars):
  from pos_def_matrices import check
  return check(suff_stats, pars)


def get_TA_std_estimates(data, pars):
  palautettava = np.var(data, 0)
  #use Gaussian mechanism for DP
  eps = pars['privacy_for_marg_var']*pars['epsilon']
  delta = pars['privacy_for_marg_var']*pars['delta']
  n = data.shape[0]
  dim = data.shape[1] - 1
  if pars['drugsens_data']:
    data_bound = np.ceil(np.amax(np.abs(data),0))
    sigma = np.sqrt( 2*np.log(1.25/delta)) * 1/n * np.sqrt((dim*(pars['assumed_data_range'][0]**2) + data_bound[-1]**2))/eps
  else:
    sigma = np.sqrt( 2*np.log(1.25/delta)) * 1/n *np.sqrt( (dim+1)*(pars['scale_to_range']**2))/eps
  
  #add noise
  palautettava = palautettava + np.random.normal(0, sigma, [data.shape[1]])
  #constrain stds to be positive
  inds = palautettava <= 0
  if len(inds) > 0:
    palautettava[inds] = pars['small_const_for_std'] #set non-positive std to small arbitrary constant
  return np.sqrt(palautettava)


# Process drugsens data
def processData(x,y,d,n_test,n_pv,pv_max,w_x,w_y,drugid,seed, pars):
  
  n_train = n_pv
  
  # Set rng seed
  np.random.seed(seed)
  
  # Test/training split + dimensionality reduction
  ind = np.random.permutation(x.shape[0])
  x_test = x[ind[0:n_test],0:d]
  y_test = y[ind[0:n_test],:]
  x_train = x[ind[n_test:],0:d]
  y_train = y[ind[n_test:],:]

  # Training data
  x_pv = x_train[0:n_pv,:]
  y_pv = y_train[0:n_pv,:]
  
  # Normalise x-data (remove mean and L2-normalize)
  x_test = xnormalise(x_test)
  x_pv = xnormalise(x_pv)
  
  # Normalise y-data (remove mean)
  y_test = ynormalise(y_test)
  y_pv = ynormalise(y_pv)

  
  # get marginal std estimates for clipping
  data = np.copy(np.hstack( (x_pv, y_pv[:,drugid].reshape(y_pv.shape[0],1)) ))
  
  stds = estimate_stds(np.copy(data), pars)
  
  #true std for comparison
  stds_true = np.std(data, 0)
  
  #DP std estimates for TA
  stds_TA = get_TA_std_estimates(np.copy(data), pars)
  
  # Clip data
  n = np.sum(~np.isnan(y_pv[:,drugid]))
  
  x_pv_orig = np.copy(x_pv)
  y_pv_orig = np.copy(y_pv)
  
  if n == 1:
    B_x = np.max(np.abs(x_pv))
    B_y = np.nanmax(np.abs(y_pv))
    x_pv,y_pv = clip(x_pv,y_pv,B_x,B_y)
    print('\nn==1!\n')
    
  elif n > 1:
    B_x = w_x * stds[0:-1]
    B_y = w_y * stds[-1]
    
    B_x_true = w_x * stds_true[0:-1]
    B_y_true = w_y * stds_true[-1]
    
    B_x_true_DP = w_x * stds_TA[0:-1]
    B_y_true_DP = w_y * stds_TA[-1]
    x_pv,y_pv = clip(x_pv,y_pv,B_x,B_y)
    
    x_pv_true,y_pv_true = clip(np.copy(x_pv_orig),np.copy(y_pv_orig),B_x_true,B_y_true)
    
    x_pv_true_DP,y_pv_true_DP = clip(np.copy(x_pv_orig),np.copy(y_pv_orig),B_x_true_DP,B_y_true_DP)
    
  else:
    B_x = 0.0
    B_y = 0.0
  
  # Select drug and drop cell lines with missing response
  x_pv,y_pv = ignoreNaN(x_pv,y_pv,drugid)
  x_test,y_test = ignoreNaN(x_test,y_test,drugid)
  n_train = x_pv.shape[0]
  x_pv_true,y_pv_true = ignoreNaN(x_pv_true,y_pv_true,drugid)
  x_pv_true_DP,y_pv_true_DP = ignoreNaN(x_pv_true_DP,y_pv_true_DP,drugid)
  
  # Compute suff.stats
  data_clipped = np.hstack( (x_pv, y_pv.reshape(y_pv.shape[0],1)) )
  data_clipped_true = np.hstack( (x_pv_true, y_pv_true.reshape(y_pv_true.shape[0],1)) )
  data_clipped_true_DP = np.hstack( (x_pv_true_DP, y_pv_true_DP.reshape(y_pv_true_DP.shape[0],1)) )
  
  
  suff_stats_all, sigma_all, added_noise_dict = suff_stats_crypto(data, data_clipped, n_train, 0, np.hstack((B_x,B_y)), pars, data_clipped_true, np.hstack((B_x_true,B_y_true)), data_clipped_true_DP, np.hstack((B_x_true_DP,B_y_true_DP)) )

  #enforce pos.def. Cov matrices
  suff_stats_all = enforce_pos_def(suff_stats_all, pars)
  
  return suff_stats_all, sigma_all, added_noise_dict, x_test, y_test, B_x, B_y, n_train





