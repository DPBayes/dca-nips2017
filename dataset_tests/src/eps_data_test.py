'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data (abalone, red wine, white wine)

Script for testing distributed Bayesian learning on UCI datasets.

Run: python3 eps_data_test.py
'''

import getopt
import numpy as np
import pickle
import os
import re
import sys
from time import sleep

import calculate_pred_errors
import estimate_vars
import linear_regression_master
import pos_def_matrices
import setup_handler
import suff_stats_master
import UCI_data_getter
from drugsens_code import diffpri as dp

pars = {}
################################################################################
# SETUP
################################################################################
# Use setup-script
# Note: overrides all the options in this script if used!
# 0=False, 1=use given setup-file, 2=write current setup to given file, 3=print given setup file and quit
use_saved_setup = 0
setup_filename = 'test_setups/abalone_setup'

# Check for positive definite Cov matrices
pars['enforce_pos_def'] = 1
# 0 = only flag non-pos.def matrices
# 1 = ensure pos.def. Cov
# 1 in the paper

pars['random_seed'] = 1
# 1 in the paper

# Number of cross-validation runs for each fixed sample size
pars['n_repeats'] = 25
# 25 in the paper

# Number of repeats for finding optimal clipping threshold
pars['opt_clip_repeats'] = 20
# 20 in the paper

# Possible datasets: red_wine, white_wine, abalone; uncomment the selected data
pars['dataset_name'] = 'abalone'
#pars['dataset_name'] = 'white_wine'
#pars['dataset_name'] = 'red_wine'

# Number of observations(=clients) to be used
# Note: looped n_repeats times for each element in the list (CV), the elements are picked at random
pars['n_clients'] = [3000]
pars['n_test'] = [1000]
# Note: n_clients & n_test need to have same length; if n_test = 0, uses all the data left after splitting training set for testing
# number of clients in datasets:
#       red wine 1599
#       white wine 4898
#       abalone 4177
# In the paper the following sizes are used:
# red wine: n_clients=1000, n_test=500
# white wine: n_clients=3000, n_test=1000
# abalone: n_clients=3000, n_test=1000

# Use selected data dimensions; uncomment according to the data used
#pars['sel_dims'] = [0,1,2,3,4,5,6,7,8,9,10] #wines
pars['sel_dims'] = [0,1,2,3,4,5,6,7] #abalone
# Note: for UCI red wines max dim = 11
#               white whine = 11
#               abalone = 8

# Percentage of privacy used for estimating std. Used in both distributed and TA settings
pars['privacy_for_marg_var'] = .3
# .3 in the paper
  
#use clipping trick
#pars['do_clipping'] = True
#list of clipping thresholds, for each clipping is [-c,c]
  #Note: number of file ids need to match the number of clipping thresholds
  #empty list = use estimated optimal clipping
pars['all_clips'] = []
# empty list in the paper


# Scale data to specific range 
# Note: range is scaled to be of length (2*given value) with mean 0
# the distributions are NOT enforced to be symmetric around the mean though
pars['scale_to_range'] = 5
# 5 in the paper
# Assumed data and target ranges, each is interpreted as [-c,c]
pars['assumed_data_range'] = [7.5,7.5]
# [7.5,7.5] in the paper

# Folder for tmp files & output
pars['tmp_folder'] = 'tmp/'
pars['output_folder'] = 'test_results/'

# Add DP noise to suff. stats
# 0=no noise; 1=DP noise to suff stats; 2=noise addition by individuals, 3=both for comparison
pars['add_noise'] = 3
# 3 in the paper

# Privacy parameters
# Note: need to be equal length lists
pars['epsilon_tot'] = np.power(10,[0,.25,.5,.75,1,1.5])
pars['delta_tot'] = np.zeros(len(pars['epsilon_tot'])) + 10**(-4)

# File ids; each privacy par pair generates separate output files
# Note: needs to match the length of privacy par lists. 1.id is also used as a general label (e.g. for saving results & settings used; this needs to match the settings in combine_pred_errors.py for plotting)
pars['all_file_ids'] = ['_abalone_'+str(int(i)) for i in np.linspace(1,len(pars['epsilon_tot']),len(pars['epsilon_tot']))]

#comparison with T honest clients, T = ceil(scale*clients)
pars['scaling_comparison'] = 0
# Note: set to 0 to get standard input perturbation, 1=trusted aggregator
# 0 in the paper

# Comparisons to trusted aggregator DP
pars['compare_to_std_DP'] = True
# True=unclipped noise var is calculated as in standard DP, False=use n/(n-1) factor for the noise as in crypto (for checking)
# True in the paper

# Small constant to use if marg. std estimate <= 0
pars['small_const_for_std'] = .5
# .5 in the paper

# Extra options
pars['drugsens_data'] = False

pars['spark_filename'] = 'tmp/sparktest.csv'
# Note: this can be overwritten by command line options

################################################################################
# END OF SETUP
################################################################################

#check for needed folders
all_folders = [pars['output_folder'],pars['tmp_folder']]
m = re.split(r'/',setup_filename)
if m is not None and len(m) > 1:
  setup_folder = ''
  for k in range(len(m)-1):
    setup_folder += str(m[k]) + '/'
  all_folders.append(setup_folder)
for folder in all_folders:
  if not os.path.exists(folder):
    print('\nCreating folder ' + str(folder))
    os.makedirs(folder)

# Spark
pars['use_spark'] = False
pars['n_spark_messages'] = 10
pars['spark_noise_range'] = 10e13
pars['fixed_point_int'] = 10e6
# uses numpy randint  [-given val,given val]
# Note: this shouldn't be considered a cryptographically safe implementation
if len(sys.argv) > 1:
  try:
    opts, args = getopt.getopt(sys.argv[1:], "c:hs:f:n:", ["compute=","help", "spark=","fixed_point=","noise="])
  except getopt.GetoptError as err:
    print(str(err) + '. Use -h for help.')
    sys.exit(2)
  for o, a in opts:
    if o in ("-h", "--help"):
      print('Options:\n-s or --spark [filename] run a test using Spark. When using Spark, consider also setting the other options.\n-c or --compute [number of messages] sets the total number of messages used for Spark (default=10).\n-f or --fixed_point [fixed-point integer] defines the integer used for fixed-point arithmetic (default=10e6).\n-n or --noise sets the noise range used for Spark messages (default=10e14).')
      sys.exit()
    elif o in ("-s", "--spark"):
      pars['use_spark'] = True
      # Note: if use_spark = True, saves the individual contributions to the distributed non-projected model sufficient statistics to file on first round and terminates the run
      if a is not '':
        pars['spark_filename'] = a
      print('Running Spark test, saving to file \'{}\'.'.format(pars['spark_filename']))
      pars['n_repeats'] = 1
      pars['n_clients'] = [pars['n_clients'][0]]
      pars['epsilon_tot'] = [pars['epsilon_tot'][0]]
      pars['delta_tot'] = [pars['delta_tot'][0]]
    elif o in ["-c","--compute"]:
      if a is not '':
        pars['n_spark_messages'] = int(a)
        print('Using {} messages for each data point for Spark.'.format(a))
      else:
        print('Number of messages for Spark should be an int.')
    elif o in ['-f','--fixed_point']:
      pars['fixed_point_int'] = int(float(a))
    elif o in ['-n','--noise']:
      pars['spark_noise_range'] = int(float(a))
    else:
      assert False, "unhandled option"


pars['dim'] = len(pars['sel_dims'])

#check for optimal clipping
if len(pars['all_clips']) == 0:
  pars['do_optimal_clip'] = True
else:
  pars['do_optimal_clip'] = False


#setup-script use
#0=False, 1=use given setup-file, 2=write current setup to given file, 3=print given setup and quit
if use_saved_setup is 1:
  print('Reading setup from\n' + setup_filename + ', press y to continue..')
  apu = sys.stdin.read(1)
  if apu[0] is not 'y':
    print('Aborted')
    sys.exit()
  pars = setup_handler.get_setup(setup_filename)  
  
#write current setup to file
elif use_saved_setup is 2:
  print('Saving setup to\n' + setup_filename + ', press y to continue..')
  apu = sys.stdin.read(1)
  if apu[0] is not 'y':
    print('Aborted')
    sys.exit()
  setup_handler.write_setup(setup_filename, pars)
  print('setup written, exiting..')
  sys.exit()
#read & print the given pars
elif use_saved_setup is 3:
  print('Reading setup from\n' + setup_filename + '\n')
  apu = setup_handler.get_setup(setup_filename)
  for i in apu.items():
    print(str(i[0]) + ': ' + str(i[1]))
  sys.exit()

if not pars['do_optimal_clip']:
  clip_threshold = np.zeros((pars['dim'] + 1)) + pars['all_clips']

np.random.seed(pars['random_seed'])

print('Selected dims: {}'.format(pars['sel_dims']))
#get data
exec('data_master = UCI_data_getter.get_' + pars['dataset_name'] + '()')

#check that target is not selected as predictor
if data_master.shape[1]-1 in pars['sel_dims']:
  print('Target dim selected as predictor! Aborted.')
  sys.exit()

#drop unused dims
data_master = np.hstack((data_master[:,pars['sel_dims']],np.reshape(data_master[:,-1],(data_master.shape[0],1)) ))

#center data
data_master = data_master - np.mean(data_master, axis = 0)
  
#scale data to assumed range
data_master = np.multiply(data_master, 1/np.ptp(data_master,0)) * 2*pars['scale_to_range']
print('Data range lengths scaled to ' +str(2*pars['scale_to_range']))

#generate fixed train-test splits for each repeat that are used with all privacy pars and sample sizes
filename = pars['tmp_folder'] + 'permu_'
for k_file in range(pars['n_repeats']):
  if 0 in pars['n_test']:
    all_inds = np.random.permutation(data_master.shape[0])
  else:
    all_inds = np.random.choice( np.arange(data_master.shape[0]), np.amax(pars['n_clients'])+np.amax(pars['n_test']),False)
  with open(filename+str(k_file)+'.pickle', 'wb') as f:
    pickle.dump(all_inds, f, pickle.HIGHEST_PROTOCOL)


#loop over privacy pars
for k_privacy_par in range(len(pars['epsilon_tot'])):

  print('\nStarting iteration ' + str(k_privacy_par+1) +'/' + str(len(pars['epsilon_tot'])) + '...\n')
  sleep(.5)
  
  pars['epsilon'] = pars['epsilon_tot'][k_privacy_par]
  pars['delta'] = pars['delta_tot'][k_privacy_par]
  
  file_id = pars['all_file_ids'][k_privacy_par]
  pred_errors_filename = pars['output_folder'] + 'pred_errors_test' + file_id + '.pickle'
  
  pred_errors = list()
  
  client_round = -1
  
  for k_client in pars['n_clients']:
    print('\nNumber of clients: ' + str(k_client) + ' ('+str(client_round+2) +'/'+str(len(pars['n_clients']))+')')
    client_round = client_round + 1
    k_test = pars['n_test'][client_round]
    
    pred_errors_client_loop = list()
    
    if pars['do_optimal_clip']:
      clipping_array = np.zeros((pars['n_repeats'],pars['dim']+1))
    
    for k_repeat in range(pars['n_repeats']):
      print('\nStarting repeat ' + str(k_repeat + 1) + '/'+str(pars['n_repeats'])+'...\n')
      
      data = np.copy(data_master)
      #load fixed train-test split
      filename = pars['tmp_folder'] + 'permu_'
      permu = np.load(filename + str(k_repeat) + '.pickle')
      train_ind = permu[0:k_client]
      if k_test == 0: #use all elements not in training set
        test_ind = permu[k_client:]
      else:
        test_ind = permu[-k_test:]
      
      data_test = data[test_ind,:]
      data = data[train_ind,:]
      
################################################
# FIND OPTIMAL CLIPPING RATE
      
      if pars['do_optimal_clip']:
      
        print('Finding optimal clipping thresholds..\n')
        optimal_clip_values = np.zeros(2)
        
        optimal_clip_values[0], optimal_clip_values[1] = dp.omega(k_client, pars['dim'], pars['epsilon'], pars['delta'], 'mae', pars['opt_clip_repeats'])
        
        clip_threshold = np.zeros((pars['dim']+1))
        #estimate marginal std for each dimension & use for clipping
        stds = np.zeros(pars['dim']+1)
        pars['marginal_vars'] = estimate_vars.get_estimates(np.copy(data), pars=pars, small_pos = pars['small_const_for_std'])
        stds = np.sqrt( pars['marginal_vars'])
        
        stds_true = np.std(data,0)
        stds_TA_DP = dp.get_TA_std_estimates(np.copy(data),pars)
        
        #optimal clipping
        clip_threshold[0:-1] = stds[0:-1] * optimal_clip_values[0]
        clip_threshold[-1] = stds[-1] * optimal_clip_values[1]
        
        clip_threshold_true = np.zeros((pars['dim']+1))
        clip_threshold_TA_DP = np.zeros((pars['dim']+1))
        
        clip_threshold_true[0:-1] = stds_true[0:-1] * optimal_clip_values[0]
        clip_threshold_true[-1] = stds_true[-1] * optimal_clip_values[1]
        
        clip_threshold_TA_DP[0:-1] = stds_TA_DP[0:-1] * optimal_clip_values[0]
        clip_threshold_TA_DP[-1] = stds_TA_DP[-1] * optimal_clip_values[1]
        
        #check that clipping threshold is not greater than the assumed data range
        for k_dim in range(pars['dim']):
          clip_threshold[k_dim] = np.minimum(clip_threshold[k_dim],pars['assumed_data_range'][0])
          
        clip_threshold[-1] = np.minimum(clip_threshold[-1],pars['assumed_data_range'][-1])
        
        
################################################
#CLIPPING
      
      data_clipped = np.multiply( np.sign(data), np.minimum(clip_threshold,np.absolute(data) ) )
      
      data_clipped_true  = np.multiply( np.sign(data), np.minimum(clip_threshold_true,np.absolute(data) ) )
      data_clipped_TA_DP  = np.multiply( np.sign(data), np.minimum(clip_threshold_TA_DP,np.absolute(data) ) )
      
################################################
#CALCULATE (PERTURBED) SUFFICIENT STATS
      
      suff_stats, sigma_all, added_noise_dict = suff_stats_master.get_suff_stats(np.copy(data), np.copy(data_clipped), k_client, k_repeat, clip_threshold, pars, data_clipped_true, clip_threshold_true, data_clipped_TA_DP, clip_threshold_TA_DP)
      
      
################################################
#CHECK POSITIVE DEFINITENESS
      suff_stats = pos_def_matrices.check(suff_stats, pars)
      
################################################
#LINEAR REGRESSION
      model_coeffs = linear_regression_master.get_regression_est(suff_stats, pars)
      
################################################
#CALCULATE PREDICTION ERRORS
      
      pred_errors_client_loop = {}
      
      for k_model in model_coeffs:
        MAE, MSE, E_pred, std_pred, range_pred = calculate_pred_errors.calculate_errors(data=np.copy(data_test), dim=pars['dim'], filename_data='', model_coeff = model_coeffs[k_model])
        pred_errors_client_loop[k_model] = [MAE,MSE, E_pred, std_pred, range_pred]
      
      pred_errors.append(pred_errors_client_loop)
      
################################################
#end of loop over n_repeat
    
################################################
#end of loop over n_clients

  #pickle prediction errors
  with open(pred_errors_filename, 'wb') as f:
    pickle.dump(pred_errors, f, pickle.HIGHEST_PROTOCOL)

################################################
#end of loop over privacy pars

with open(pars['output_folder'] + 'pars_test' + pars['all_file_ids'][0] + '.pickle', 'wb') as f:
  pickle.dump(pars, f, pickle.HIGHEST_PROTOCOL)

print('\nAll done.')