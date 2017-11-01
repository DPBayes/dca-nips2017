'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data

Script for combining prediction error results from individual pickled files produced by eps_data_test.py.

eps_data_test.py should be run before this.

Run: python3 combine_pred_errors.py
'''

import sys

import numpy as np
from matplotlib import pyplot as plt

################################################################################
# SETUP
################################################################################

# Plot settings
pars_filename = 'test_results/NIPS_camera_ready/pars_test_red_wine_1.pickle'
#pars_filename = 'test_results/NIPS_camera_ready/pars_test_white_wine_1.pickle'
#pars_filename = 'test_results/NIPS_camera_ready/pars_test_abalone_1.pickle'
# Note: set this to match the settings in eps_data_test.py

# for reproducing the figures in the paper
figure_bounds = 'red_wine'
#figure_bounds = 'white_wine'
#figure_bounds = 'abalone'

#save figure
save_to_file = False
fig_name = 'plots/UCI_redwine_NIPS_final.pdf'
#fig_name = 'plots/UCI_whitewine_NIPS_final.pdf'
#fig_name = 'plots/UCI_abalone_NIPS_final.pdf'

#PLOTTING CONFIGURATIONS & COLORS
no_plotting = ['cl_scaling', 'cl_noisy','cl_true_TA']

nimet_dict = {'true':'NP', 'clipped':'proj NP','noisy':'TA', 'cl_noisy':'proj TA', 'noisy_ind':'DDP', 'cl_noisy_ind':'proj DDP', 'scaling':'input\nperturbed','cl_scaling':'proj scaling','cl_true_TA':'proj TA (non DP)' ,'cl_true_TA_DP': 'proj TA'}

# colors
col_dict = {'true':'blue', 'clipped':'gray','noisy':'lightseagreen', 'cl_noisy':'green', 'noisy_ind':'red', 'cl_noisy_ind':'magenta', 'scaling':'orange','cl_scaling':'orange', 'cl_true_TA': 'black','cl_true_TA_DP':'green'}

################################################################################
# END OF SETUP
################################################################################
metodit = ['true', 'clipped', 'noisy', 'cl_noisy', 'noisy_ind', 'cl_noisy_ind', 'scaling','cl_scaling', 'cl_true_TA','cl_true_TA_DP']

#load parameters used
pars = np.load(pars_filename)
print('Parameters read from ' + str(pars_filename))

#lists with one element for each clipping rate
abs_error_list = list()
sq_error_list = list()

#create names list
nimet = []
for m in metodit:
  if m not in no_plotting:
    nimet.append(nimet_dict[m])

for k_test in pars['all_file_ids']:
  abs_err = {}
  sq_err = {}
  for m in metodit:
    abs_err[m] = np.zeros((len(pars['n_clients']), pars['n_repeats']))
    sq_err[m] = np.zeros((len(pars['n_clients']), pars['n_repeats']))
  
  filename = pars['output_folder'] + 'pred_errors_test' + str(k_test) + '.pickle'
  apu = np.load(filename)
  i = 0
  for k_clients in range(len(pars['n_clients'])):
    for k_repeat in range(pars['n_repeats']):
      for m in metodit:
        #MAE
        abs_err[m][k_clients,k_repeat] = apu[i][m][0]
        #MSE
        sq_err[m][k_clients,k_repeat] = apu[i][m][1]
      i = i+1
       
  abs_error_list.append(abs_err)
  sq_error_list.append(sq_err)

###############################################################################
#simple plotting function
def plotter(x,y,metodit, bounds, x_label, y_label, subtitle, x_ticks, add_noise_mean, y_err_lower=None, y_err_upper=None, y_all_clip_means=None,y_true_clip_means=None):
  round = -3
  k_col = 0
  for m in metodit:
    k_col = k_col + 1
    if m not in no_plotting: #skip non-used
      #plot non-private with dashed line
      if m in ['true','clipped']:
        linetype = '--'
      else:
        linetype = '-'
        
      if y_err_lower == None:
      #line, = plt.plot(x,y[m], '*-', linewidth=2,label=m)
        plt.plot(x,y[m], '*-', linewidth=2.5,label=m,linestyle=linetype)
      else:
      #with errorbars
        plt.errorbar(x+round*.05, y[m],linewidth=2.2, yerr=[y_err_lower[m],y_err_upper[m] ], linestyle=linetype, color=col_dict[m],label=m )
      round = round + 1
        
  #add clipping thresholds if applicable
  if y_all_clip_means != None:
    plt.plot(x,y_all_clip_means,label=m )
  if y_true_clip_means != None:
    plt.plot(x,y_true_clip_means,label=m )
        
  #add line for unclipped noise mean
  if add_noise_mean:
    plt.plot(x,np.repeat(np.mean(y['noisy']),len(x)), '--', linewidth=1,label='noise mean' )
    
  #define custom bounds for result figures
  if figure_bounds == 'abalone':
    bounds[2:] = [.55,2.5]
  elif figure_bounds == 'red_wine':
    bounds[2:] = [.59,4.0]
  elif figure_bounds == 'white_wine':
    bounds[2:] = [.63,2.5]
  
  plt.axis(bounds)
  plt.tight_layout(pad=7)
  plt.legend(nimet,bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.suptitle(subtitle, y=.12, fontsize=13)
  plt.xticks(x_ticks[0],x_ticks[1])
  if save_to_file:
    plt.savefig(fig_name, bbox_inches='tight')
  else:
    plt.show()


###############################################################################
for sample_size in range(len(pars['n_clients'])):
  x = np.linspace(1,len(sq_error_list),num=len(sq_error_list))
  y_mae = {}
  y_mse = {}
  y_mae_lower = {}
  y_mae_upper = {}
  y_mae_err = {}
  for m in metodit:
    y_mae[m] = np.zeros(len(sq_error_list))
    y_mse[m] = np.zeros(len(sq_error_list))
    y_mae_err[m] = np.zeros(len(sq_error_list))
    y_mae_lower[m] = np.zeros(len(sq_error_list))
    y_mae_upper[m] = np.zeros(len(sq_error_list))
    for k_priv in range( len(pars['epsilon_tot']) ):
      y_mae[m][k_priv] = np.median(abs_error_list[k_priv][m][sample_size, :] )
      #calculate .25 and .75 quantiles for errorbars
      apu = np.sort(abs_error_list[k_priv][m][sample_size, :])
      y_mae_lower[m][k_priv] = np.absolute( apu[ int(np.floor(.25*len(apu))) ] - y_mae[m][k_priv] )
      y_mae_upper[m][k_priv] = np.absolute( apu[ int(np.ceil(.75*len(apu))) ] - y_mae[m][k_priv] )
      
      y_mse[m][k_priv] = np.mean(sq_error_list[k_priv][m][sample_size, :] )
  y_mae_err = None#obsolete

for sample_size in range(len(pars['n_clients'])):
  if len(x) < 10:
    x_ticks = [x, np.round(pars['epsilon_tot'],2)]
  else:
    x_ticks = [x[0::3], np.round(pars['epsilon_tot'][0::3], 2)]
  #mae
  if not pars['do_optimal_clip']:
    plotter(x, y_mae, metodit, [0,len(x)+1,0,1], 'epsilon', 'MAE', 'clipping: '+str(pars['all_clips']) + ', sample size=' + str(pars['n_clients'][sample_size]) + ', delta=' + str(pars['delta_tot'][0]), x_ticks, False,  y_mae_lower,y_mae_upper)
  else:
    plotter(x, y_mae, metodit, [0,len(x)+1,0,1], 'epsilon', 'MAE', 'd=' + str(pars['dim']) + ', sample size=' + str(pars['n_clients'][sample_size]) + ', repeats=' + str(pars['n_repeats']) + ', $\delta=$' + str(pars['delta_tot'][0]), x_ticks, False, y_mae_lower,y_mae_upper)


