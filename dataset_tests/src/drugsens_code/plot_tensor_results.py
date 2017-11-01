'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

Modified from the original code:
  Differentially private Bayesian linear regression 
  Arttu Nieminen 2016-2017
  University of Helsinki Department of Computer Science
  Helsinki Institute of Information Technology HIIT

GDSC/drug sensitivity data

Script for plotting tensor test results from pickle files.

tensorresults.py should be run before this.

Run: python3 plot_tensor_results.py
'''

import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt


# PLOTTING CONFIGURATIONS
# set these to match tensorresults.py

# Result filename
filename = 'resultsdata/drugsens_test.pickle'
#filename = 'resultsdata/NIPS_camera_ready/data_bounds/drugsens_test.pickle'
#filename = 'resultsdata/NIPS_camera_ready/fixed_bounds/drugsens_test.pickle'

# Save figure to file
save_to_file = False
fig_name = 'NIPS_camera_ready_plots/GDSC_drugsens_NIPS_final_all.pdf'
#fig_name = 'NIPS_camera_ready_plots/GDSC_drugsens_NIPS_final_selected.pdf'

# Methods to plot
# plot all methods
no_plotting = ['cl_scaling','cl_noisy', 'cl_true_TA']
# plot selected methods
#no_plotting = ['clipped','cl_scaling','noisy','cl_noisy', 'cl_true_TA','cl_true_TA_DP']

###############################################################################
metodit = ['true', 'clipped', 'noisy', 'cl_noisy', 'noisy_ind', 'cl_noisy_ind', 'scaling','cl_scaling','cl_true_TA','cl_true_TA_DP']

nimet_dict = {'true':'NP', 'clipped':'proj NP','noisy':'TA', 'cl_noisy':'proj TA (noise)', 'noisy_ind':'DDP', 'cl_noisy_ind':'proj DDP', 'scaling':'input\nperturbed','cl_scaling':'proj scaling','cl_true_TA':'proj TA (not DP)', 'cl_true_TA_DP':'proj TA' }

#use same colors for corresponding non-clipped & clipped methods except for DDP
col_dict = {'true':'blue', 'clipped':'gray','noisy':'lightseagreen', 'cl_noisy':'green', 'noisy_ind':'red', 'cl_noisy_ind':'magenta', 'scaling':'orange','cl_scaling':'orange','cl_true_TA':'brown', 'cl_true_TA_DP':'darkgreen'}

with open(filename, 'rb') as f:
  res_all = pickle.load(f)

# parameters
eps = [1.0,3.0,5.0,7.5,10.0]
n_train = 840
# Note: some drugs might not have the same number of training data
n_test = 100

x = np.linspace(1,len(eps),num=len(eps))
y_err_lower = None
y_lower = {}
y_upper = {}
k_col = 0
plt.figure()
ax = plt.gca()
offset = -3
offset_factor = 0.05
for m in metodit:
  k_col = k_col + 1 #?
  if m not in no_plotting:
    y_lower[m] = np.zeros(len(eps))
    y_upper[m] = np.zeros(len(eps))
    
    #plot non-private with dashed line
    if m in ['true','clipped']:
      linetype = '--'
    else:
      linetype = '-'
      
    ax.errorbar(x+offset*offset_factor,np.mean(res_all[m]['mean'],0),
 yerr=np.std(res_all[m]['mean'],0),ls=linetype,marker='',linewidth=2,label=m ,color=col_dict[m], elinewidth=1.5)
    plt.axis([.79,5.21,-.03,.27])
    
    offset += 1

plt.tight_layout(pad=7)
nimet = []
for m in metodit:
  if m not in no_plotting:
    nimet.append(nimet_dict[m])
    
plt.xticks(x,eps)
plt.ylabel('Predictive accuracy')
plt.xlabel('epsilon')
plt.suptitle('d=10, sample size=840, CV=25, $\delta$=0.0001', y=.12, fontsize=13)

#legend on top
plt.legend(nimet,bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

if save_to_file:
  plt.savefig(fig_name, bbox_inches='tight')
else:
  plt.show()
