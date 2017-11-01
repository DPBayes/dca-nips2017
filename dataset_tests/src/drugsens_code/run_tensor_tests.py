'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

GDSC/drug sensitivity data

Script for running tensor.py for a collection of drugs and CVs.

clippingomega.py should be run before this.

Run: python3 run_tensor_tests.py
'''

import subprocess
import sys

import numpy as np

n_drugs = 264 # 264 in the paper
n_cv = 25 # 25 in the paper
drugs_to_run = np.linspace(0,n_drugs,n_drugs+1,dtype='int')
seeds_for_cv = np.linspace(0,n_cv,n_cv+1,dtype='int')

#args to tensor.py: drug_id, seed
for drug in drugs_to_run:
  print('Starting drug ' + str(drug))
  for seed in seeds_for_cv:
    testi = subprocess.run(args=['python','tensor.py',str(drug),str(seed)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('stdout:\n' + testi.stdout.decode('utf-8'))
    print('stderr:\n' + testi.stderr.decode('utf-8'))

print('All tensor tests done!')