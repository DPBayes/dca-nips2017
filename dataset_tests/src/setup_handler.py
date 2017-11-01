'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data

Setup script handler: pickle setup parameters & read them back.
'''

import numpy as np
import pickle

def get_setup(saved_setup):
  with open(saved_setup + '.pickle', 'rb') as f:
    apu = pickle.load(f)
  return apu

def write_setup(saved_setup, pars):
  with open(saved_setup + '.pickle', 'wb') as f:
    pickle.dump(pars, f, pickle.HIGHEST_PROTOCOL)
