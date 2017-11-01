'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data

Script for reading UCI datasets. Returns a dataset with target as the last col.
'''

import numpy as np

import data_reader

data_folder = 'data/'

#abalone dataset, predict abalone age
def get_abalone():
  filename = data_folder + 'abalone/abalone.data'
  apu = data_reader.read_data(filename)
  data = np.zeros( (len(apu),len(apu[0])))
  for k_row in range(data.shape[0]):
    data[k_row,1:] = apu[k_row][1:]
    #code categorical sex as 0=male, 1=female
    if apu[k_row][0] == 'M':
      data[k_row,0] = 0
    else:
      data[k_row,0] = 1
  return data

#predict concrete compressive strength
def get_concrete():
  #8 mittausta ja target
  filename = data_folder + 'concrete/Concrete_Data.txt'
  apu = data_reader.read_data(filename)
  data = np.zeros( (len(apu)-1,len(apu[0])))
  #0s rivi=nimet
  for k_row in range(1,data.shape[0]):
    data[k_row,:] = apu[k_row]
  return data

#predict wine quality, red & white separately
def get_red_wine():
  filename = data_folder + 'wine/winequality-red.csv'
  apu = data_reader.read_data(filename)
  data = np.zeros((len(apu)-1,len(apu[0][0].split(";") )))
  for k_row in range(1, data.shape[0]):
    data[k_row-1,:] = apu[k_row][0].split(";")
    #0th row=names
  return data
def get_white_wine():
  filename = data_folder + 'wine/winequality-white.csv'
  apu = data_reader.read_data(filename)
  data = np.zeros((len(apu)-1,len(apu[0][0].split(";") )))
  for k_row in range(1, data.shape[0]):
    data[k_row-1,:] = apu[k_row][0].split(";")
    #0th row=names
  return data

if __name__=='__main__':
  get_white_wine()