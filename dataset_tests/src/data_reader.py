'''
Differentially private Bayesian learning on distributed data
Mikko Heikkilä 2016-17

UCI data

Function for reading data from a given file and returning it as a list.
'''

import numpy as np
import csv

def read_data(filename):
  with open(filename,newline='',encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    data = list()
    for row in reader:
      data.append(row)
  return data
