# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import numpy.dual as npdual

class Recording:
  """A (labeled or unlabeled) data set from the electrodes of the recording device."""

  def __init__(self, nr_e, nr_s):
    self.__data = np.zeros((nr_e, nr_s))
    self.__is_labeled = False
    self.__labels = np.zeros(nr_s, int)
  
  def get_data(self, i, t):
    if i < 0 or i > self.get_number_of_electrodes()-1:
      print "DEBUG: error in Recording::get_data: invalid electrode index ({x})!".format(x=i)
      raise RuntimeError("invalid electrode index")
    if t < 0 or t > self.get_number_of_samples()-1:
      print "DEBUG: error in Recording::get_data: invalid sample index ({x})!".format(x=t)
      raise RuntimeError("invalid sample index")
    return self.__data[i][t]
  
  def get_whole_data(self, i):
    if i < 0 or i > self.get_number_of_electrodes()-1:
      print "DEBUG: error in Recording::get_data: invalid electrode index ({x})!".format(x=i)
      raise RuntimeError("invalid electrode index")
    return self.__data[i]
  
  def set_data(self, i, t, value):
    if i < 0 or i > self.get_number_of_electrodes()-1:
      # print "DEBUG: error in Recording::get_data: invalid electrode index ({x})!".format(x=i)
      raise RuntimeError("invalid electrode index")
    if t < 0 or t > self.get_number_of_samples()-1:
      # print "DEBUG: error in Recording::get_data: invalid sample index ({x})!".format(x=t)
      raise RuntimeError("invalid sample index")
    self.__data[i][t] = value
  
  def get_labels(self):
    return self.__labels
  
  def set_label(self, t, value):
    if t < 0 or t > self.get_number_of_samples()-1:
      # print "DEBUG: error in Recording::get_data: invalid sample index ({x})!".format(x=t)
      raise RuntimeError("invalid sample index")
    self.__is_labeled = True
    self.__labels[t] = value
  
  def is_labeled(self):
    return self.__is_labeled
  
  def plot(self):
    times = range(len(self.__labels))
    plt.subplot(211)
    plt.ylabel('signal e1')
    plt.plot(times, self.__data[0])
    
    plt.subplot(212)
    plt.xlabel('sample #')
    plt.ylabel('label')
    plt.plot(times, self.__labels, 'g')
    plt.show()
  
  def get_number_of_electrodes(self):
    return len(self.__data)

  def get_number_of_samples(self):
    return len(self.__data[0])
  
  def get_label_of_sample(self, index):
    return self.__labels[index]
  
  def get_power_spectrum_of_sample_range(self, electrode_index, start_index, end_index):
    if electrode_index < 0 or electrode_index > self.get_number_of_electrodes()-1:
      raise RuntimeError("invalid electrode index")
    if start_index < 0 or start_index > self.get_number_of_samples()-1:
      raise RuntimeError("invalid start sample index")
    if end_index > self.get_number_of_samples()-1:
      raise RuntimeError("invalid end sample index")
    if start_index > end_index:
      raise RuntimeError("end sample index must be greater than start sample index")
    
    # return power spectrum of cutout
    return np.abs( npdual.fft( self.__data[electrode_index, start_index:end_index+1] ) )**2
  
