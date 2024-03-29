# -*- coding: utf-8 -*-

import math as math
import numpy as np
import sets as sets

PI = 3.141
MAX_SAMPLES_PER_INFERRED_VARIABLE = 37500 # meaning the first 5 iterations in the DH data set

class GaussModel:
  """Model of signal (likelihood function)."""
  
  def __init__(self, nr_e, v_length, l_nr):
    self._label_nr = l_nr
    self._number_of_electrodes = nr_e
    self._length_in_time = v_length
    self._mean_vector = np.zeros((self._number_of_electrodes, self._length_in_time), float)
    self._stddev_vector = np.zeros_like(self._mean_vector)
    self._model_has_been_learned = False
  
  def learn_based_on_recordings(self, recs):
    self._model_has_been_learned = True
    relevant_sample_points = np.zeros(MAX_SAMPLES_PER_INFERRED_VARIABLE, float)
    for e in range(self._number_of_electrodes):
      # print "DEBUG: e = {x}".format(x=e)
      for t_lag in range(self._length_in_time):
        number_of_gathered_points = 0
        for rec in recs:
          for s in range(t_lag, rec.get_number_of_samples()):
            if(rec.get_label_of_sample(s) == self._label_nr):
              # so, if this is a sample point with the label of this model
              if number_of_gathered_points >= MAX_SAMPLES_PER_INFERRED_VARIABLE: break
              relevant_sample_points[number_of_gathered_points] = rec.get_data(e, s-t_lag)
              number_of_gathered_points += 1
        if len(sets.Set(relevant_sample_points[:number_of_gathered_points])) > 1:
          self._stddev_vector[e, t_lag] = np.std(relevant_sample_points[:number_of_gathered_points])
        else:
          print "Warning: model #{x} could not be learned (n = {n}).".format(x=self._label_nr,n=number_of_gathered_points)
          self._model_has_been_learned = False
  
  def compute_nonnorm_log_likelihood(self, recording, electrode_list, sample_index, prior_prob=1.0):
    assert self._model_has_been_learned
    assert sample_index >= self._length_in_time
    log_l = math.log(prior_prob)
    for i_electr in electrode_list:
      for t_lag in range(self._length_in_time):
        data = recording.get_data(i_electr, sample_index-self._length_in_time+t_lag)
        mu = self._mean_vector[i_electr, t_lag]
        stddev = self._stddev_vector[i_electr, t_lag]
        log_l += -1.0*math.log(stddev * math.sqrt(2.0*PI)) - 0.5*math.pow((mu-data)/stddev, 2.0)
    return log_l
  
  def model_has_been_learned(self):
    return self._model_has_been_learned
  
  def get_label_nr(self):
    return self._label_nr
  

class GaussModelFrequencyDomain(GaussModel):
  """Model of signal based on the power spectrum (likelihood function)."""
  
  def learn_based_on_recordings(self, recs):
    self._model_has_been_learned = True
    relevant_sample_points = np.zeros(MAX_SAMPLES_PER_INFERRED_VARIABLE, float)
    for e in range(self._number_of_electrodes):
      # print "DEBUG: e = {x}".format(x=e)
      for freq_component in range(self._length_in_time):
        number_of_gathered_points = 0
        # relevant_sample_points = np.zeros(0, float)
        for rec in recs:
          for s in range(self._length_in_time, rec.get_number_of_samples()):
            if(rec.get_label_of_sample(s) == self._label_nr):
              # so, if this is a sample point with the label of this model at sample #s
              if number_of_gathered_points >= MAX_SAMPLES_PER_INFERRED_VARIABLE: break
              spectrum = rec.get_power_spectrum_of_sample_range(e, s-self._length_in_time + 1, s)
              relevant_sample_points[number_of_gathered_points] = spectrum[-(freq_component+1)]
              number_of_gathered_points += 1
        if len(sets.Set(relevant_sample_points)) > 1:
          self._mean_vector[e, freq_component] = np.mean(relevant_sample_points)
          self._stddev_vector[e, freq_component] = np.std(relevant_sample_points)
        else:
          self._model_has_been_learned = False
  

class GaussModelCombinedDomains:
  
  def __init__(self, nr_e, v_length, l_nr):
    self._label_nr = l_nr
    self._number_of_electrodes = nr_e
    self._length_in_time = v_length
    self.time_domain_model = GaussModel(nr_e, v_length, l_nr)
    self.freq_domain_model = GaussModelFrequencyDomain(nr_e, v_length, l_nr)
  
  def learn_based_on_recordings(self, recs):
    self.time_domain_model.learn_based_on_recordings(recs)
    self.freq_domain_model.learn_based_on_recordings(recs)
  
  def compute_nonnorm_log_likelihood(self, recording, electrode_list, sample_index, prior_prob=1.0):
    return self.time_domain_model.compute_nonnorm_log_likelihood(recording, electrode_list, sample_index, prior_prob) + self.freq_domain_model.compute_nonnorm_log_likelihood(recording, electrode_list, sample_index, prior_prob)
  
  def model_has_been_learned(self):
    return ( self.time_domain_model.model_has_been_learned() and  self.freq_domain_model.model_has_been_learned() )
  
  def get_label_nr(self):
    return self._label_nr
  

