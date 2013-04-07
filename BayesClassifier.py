# -*- coding: utf-8 -*-

import math as math
import numpy as np
from GaussModel import *
from Recording import *

TRAINING_VECTOR_LENGTH = 20 # number of samples to use for training

class BayesClassifier:
  """A Bayesian classifier based on a hidden Markov model analysis."""
  
  def __init__(self, nr_e, labels):
    self.__number_of_outcomes = len(labels)
    self.__number_of_electrodes = nr_e
    self.__outcome_labels = labels
    self.__models = [GaussModel(self.__number_of_electrodes, TRAINING_VECTOR_LENGTH, l) for l in range(self.__number_of_outcomes)]
  
  def learn_models(self, recordings):
    for m in self.__models:
      m.learn_based_on_recordings(recordings)
  
  def compute_log_likelihoods(self, rec, e_list):
    pass
  
  def has_been_trained(self):
    return [m.model_has_been_learned() for m in self.__models] == [True for _ in range(len(self.__models))]
  
  def classify(self, recording):
    assert self.has_been_trained()
    prob = np.zeros((self.__number_of_outcomes, recording.get_number_of_samples()), float)
    for s in range(TRAINING_VECTOR_LENGTH, recording.get_number_of_samples()):
      log_likelihoods = [m.compute_nonnorm_log_likelihood(recording, range(self.__number_of_electrodes), s) for m in self.__models]
      for o in range(self.__number_of_outcomes):
        # first we skip exponentiation, TODO
        prob[o, s] = log_likelihoods[o] - max(log_likelihoods)
    return prob
