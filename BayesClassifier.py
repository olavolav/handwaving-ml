# -*- coding: utf-8 -*-

import math as math
import numpy as np
from GaussModel import *
from Recording import *

TRAINING_VECTOR_LENGTH = 250 # number of samples to use for training
PRIOR_PROBABILIY_OF_SWITCHING = 10.0/7500.0 # DH's real data
# PRIOR_PROBABILIY_OF_SWITCHING = 10.0/1200.0 # simulated data
FORCE_FLAT_PRIOR = False

class BayesClassifier:
  """A Bayesian classifier based on a hidden Markov model analysis."""
  
  def __init__(self, nr_e, labels):
    self.__number_of_outcomes = len(labels)
    self.__number_of_electrodes = nr_e
    self.__outcome_labels = labels
    self.__models = [GaussModelCombinedDomains(self.__number_of_electrodes, TRAINING_VECTOR_LENGTH, l) for l in range(self.__number_of_outcomes)]
  
  def learn_models(self, recordings):
    for m in self.__models:
      m.learn_based_on_recordings(recordings)
      print "-> completed learning of model #{l}.".format(l=m.get_label_nr())
  
  def has_been_trained(self):
    return [m.model_has_been_learned() for m in self.__models] == [True for _ in range(len(self.__models))]
  
  def classify(self, recording):
    assert self.has_been_trained()
    prob = np.zeros((self.__number_of_outcomes, recording.get_number_of_samples()), float)
    # start with flat prior
    prior_probabilities = np.ones(self.__number_of_outcomes, float) / float(self.__number_of_outcomes)
    
    for s in range(TRAINING_VECTOR_LENGTH, recording.get_number_of_samples()):
      log_likelihoods = [m.compute_nonnorm_log_likelihood(recording, range(self.__number_of_electrodes), s, prior_probabilities[m.get_label_nr()]) for m in self.__models]
      probabilities = [math.exp(l-max(log_likelihoods)) for l in log_likelihoods]
      probabilities = [p/sum(probabilities) for p in probabilities]
      for o in range(self.__number_of_outcomes):
        prob[o, s] = probabilities[o]
      if(not(FORCE_FLAT_PRIOR)):
        winner = probabilities.index(max(probabilities))
        for oo in range(self.__number_of_outcomes):
          if(oo == winner):
            prior_probabilities[oo] = 1.0 - PRIOR_PROBABILIY_OF_SWITCHING
          else:
            prior_probabilities[oo] = PRIOR_PROBABILIY_OF_SWITCHING/(self.__number_of_outcomes-1)
    return prob
