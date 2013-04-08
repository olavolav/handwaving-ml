# -*- coding: utf-8 -*-

import math as math
import numpy as np
import random

from Recording import *

class RecordingSimulator:
  """A simulator."""
  
  def __init__(self, nr_e=1, labels={"null"}):
    self.outcome_labels = labels
    self.number_of_electrodes = nr_e
  
  def generate_data(self, nr_per_label=1):
    random.seed(1234)
    np.random.seed(1234)
    recordings = []
    for l in range(1, len(self.outcome_labels)): # note that here we skip the null outcome (HACK)
      for trial in range(nr_per_label):
        recordings.append( self.__generate_single_trial(l) )
    return recordings
  
  def __generate_single_trial(self, label_nr, samples=500):
    t0 = round(0.5*samples)
    rec = Recording(self.number_of_electrodes, samples)
    for i in range(self.number_of_electrodes):
      for t in range(samples):
        fraction_of_signal = 1.0/(1+ math.exp(-(t-t0)))
        rec.set_data(i, t, (1.0-fraction_of_signal)*np.random.normal(0.0, 0.1) + fraction_of_signal*(np.random.normal(0.0, 0.2*(label_nr+1)) + 0.1*math.cos(t*label_nr)))
        rec.set_label(t, label_nr if t >= t0 else 0)
    return rec
  
