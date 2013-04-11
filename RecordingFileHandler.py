# -*- coding: utf-8 -*-

import numpy as np

from Recording import *

class RecordingFileHandler:
  """The import interface to load recordings from disk."""
  
  def __init__(self, f_name):
    self.filename = f_name
  
  def load_recordings(self):
    try:
      data = np.loadtxt( self.filename )
    except IOError as e:
      print "I/O error({0}): {1}".format(e.errno, e.strerror)
    
    samples = data.size/3 # a single recording
    # samples = round(samples*0.03) # SPEEDUP HACK
    nr_electrodes = data[0].size - 1 # first column is labels
    
    rec = Recording(nr_electrodes, samples)
    sample_index = 0
    for t_vec in data[:samples]:
      rec.set_label(sample_index, round(t_vec[0]))
      for e in range(nr_electrodes):
        rec.set_data(e, sample_index, t_vec[1+e])
      sample_index += 1
    
    return [rec]
  
