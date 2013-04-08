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
    
    samples = len(data[0])
