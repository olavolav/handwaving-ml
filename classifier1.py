# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import time as time
from Recording import *
from BayesClassifier import *

OUTCOME_LABELS = {"null", "test1"}
USE_SIMULATED_SIGNALS = True
NUMBER_OF_ELECTRODES = 1
SAMPLING_RATE_FOR_PLOTTING = 2500.0 # in Hz

print "--- HMM classifier, OS, April 2013 ---"

# ------ loading/generating recordings ------
if(USE_SIMULATED_SIGNALS):
  print "simulating recordings..."
  from RecordingSimulator import *
  sim = RecordingSimulator(NUMBER_OF_ELECTRODES, OUTCOME_LABELS)
  recordings = sim.generate_data(3)
  # recordings[0].plot()
  # recordings[-1].plot()
else:
  print "loading recordings from file..."
print "-> number of recordings: {n}".format(n=len(recordings))

# ------ training classifier ------
print "training classifier..."
analyzer = BayesClassifier(NUMBER_OF_ELECTRODES, OUTCOME_LABELS)
analyzer.learn_models(recordings)
print "-> training completed, success = {d}".format(d=analyzer.has_been_trained())

# ------ applying classifier ------
print "applying classifier..."
target_recording = recordings[-1]
prediction_result = analyzer.classify( target_recording )
times = np.arange(target_recording.get_number_of_samples())/SAMPLING_RATE_FOR_PLOTTING

plt.subplot(311)
plt.ylabel('signal -1')
plt.plot(times, target_recording.get_whole_data(0))

# plt.subplot(312)
# plt.ylabel('label')
# plt.plot(times, target_recording.get_labels(), 'g')
# plt.ylim(-0.5, max(target_recording.get_labels())+0.5)

plt.subplot(312)
plt.xlabel('sample #')
plt.ylabel('l_logs')
plt.plot(times, prediction_result[0], 'grey', times, prediction_result[1], 'red')

plt.subplot(313)
final_prediction = [prediction_result[1,s] > prediction_result[0,s] for s in range(len(prediction_result[0]))]
plt.xlabel('time (s)')
# plt.ylabel('prediction')
# plt.plot(times, final_prediction, 'r')
plt.ylabel('l. vs. pred.')
plt.plot(times, target_recording.get_labels(), 'g', times, final_prediction, 'r')
plt.ylim(-0.5, max(final_prediction)+0.5)
plt.show()