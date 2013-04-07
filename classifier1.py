# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import time as time
from Recording import *
from BayesClassifier import *

OUTCOME_LABELS = {"null", "test1", "test2"}
USE_SIMULATED_SIGNALS = True
NUMBER_OF_ELECTRODES = 1
SAMPLING_RATE_FOR_PLOTTING = 2500.0 # in Hz

print "--- HMM classifier, OS, April 2013 ---"

# ------ loading/generating recordings ------
if(USE_SIMULATED_SIGNALS):
  print "simulating recordings..."
  from RecordingSimulator import *
  sim = RecordingSimulator(NUMBER_OF_ELECTRODES, OUTCOME_LABELS)
  recordings = sim.generate_data(1)
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
target_recording = recordings[0]
ideal_signal = target_recording.get_labels()
prediction_result = analyzer.classify( target_recording )
times = np.arange(target_recording.get_number_of_samples())/SAMPLING_RATE_FOR_PLOTTING

# ------ compute overall accuracy ------
final_prediction = [np.argmax(prediction_result[:,s]) for s in range(len(prediction_result[0]))]
hits = 0
for t in range(len(prediction_result[0])):
  if final_prediction[t] == ideal_signal[t]:
    hits += 1
print "-> final accuracy: {x}%".format(x=(100.0*hits)/len(prediction_result[0]))

# ------ pot result ------
plt.subplot(311)
plt.ylabel('signal -1')
plt.plot(times, target_recording.get_whole_data(0))

plt.subplot(312)
plt.xlabel('sample #')
plt.ylabel('l_logs')
plt.plot(times, prediction_result[0], 'grey', times, prediction_result[1], 'red', times, prediction_result[2], 'green')

plt.subplot(313)
plt.xlabel('time (s)')
plt.ylabel('l. vs. pred.')
plt.plot(times, ideal_signal, 'g', times, final_prediction, 'r')
plt.ylim(-0.5, max(final_prediction)+0.5)
plt.show()
