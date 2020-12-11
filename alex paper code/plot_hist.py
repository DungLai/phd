import matplotlib.pyplot as plt
import numpy
from scipy import stats
from scipy.stats import norm
import pandas as pd

# Read in data and examine first 10 rows
data = pd.read_csv('delta-changes.csv')

data_emphasis_change = [] 
data_generalisation = []
data_specialisation = []

for i in range(len(data)):
	if data['type_of_label_change'][i] == "Emphasis Change":
		data_emphasis_change.append(data['delta_conf'][i])
	if data['type_of_label_change'][i] == "Specialisation":
		data_specialisation.append(data['delta_conf'][i])
	if data['type_of_label_change'][i] == "Generalisation":
		data_generalisation.append(data['delta_conf'][i])

# matplotlib histogram
plt.subplot(3, 1, 1)
plt.xlim(left = -0.35)
plt.xlim(right = 0.45)
plt.hist(data_emphasis_change, color = 'red', label="Emphasis Change", edgecolor = 'black', bins = int(180/2))
plt.legend(loc="upper right")
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
plt.axvline(numpy.array(data_emphasis_change).mean(), color='k', linestyle='dashed', linewidth=1)

plt.subplot(3, 1, 2)
plt.xlim(left = -0.35)
plt.xlim(right = 0.45)
plt.hist(data_generalisation, color = 'green', label="Generalisation", edgecolor = 'black', bins = int(180/2))
plt.legend(loc="upper right")
frame2 = plt.gca()
frame2.axes.xaxis.set_ticklabels([])
plt.ylabel('Frequency')
plt.axvline(numpy.array(data_generalisation).mean(), color='k', linestyle='dashed', linewidth=1)

plt.subplot(3, 1, 3)
plt.xlim(left = -0.35)
plt.xlim(right = 0.45)
plt.hist(data_specialisation, color = 'blue', label="Specialisation", edgecolor = 'black', bins = int(180/2))
plt.legend(loc="upper right")
plt.axvline(numpy.array(data_specialisation).mean(), color='k', linestyle='dashed', linewidth=1)
plt.xlabel('Confidence ±Delta')

plt.show()