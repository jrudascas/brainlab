import numpy as np
import csv
import matplotlib.pyplot as plt
import math

path_mritoolbox_predictors = '/home/brainlab/test.csv'
path_output = '/home/brainlab/mritoolbox_predictors.csv'

n = 150

predictors = []
with open(path_mritoolbox_predictors, "rt", encoding='ascii') as infile:
    read = csv.reader(infile)
    for row in read:
        predictors.append(np.asarray(row).astype(np.float))

downsample_predictor = []

for i_pred in range(len(predictors)):
    window = len(predictors[i_pred]) / n
    print(len(predictors[i_pred]))
    print(window)

    if len(predictors[i_pred]) > n:
        sample_reduced = []
        for i in range(n):
            start = round(window * (i))
            end = round(window * (i + 1))
            sample_reduced.append(np.mean(predictors[i_pred][start:end]))
        downsample_predictor.append(sample_reduced)
        #plt.plot(downsample_predictor[:, i_pred])
        #plt.show()

np.savetxt(path_output, np.transpose(np.asarray(downsample_predictor)), fmt='%10.6f', delimiter=',')



