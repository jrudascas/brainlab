
import scipy.io
from scipy.signal import hilbert
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import animation
from QMBrain.utils import *


filepathMat = '/home/user/Desktop/QMBrain/EEG1.mat'
filepathChanLoc = '/home/user/Desktop/QMBrain/chanLocXY.csv'
filepathData = '/home/user/Desktop/QMBrain/data.csv'
filepathTimes = '/home/user/Desktop/QMBrain/times.csv'



#dataSet = load_matrix(filepathMat)['EEG1']

data = load_matrix(filepathData)
times = load_matrix(filepathTimes)

chanLocs = load_matrix(filepathChanLoc)

x = chanLocs[:,0]
y = chanLocs[:,1]

hilbertTransData = hilbert(data)

amplitude = np.abs(hilbertTransData)

phase = np.unwrap(np.angle(hilbertTransData))

ampMag = np.linalg.norm(amplitude.T)

normAmp = (np.asarray(amplitude.T)/np.asarray(ampMag))

probability = normAmp*normAmp

xAvg = probability@x
yAvg = probability@y

xSqrAvg = probability@(x*x)
ySqrAvg = probability@(y*y)

dx = np.sqrt(xSqrAvg-(xAvg*xAvg))
dy = np.sqrt(ySqrAvg-(yAvg*yAvg))


#Calculate momentum of brain state

pxSum = np.zeros((len(times),len(x)),dtype=np.complex64)
pxSqrSum = np.copy(pxSum)

pySum = np.zeros((len(times),len(y)),dtype=np.complex64)
pySqrSum = np.copy(pySum)

for i in range(len(x)):
    current_x = x[i]
    current_y = y[i]

    #find closest nodes

    xDiff = current_x - x
    yDiff = current_y - y

    xPairs = np.where(abs(xDiff)>abs(yDiff))
    yPairs = np.where(abs(yDiff)>abs(xDiff))

    distance = np.sqrt(np.square(xDiff)+np.square(yDiff))

    xPairDistance = distance[xPairs]
    yPairDistance = distance[yPairs]

    firstX = sorted(xPairs)[-1][-1]
    secondX = sorted(xPairs)[-1][-2]

    firstY = sorted(yPairs)[-1][-1]
    secondY = sorted(yPairs)[-1][-2]

    diffX1 = x[firstX]-current_x
    diffX2 = x[secondX] - current_x

    diffY1 = x[firstY] - current_y
    diffY2 = x[secondY] - current_y

    psiConj = normAmp[:,i]*np.exp((-1j)*phase[i,:])
    psi_n = normAmp[:,i]*np.exp((1j)*phase[i,:])

    psi_n1x = normAmp[:,firstX]*np.exp((1j)*(phase[firstX,:]))
    psi_n2x = normAmp[:, secondX]*np.exp((1j)*(phase[secondX,:]))

    psi_n1y = normAmp[:, firstY]*np.exp((1j) * (phase[firstY,:]))
    psi_n2y = normAmp[:, secondY]*np.exp((1j) * (phase[secondY,:]))

    pxSum[:,i] = 0.5 * (psiConj*((psi_n1x-psi_n)/diffX1)+psiConj*((psi_n2x-psi_n)/diffX2))
    pxSqrSum[:,i] = 0.5 * (psiConj*((psi_n-2*psi_n1x + psi_n2x)/np.square(diffX1)))

    pySum[:,i] = 0.5 * (psiConj*((psi_n1y-psi_n)/diffY1)+psiConj*((psi_n2y-psi_n)/diffY2))
    pySqrSum[:,i] = 0.5 * (psiConj*((psi_n-2*psi_n1y + psi_n2y)/np.square(diffY1)))

#print(pySum)

pxAvg = np.sum(pxSum,axis=1)
pxSqrAvg = np.sum(pxSqrSum,axis=1)

pyAvg = np.sum(pySum,axis=1)
pySqrAvg = np.sum(pySqrSum,axis=1)

#Considering only the length
pxAvgL = np.abs(pxAvg)
pxSqrAvgL = np.abs(pxSqrAvg)

pyAvgL = np.abs(pyAvg)
pySqrAvgL = np.abs(pySqrAvg)

# Find uncertainties
deltaX = np.sqrt(xSqrAvg-np.square(xAvg))
deltaY = np.sqrt(ySqrAvg-np.square(yAvg))

deltaPX = np.sqrt(pxSqrAvgL-np.square(pxAvgL))
deltaPY = np.sqrt(pySqrAvgL-np.square(pyAvgL))

f = plt.figure()
f.plt