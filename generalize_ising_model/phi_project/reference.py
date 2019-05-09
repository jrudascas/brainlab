import numpy as np
import scipy.io as sio
import pyphi
import time
from ising import gen_reservoir
import matplotlib.pyplot as plt
import os


def calc_mean_phi(J, TPM, spinBin):
    M = np.array(gen_reservoir(J.shape[1]), dtype='uint8')

    templen = T.shape[1]
    print('The number of data points to calculate Phi for is ' + str(templen))

    # tempstart = input('Start from data point: ')
    # tempstart = int(tempstart)
    tempstart = 0

    # tempend = input('End at data point: ')
    # tempend = int(tempend)
    tempend = templen

    # increment = input('Increment every _ data points: ')
    # increment = int(increment)
    increment = 1

    numStates = M.shape[0]

    ind = np.arange(tempstart, tempend, increment)  # indices of data points that phi will be calculated for
    T2 = T[0, ind]

    looplen = ind.shape[0]  # number of iterations of loop

    # phi = np.zeros([numStates,templen])
    phi = np.zeros([numStates, looplen])
    phiSqr = np.zeros([numStates, looplen])
    count = -1

    print('Calculating...')
    for temp in range(tempstart, tempend, increment):
        count += 1
        print(((temp) / (tempend - tempstart)) * 100, "% Complete")
        for state in range(numStates):  # numflips
            if spinBin[state, temp] != 0:
                start = time.time()
                # print("Starting state ", M[state,:], "at temp. ", T[0,temp])
                network = pyphi.Network(TPM[:, :, temp])
                # subsystem = pyphi.Subsystem(network, S[:,state,temp], range(network.size))
                subsystem = pyphi.Subsystem(network, M[state, :], range(network.size))
                # print(subsystem)
                phi[state, count] = pyphi.compute.big_phi(subsystem)
                phiSqr[state, count] = phi[state, count] * phi[state, count]
                print("Phi = ", phi[state, count])
                # input()
                end = time.time()
                # print(end - start, "seconds elapsed")

    phiSum = np.sum(phi * spinBin[:, ind], 0)
    phiSqrSum = np.sum(phiSqr * spinBin[:, ind], 0)

    phiSus = (phiSqrSum - phiSum * phiSum) / (T2 * J.shape[0])
    # print('Done!')

    return phiSum, phiSus, T2;


networks = ["Aud", "DMN", "ECN_R", "ECN_L", "Salience", "Sensorimotor", "VISM", "VISO", "VISL"]
methods = ["met", "glaub"]

# directory = 'reduced_networks/'
directory = 'Parallel Code Stuff/Simulations/'
wd = 'Ising_Networks/Current_sims/'

for method in methods:

    print('Starting method: ' + method)

    for network in networks:

        print('Starting network: ' + network)

        tempDir = directory + wd + network + '/' + method + '/'
        print(tempDir)
        path, dirs, files = os.walk(tempDir).__next__()
        numFiles = len(files)

        for file in files:
            filepath = path + file

            print('Loading: ' + filepath)

            mat = sio.loadmat(filepath)

            T = mat['temp']
            J = mat['J']
            J = J != 0
            spinBin = mat['spinBin']
            # S = mat['EXPORT'] # spin time-series through temperature
            TPM = mat['TPM']

            # phiSum, phiSus, T2 = calc_mean_phi(J,TPM,spinBin)
            phiSum = 0
            phiSus = 0
            T2 = 0

            suffix = '_meanPhi'
            spath = directory + 'pyPhi/' + 'Ising_Networks/' + network + '/' + method + '/'
            filename, file_extension = os.path.splitext(file)

            nfilename = spath + filename + suffix

            os.makedirs(spath, exist_ok=True)

            np.savez(nfilename, phiSum=phiSum, phiSus=phiSus, T2=T2)

