import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def load_matrix(file):
    extension = file.split('.')[1]
    if str(extension) == 'csv':
        return np.genfromtxt(file,delimiter = ',')
    elif str(extension) == 'npy':
        return np.load(file)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(file)
    elif str(extension) == 'npz':
        return np.load(file)

def moving_avg(array):
    return np.sum(array,axis=0)/array.shape[-1]

def plot_av(data,ts,xlabel = "Temperature (T)",ylabel = 'Phi' ):
    f = plt.figure(figsize=(18, 10))  # plot the calculated values
    ax1 = f.add_subplot(1, 1, 1)
    ax1.scatter(ts, data, s=50, marker='o', color='IndianRed')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    ax1.set_xscale('log')
    plt.show()

def subplt_avg(phi,phiSus,mag,sus,ts,path_output,dim):
    f = plt.figure(figsize=(18, 10))  # plot the calculated values

    ax1 = f.add_subplot(2, 2, 1)
    ax1.scatter(ts, sus, s=50, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)
    ax1.set_xscale('log')

    ax2 = f.add_subplot(2, 2, 2)
    ax2.scatter(ts, phi, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi", fontsize=20)
    ax2.set_xscale('log')

    ax3 = f.add_subplot(2, 2, 3)
    ax3.scatter(ts, mag, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Magnetization", fontsize=20)
    ax3.set_xscale('log')

    ax4 = f.add_subplot(2, 2, 4)
    ax4.scatter(ts, phiSus, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi_sus", fontsize=20)
    ax4.set_xscale('log')

    #plt.show()f=
    plt.savefig(path_output + 'plots_'  + str(dim) + '.png', dpi=300)
