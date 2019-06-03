import numpy as np
import pyphi
from pyphi.compute import phi
import matplotlib.pyplot as plt

def to_calculate_mean_phi(tpm, spin_mean,t):
    N = tpm.shape[-1]

    setting_int = np.linspace(0, np.power(2, N) - 1, num=np.power(2, N)).astype(int)

    M = list(map(lambda x: list(np.binary_repr(x, width=N)), setting_int))
    M = np.flipud(np.fliplr(np.asarray(M).astype(np.int)))

    num_states = M.shape[0]
    phi_values = []

    network = pyphi.Network(tpm)
    for state in range(num_states):
        if spin_mean[state] != 0:
            phi_values.append(phi(pyphi.Subsystem(network, M[state, :], range(network.size))))
            #phi_values_sum = phi_values*spin_mean[state]

    phi_values_sqr = [phi_ * phi_ for phi_ in phi_values]

    weigth = spin_mean[np.where(spin_mean != 0)]

    phiSum = np.sum(phi_values*weigth)
    phiSus = (np.sum(phi_values_sqr*weigth) - (phiSum * phiSum)) / (N*t)

    return np.mean(phi_values), phiSum, phiSus


def to_estimate_tpm_from_ising_model(J, T):
    N = J.shape[-1]
    setting_int = np.linspace(0, np.power(2, N) - 1, num=2**N).astype(int)

    M = list(map(lambda x: list(np.binary_repr(x, width=N)), setting_int))
    M = np.flipud(np.fliplr(np.asarray(M).astype(np.int)))
    M = M * 2 - 1

    a = np.matmul(M, J)
    b = np.multiply(M, 2)
    dE_sys = np.multiply(a, b)

    detFlip = np.where(dE_sys <= 0)
    notdetFlip = np.where(dE_sys > 0)
    FPM = dE_sys
    FPM[detFlip] = 1
    FPM[notdetFlip] = np.exp(-FPM[notdetFlip] / T)

    TPM = np.zeros((2**N, 2**N))

    for iTPM in range(2**N):
        for jTPM in range(2**N):

            lgclP = M[iTPM, :] - M[jTPM, :]
            lgclP = lgclP.astype(bool)

            if np.sum(lgclP.astype(int)) > 1:
                TPM[iTPM, jTPM] = 0
            elif np.sum(lgclP.astype(int)) == 1:
                FPMtemp = np.copy(FPM[iTPM, :])
                TPM[iTPM, jTPM] = FPMtemp[lgclP] / N
            else:
                FPMtemp = np.copy(FPM[iTPM, :])
                FPMtemp[~lgclP] = 1 - FPMtemp[~lgclP]
                TPM[iTPM, jTPM] = np.mean(FPMtemp)

    return TPM, FPM

def to_save_phi(ts, phi , phiSum, phiSus, S, critTemp, network, path_output):
    default_delimiter = ','
    format = '%1.5f'

    np.savetxt(path_output + 'temps.csv', ts, delimiter=default_delimiter, fmt=format)
    np.savetxt(path_output + 'phi.csv', phi, delimiter=default_delimiter, fmt=format)
    np.savetxt(path_output + 'phiSum.csv', phiSum, delimiter=default_delimiter, fmt=format)
    np.savetxt(path_output + 'phiSus.csv', phiSus, delimiter=default_delimiter, fmt=format)

    f = plt.figure(figsize=(18, 10))  # plot the calculated values

    ax1 = f.add_subplot(2, 2, 1)
    ax1.scatter(ts, S, s=50, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)

    #plt.xticks(x)
    plt.axvline(x=critTemp,linestyle='--',color='k')
    ax1.set_xscale('log')
    #plt.xticks(x, ['0', '1', '2', '3', '4', '5'])

    ax2 = f.add_subplot(2, 2, 2)
    ax2.scatter(ts, phi, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi", fontsize=20)
    #plt.xticks(x)

    plt.axvline(x=critTemp, linestyle='--', color='k')
    ax2.set_xscale('log')
  #  plt.xticks(x, ['0', '1', '2', '3', '4', '5'])

    ax3 = f.add_subplot(2, 2, 3)
    ax3.scatter(ts, phiSum, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi_sum", fontsize=20)
    #plt.xticks(x)
    plt.axvline(x=critTemp, linestyle='--', color='k')
    ax3.set_xscale('log')
    #plt.xticks(x, ['0', '1', '2', '3', '4', '5'])


    ax4 = f.add_subplot(2, 2, 4)
    ax4.scatter(ts, phiSus, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi_sus", fontsize=20)
    #ax4.xticks(x,['0','1','2','3','4','5'])
    plt.axvline(x=critTemp, linestyle='--', color='k')
    ax4.set_xscale('log')
    #plt.xticks(x, ['0', '1', '2', '3', '4', '5'])

    #plt.show()f=
    plt.savefig(path_output + 'plots_' + str(network) + '.png', dpi=300)

    plt.close()