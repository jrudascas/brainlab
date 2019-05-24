import numpy as np
import pyphi
from pyphi.compute import phi

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
