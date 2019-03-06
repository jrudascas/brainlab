import numpy as np
from numpy.random import permutation, random
import time
import numba
import multiprocessing
import math
from generalize_ising_model.ising_utils import to_find_critical_temperature
import warnings
warnings.filterwarnings("ignore")

n_cpu = multiprocessing.cpu_count() - 1

# @numba.jit(nopython=True)
def initial_spin(N):
    # Set a random spin configuration as a initial condition
    initial = 2 * np.random.randint(2, size=(N, 1), dtype=np.int8) - 1
    return initial


def calc_energy(start_end):

    energy = 0
    energy_squard = 0

    spin_thermalized = start_end[0]
    static = start_end[1]
    moving = start_end[2]
    C = start_end[3]

    for i_spin in range(start_end[4], start_end[5]):
        A = spin_thermalized[i_spin, :][static]
        B = spin_thermalized[i_spin, :][moving]
        AB = A * B
        ener = np.dot(AB, C)
        energy += ener
        energy_squard += ener ** 2

    del A, B, AB, ener
    return energy, energy_squard


# Montecarlo Simulation - Metroplolis Algorithm
def monte_carlo_metropolis(J, spin_vec, t, iterations, thermalize_time=None):
    no_spin = len(spin_vec)
    static, moving = np.triu_indices(no_spin, k=1)
    static = static.astype(np.int16)
    moving = moving.astype(np.int16)

    list_spin = []

    for i in range(int(iterations/no_spin)):
        spin_permutation = permutation(no_spin)
        for j in range(no_spin):

            d_e = 2 * np.dot(np.delete(spin_vec, spin_permutation[j]), np.delete(J[spin_permutation[j], :], spin_permutation[j]))
            d_e *= spin_vec[spin_permutation[j]]

            if d_e <= 0 or random() <= np.exp(-d_e / t):
                spin_vec[spin_permutation[j]] *= -1

            list_spin.append(np.copy(spin_vec))
    '''
    nn=0
    spin_permutation = permutation(no_spin)
    for i in range(iterations):
        # print(i)
        if nn > no_spin - 1:
            nn = nn - no_spin
            spin_permutation = permutation(no_spin)

        d_e = 2 * np.dot(np.delete(spin_vec, spin_permutation[nn]), np.delete(J[spin_permutation[nn], :], spin_permutation[nn]))
        d_e *= spin_vec[spin_permutation[nn]]

        if d_e <= 0 or random() <= np.exp(-d_e / t):
            spin_vec[spin_permutation[nn]] *= -1
        nn += 1

        list_spin.append(np.copy(spin_vec))        
    '''
    if thermalize_time is not None:

        index_thermalize_time = np.round(iterations * thermalize_time).astype(int)
        spin_thermalized = np.squeeze(np.array(list_spin))[index_thermalize_time:, :].astype(np.int8)
        C = -J[static, moving]
        step_len = math.ceil(spin_thermalized.shape[0] / n_cpu)
        previus = 0
        l = []

        for next in range(n_cpu):
            if (next + 1) * step_len > spin_thermalized.shape[0]:
                l.append((spin_thermalized, static, moving, C, previus, spin_thermalized.shape[0]))
            else:
                l.append((spin_thermalized, static, moving, C, previus, int((next + 1) * step_len)))

            previus = int((next + 1) * step_len)

        pool = multiprocessing.Pool(n_cpu)
        results = np.asarray(pool.map(calc_energy, l))
        pool.close()
        pool.join()

        es = np.sum(results[:, 0])
        ess = np.sum(results[:, 1])
        ms = abs(np.sum(abs(np.sum(spin_thermalized, axis=1))))
        mss = np.sum(abs(np.sum(spin_thermalized, axis=1)) ** 2)



        return es, ess, ms, mss
    else:
        return spin_vec

        #return 2,2,2,2

def generalized_ising(Jij, temperature_parameters=(0.1, 5, 100), no_simulations=100, thermalize_time=0.3):
    n = Jij.shape[-1]
    ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])
    no_flip = 10 * n ** 2
    avg_therm = no_flip * (1 - thermalize_time)
    no_temperature = len(ts)
    simulated_fc = np.zeros((n, n, no_temperature))
    E, M, S, H = np.zeros(no_temperature), np.zeros(no_temperature), np.zeros(no_temperature), np.zeros(no_temperature)
    simulation = np.zeros((n, no_simulations, no_temperature))

    #pool = multiprocessing.Pool(n_cpu)

    for tT in range(no_temperature):
        print('|', end='')
        start_time = time.time()
        spin_vec = initial_spin(n)

        es, ess, ms, mss = monte_carlo_metropolis(Jij, spin_vec, ts[tT], no_flip, thermalize_time)

        E[tT] = (es / avg_therm) / n
        M[tT] = (ms / avg_therm) / n
        S[tT] = (((mss / avg_therm) - (ms / avg_therm) ** 2) / n / ts[tT]) / n
        H[tT] = (((ess / avg_therm) - (es / avg_therm) ** 2) / n / ts[tT] ** 2) / n

        for sim in range(no_simulations):
            spin = monte_carlo_metropolis(Jij, spin_vec, ts[tT], n)
            simulation[:, sim, tT] = spin[:, 0]

        simulated_fc[:, :, tT] = np.corrcoef(simulation[:, :, tT])
        #print(time.time() - start_time)
    critical_temperature = to_find_critical_temperature(S, ts)

    return simulated_fc, critical_temperature, E, M, S, H
