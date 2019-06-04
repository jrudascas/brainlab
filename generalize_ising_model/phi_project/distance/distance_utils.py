import numpy as np
from scipy.signal import argrelextrema
import scipy.io
import matplotlib.pyplot as plt
import scipy.stats


def matrix_distance(A,B,type=None):
    return np.linalg.norm(A-B,ord=type)

def ks_test(A,B):
    return scipy.stats.ks_2samp(np.ravel(A),np.ravel(B))[0]
    #return scipy.stats.kstest(np.ravel(A),np.ravel(B))

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

def to_find_critical_temperature(data, temp, fit_type='rel_extrema'):
    y_test = data.copy()

    if fit_type == 'rel_extrema':
        local_max = argrelextrema(data, np.greater, order=1)

        y_test[local_max] = (y_test[np.array(local_max) + 1] + y_test[np.array(local_max) - 1]) / 2
        return temp[np.where(y_test == np.max(y_test[local_max]))]

    elif fit_type == 'max':
        local_max = np.where(data == max(data))
        return data[np.where(y_test == np.max(y_test[local_max]))]

def generate_ts(temperature_parameters,ts_type='log'):
    if ts_type == 'linear':
      return np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])
    elif ts_type == 'log':
      return np.logspace(temperature_parameters[0],np.log10(temperature_parameters[1]),temperature_parameters[2])
    else:
      print('Invalid temperature distribution. Use either "log" or "linear"')
      return []

def plot_distance(distance,ts,critical_temperature,temperature_distribution='log'):   #phi):


    f = plt.figure(figsize=(18, 10))  # plot the calculated values

    ax = f.add_subplot(2, 2, 1)
    plt.scatter(ts, distance, s=50, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Correlation Distance ", fontsize=20)
    plt.axvline(x=critical_temperature,linestyle='--',color='k')
    if temperature_distribution == 'log':
        ax.set_xscale('log')
    plt.show()

def to_normalize(J):
    max_J = np.max(J)
    min_J = np.min(J)

    if max_J >= 0 and max_J <= 1 and min_J >= 0 and min_J <= 1:
        return J
    else:
        return J / max_J

def get_Jij(dict,D):
    for key, value in dict.items():

        i_l = []
        j_l = []
        for v in value:
            for w in value:
                i_l.append(v - 1)
                j_l.append(w - 1)

        J = 1 / D[i_l, j_l]
        J[J == np.inf] = 0
        J[J == -np.inf] = 0
        J = J / np.max(J)
        J = np.reshape(J, (len(value), len(value)))

        return to_normalize(J)

def distance_wei(G):
    '''
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to node v. The average shortest path length is the
    characteristic path length of the network.
    Input:      L,      Directed/undirected connection-length matrix.
    Output:     D,      distance (shortest weighted path) matrix
                B,      number of edges in shortest weighted path matrix
    Notes:
       The input matrix must be a connection-length matrix, typically
    obtained via a mapping from weight to length. For instance, in a
    weighted correlation network higher correlations are more naturally
    interpreted as shorter distances and the input matrix should
    consequently be some inverse of the connectivity matrix.
       The number of edges in shortest weighted paths may in general
    exceed the number of edges in shortest binary paths (i.e. shortest
    paths computed on the binarized connectivity matrix), because shortest
    weighted paths have the minimal weighted distance, but not necessarily
    the minimal number of edges.
       Lengths between disconnected nodes are set to Inf.
       Lengths on the main diagonal are set to 0.
    Algorithm: Dijkstra's algorithm.
    '''

    n = len(G)
    D = np.zeros((n, n))  # distance matrix
    D[np.logical_not(np.eye(n))] = np.inf
    B = np.zeros((n, n))  # number of edges matrix

    for u in range(n):
        # distance permanence (true is temporary)
        S = np.ones((n,), dtype=bool)
        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                W, = np.where(G1[v, :])  # neighbors of shortest nodes

                td = np.array(
                    [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                d = np.min(td, axis=0)
                wi = np.argmin(td, axis=0)

                D[u, W] = d  # smallest of old/new path lengths
                ind = W[np.where(wi == 1)]  # indices of lengthened paths
                # increment nr_edges for lengthened paths
                B[u, ind] = B[u, v] + 1

            if D[u, S].size == 0:  # all nodes reached
                break
            minD = np.min(D[u, S])
            if np.isinf(minD):  # some nodes cannot be reached
                break

            V, = np.where(D[u, :] == minD)

    return D, B

def create_state_matrix(J):
    N = J.shape[-1]
    setting_int = np.linspace(0, ((2**N) - 1), num=2**N).astype(int)

    M = list(map(lambda x: list(np.binary_repr(x, width=N)), setting_int))
    M = np.flipud(np.fliplr(np.asarray(M).astype(np.int)))
    return M * 2 - 1


def monte_carlo_metropolis(J, spin_vec, t, iterations, thermalize_time=None):
    no_spin = len(spin_vec)
    from numpy.random import permutation, random, choice

    list_spin = []

    for i in range(int(iterations / no_spin)):
        spin_permutation = permutation(no_spin)
        for j in range(no_spin):
            # Calculate the change in energy using a vector notation instead of the full matrix multiplication. This is done for the upper triangular elements
            # and only a single number is calculated to save memory
            d_e = 2 * np.dot(np.delete(spin_vec, spin_permutation[j]),
                             np.delete(J[spin_permutation[j], :], spin_permutation[j])) * spin_vec[spin_permutation[j]]

            # Determine the spin vector that corresponds to this energy
            if d_e <= 0 or random() <= np.exp(-d_e / t):
                spin_vec[spin_permutation[j]] *= -1
            # If the flip is acceptable, add it to a list to be used for calculating thermalized time parameters
            list_spin.append(np.copy(spin_vec))

    if thermalize_time is not None:
        # The upper triangular index elements are determined and stored for calculating the energy. See the below example of why this is done.
        static, moving = np.triu_indices(no_spin, k=1)
        static = static.astype(np.int16)
        moving = moving.astype(np.int16)
        # Initiate the parameters to be used when calculating the termalization conditions
        index_thermalize_time = np.round(iterations * thermalize_time).astype(int)
        spin_thermalized = np.squeeze(np.array(list_spin))[index_thermalize_time:, :]
        energy = 0
        energy_squared = 0

        for i_spin in range(spin_thermalized.shape[0]):
            # Calculate the energy using the generalized Ising model. An example is given below
            ener = np.dot(spin_thermalized[i_spin, :][static] * spin_thermalized[i_spin, :][moving], -J[static, moving])
            # Update the parameters
            energy += ener
            energy_squared += ener ** 2


        # Get the final parameters and add on the new, thermalized values.
        es = energy
        ess = energy_squared
        ms = abs(np.sum(abs(np.sum(spin_thermalized, axis=1))))
        mss = np.sum(abs(np.sum(spin_thermalized, axis=1)) ** 2)

        # Clear the unneeded variables
        del list_spin, index_thermalize_time, spin_thermalized
        # Return the values calculated upon thermalization
        return es, ess, ms, mss
    else:
        # If not thermalized, return the spin vector that is the result of the initial metropolis algorithim
        return np.copy(spin_vec)

def initial_spin(N):
    # Set a random spin configuration as a initial condition
    initial = 2 * np.random.randint(2, size=(N, 1), dtype=np.int8) - 1
    return initial


def generalized_ising_model(J, temperature_parameters, no_sim=500, thermalize_time=0.3, ts_type='log'):
    n = J.shape[-1]
    no_flip = 100 * n ** 2
    # no_flip = 10 * n ** 2
    avg_therm = no_flip * (1 - thermalize_time)

    E, M, S, H = [], [], [], []

    simulation = np.zeros((n, no_sim))
    ts = generate_ts(temperature_parameters, ts_type)
    simulated_fc = np.zeros((n, n, len(ts)))

    count = 0

    for tT in ts:
        # print('|', end='')

        spin_vec = initial_spin(n)

        es, ess, ms, mss = monte_carlo_metropolis(J, spin_vec, tT, no_flip,
                                                                thermalize_time=thermalize_time)


        E.append((es / avg_therm) / n)
        M.append((ms / avg_therm) / n)
        S.append((((mss / avg_therm) - (ms / avg_therm) ** 2) / n / tT) / n)
        H.append((((ess / avg_therm) - (es / avg_therm) ** 2) / n / tT ** 2) / n)

        for sim in range(no_sim):
            spin = monte_carlo_metropolis(J, spin_vec, tT, n)
            simulation[:, sim] = spin[:, 0]

        simulated_fc[:, :, count] = np.corrcoef(simulation[:, :])
        count += 1

    critical_temperature = to_find_critical_temperature(np.asarray(S), ts)

    return (np.asarray(simulated_fc), np.asarray(critical_temperature), np.asarray(E), np.asarray(M), np.asarray(S),
            np.asarray(H))
