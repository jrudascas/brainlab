from generalize_ising_model.ising_utils import to_normalize, to_save_results, correlation_function, dim, find_nearest
from os import walk
import numpy as np
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from networkx.utils import *

path_simulation_output = ['/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/000geometrics/geometricxxx_1',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/geometric']
#                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/4_undirected_weighted']

sizes_ = [1, 2]

dimensionality_exp = []
for simulation in path_simulation_output:

    if os.path.isdir(simulation):
        print()
        print(simulation)
        print()

        pkl_file = open(simulation + '/parameters.pkl', 'rb')
        simulation_parameters = pickle.load(pkl_file)
        pkl_file.close()
        ts = np.linspace(simulation_parameters['temperature_parameters'][0],
                         simulation_parameters['temperature_parameters'][1],
                         simulation_parameters['temperature_parameters'][2])

        susceptibility_sim = []
        ctemp_sim = []
        dimensionality_sim = []
        for entity in natsorted(os.listdir(simulation)):
            path_entity = simulation + '/' + entity + '/'

            if os.path.isdir(path_entity):
                print(entity)

                simulated_matrix = np.load(path_entity + 'sim_fc.npy')
                J = np.loadtxt(path_entity + 'J_ij.csv', delimiter=',')
                critical_temperature = np.loadtxt(path_entity + 'ctem.csv', delimiter=',')
                ctemp_sim.append(critical_temperature)
                susceptibility_sim.append(np.loadtxt(path_entity + 'susc.csv', delimiter=','))

                c, r = correlation_function(simulated_matrix, J)

                index_ct = find_nearest(ts, critical_temperature)
                dimensionality = dim(c, r, index_ct)
                if not np.isinf(r[-1]):
                    dimensionality_sim.append(dimensionality)
        dimensionality_exp.append(dimensionality_sim)

fig, ax = plt.subplots(figsize=(10, 7))

colors = ['blue', 'green', 'red', 'black', 'cyan']

parts = plt.violinplot(dimensionality_exp, positions=np.array(sizes_), showmeans=True, showmedians=False)

cont = 0
for pc in parts['bodies']:
    pc.set_facecolor(colors[cont])
    cont += 1

blue_patch = mpatches.Patch(color='blue', label='Graph dimensionality = 1')
green_patch = mpatches.Patch(color='green', label='Graph dimensionality = 2')
#red_patch = mpatches.Patch(color='red', label='Graph dimensionality = 3')
#red_patch = mpatches.Patch(color='black', label='Graph dimensionality = 4')
#red_patch = mpatches.Patch(color='cyan', label='Graph dimensionality = 5')
# black_patch = mpatches.Patch(color='black', label='Weighted 80%')

plt.legend(handles=[blue_patch, green_patch])
plt.xlabel("Graph dimensionality")
plt.ylabel("Dimensionality")

# plt.xticks(np.array(new_size)/10, list(map(str, new_size)))
plt.show()
