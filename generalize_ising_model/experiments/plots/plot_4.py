from generalize_ising_model.ising_utils import to_normalize, to_save_results, correlation_function, dim, find_nearest
from os import walk
import numpy as np
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from networkx.utils import *

path_simulation_output = ['/home/user/Desktop/experiment_1/simulation/density_25_size_20',
                          '/home/user/Desktop/experiment_1/simulation/density_50_size_20',
                          '/home/user/Desktop/experiment_1/simulation/density_100_size_20']

sizes_ = np.linspace(5, 100, num=20).astype(np.int16)

dimensionality_ = []
labels = ['0 - 0.05',
          '0.05 - 0.1',
          '0.1 - 0.15',
          '0.15 - 0.2',
          '0.2 - 0.25',
          '0.25 - 0.3',
          '0.3 - 0.35',
          '0.35 - 0.4',
          '0.4 - 0.45',
          '0.45 - 5',
          '0.5 - 0.55',
          '0.55 - 0.6',
          '0.6 - 0.65',
          '0.65 - 0.7',
          '0.7 - 0.75',
          '0.75 - 0.8',
          '0.8 - 0.85',
          '0.85 - 0.9',
          '0.9 - 0.95',
          '0.95 - 1']

for path in path_simulation_output:
    print(path)
    dimensionality_exp = []
    for simulation in natsorted(os.listdir(path)):

        path_simulation = path + '/' + simulation

        if os.path.isdir(path_simulation):
            print()
            print(simulation)
            print()

            pkl_file = open(path_simulation + '/parameters.pkl', 'rb')
            simulation_parameters = pickle.load(pkl_file)
            pkl_file.close()
            ts = np.linspace(simulation_parameters['temperature_parameters'][0],
                             simulation_parameters['temperature_parameters'][1],
                             simulation_parameters['temperature_parameters'][2])

            susceptibility_sim = []
            ctemp_sim = []
            dimensionality_sim = []
            for entity in natsorted(os.listdir(path_simulation)):
                path_entity = path_simulation + '/' + entity + '/'

                if os.path.isdir(path_entity):

                    simulated_matrix = np.load(path_entity + 'sim_fc.npy')
                    J = np.loadtxt(path_entity + 'J_ij.csv', delimiter=',')
                    critical_temperature = np.loadtxt(path_entity + 'ctem.csv', delimiter=',')
                    ctemp_sim.append(critical_temperature)
                    susceptibility_sim.append(np.loadtxt(path_entity + 'susc.csv', delimiter=','))

                    c, r = correlation_function(simulated_matrix, J)

                    index_ct = find_nearest(ts, critical_temperature)

                    if not np.isinf(r[-1]) and not np.isnan(r[-1]):
                        dimensionality = dim(c, r, index_ct)
                        dimensionality_sim.append(dimensionality)
                    else:
                        print('Bad')
            dimensionality_exp.append(dimensionality_sim)
    dimensionality_.append(dimensionality_exp)

fig, ax = plt.subplots(figsize=(10, 7))

colors = ['blue', 'green', 'red', 'black']
cont = 0
for exp in dimensionality_:
    new_dim = []
    new_size = []
    for dim, size in zip(exp, sizes_):
        if dim:
            new_dim.append(dim)
            new_size.append(size)

    parts = plt.violinplot(new_dim, positions=np.array(new_size)/10, showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(colors[cont])
    cont += 1

blue_patch = mpatches.Patch(color='blue', label='Density Level = 25%')
green_patch = mpatches.Patch(color='green', label='Density Level = 50%')
red_patch = mpatches.Patch(color='red', label='Density Level = 100%')
# black_patch = mpatches.Patch(color='black', label='Weighted 80%')

plt.legend(handles=[blue_patch, green_patch, red_patch])
plt.xlabel("Value range")
plt.ylabel("Dimensionality")

plt.xticks(np.array(new_size)/10, labels, rotation='vertical')
plt.savefig('range_size_20.png')
plt.show()
