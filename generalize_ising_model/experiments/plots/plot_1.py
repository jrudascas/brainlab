from generalize_ising_model.ising_utils import to_normalize, to_save_results, correlation_function, dim, find_nearest
from os import walk
import numpy as np
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

path_simulation_output = ['/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/0_hcp']
#                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/3_undirected_unweighted',
#                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/4_undirected_weighted']

susceptibility_exp = []
dimensionality_ = []
sizes_ = []

for path in path_simulation_output:
    print(path)
    sizes_exp = []
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

            sizes_exp.append(simulation_parameters['temperature_parameters'][1])
            susceptibility_sim = []
            ctemp_sim = []
            dimensionality_sim = []
            for entity in natsorted(os.listdir(path_simulation)):
                path_entity = path_simulation + '/' + entity + '/'

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
                    if dimensionality != 3: #Outliear
                        dimensionality_sim.append(dimensionality)

            dimensionality_sim.remove(np.max(dimensionality_sim)) #Removing maximal dimensionality (Probably is other outlier)
            dimensionality_exp.append(dimensionality_sim)
    sizes_.append(sizes_exp)
    dimensionality_.append(dimensionality_exp)

fig, ax = plt.subplots(figsize=(7, 4))

colors = ['blue', 'green', 'red', 'black']
cont = 0
for exp in dimensionality_:
    parts = plt.violinplot(exp, positions=np.array(sizes_[cont])/10, showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(colors[cont])
    cont +=1

blue_patch = mpatches.Patch(color='blue', label='Weighted 100%')
green_patch = mpatches.Patch(color='green', label='Unweighted 80%')
red_patch = mpatches.Patch(color='red', label='Weighted 80%')
black_patch = mpatches.Patch(color='black', label='Weighted 80%')

plt.legend(handles=[blue_patch])
plt.xlabel("Graph Size")
plt.ylabel("Dimensionality")

plt.xticks(np.array(sizes_exp)/10, list(map(str, sizes_exp)))
plt.show()