from generalize_ising_model.ising_utils import to_normalize, to_save_results, correlation_function, dim, find_nearest
from os import walk
import numpy as np
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import os

path_simulation_output = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/undirected_weighted/'

susceptibility_exp = []
dimensionality_exp = []
sizes_exp = []
for simulation in natsorted(os.listdir(path_simulation_output)):

    path_simulation = path_simulation_output + '/' + simulation

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
                dimensionality_sim.append(dim(c, r, index_ct))
                #print(dim(c, r, index_ct))

        dimensionality_exp.append(dimensionality_sim)

    #    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

        # plot violin plot
plt.violinplot(dimensionality_exp, positions=np.array(sizes_exp)/10, showmeans=False, showmedians=True)
plt.xticks(np.array(sizes_exp)/10, list(map(str, sizes_exp)))
        #plt.scatter(np.linspace(0, 49, num=50), dimensionality_sim)
plt.show()