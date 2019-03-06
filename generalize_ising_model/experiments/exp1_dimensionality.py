from generalize_ising_model.ising_utils import to_normalize, to_save_results, correlation_function, dim
from os import walk
import numpy as np

path_simulation_output = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_1'

for root, simulations, files in walk(path_simulation_output):
    for simulation in sorted(simulations):
        path_simulation = root + '/' + simulation
        for root2, entities, files2 in walk(path_simulation):
            simulation_parameters = np.load(path_simulation + '/parameters.npy')
            for entity in sorted(entities):
                path_entity = path_simulation + '/' + entity + '/'
                critical_temperature = np.loadtxt(path_entity + 'ctem.csv', delimiter=',')
                simulated_matrix = np.load(path_entity + 'sim_fc.npy')
                J = np.loadtxt(path_entity + 'J_ij.csv', delimiter=',')

                c, r = correlation_function(simulated_matrix, J)

                print(dim(c, r, 8))