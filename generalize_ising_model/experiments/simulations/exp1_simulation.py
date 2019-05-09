import numpy as np
from generalize_ising_model.ising_utils import save_graph, to_save_results
import os
from generalize_ising_model.core import generalized_ising
import networkx as nx
import random
import pickle

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/'
simulation_name = 'experiment_1'

dir_output_name = path_input + simulation_name

thermalize_time = 0.3
spin_vector_sizes = range(5, 155, 10)
initial_temperature = 0
no_temperatures = 50
no_entities = 20
no_simulations = 250

if not os.path.exists(dir_output_name):
    os.mkdir(dir_output_name)

for N in sorted(spin_vector_sizes):
    temperature_parameters = (initial_temperature, N, no_temperatures)
    dir_output_subname = dir_output_name + '/' + 'N_' + str(N)
    if not os.path.exists(dir_output_subname):
        os.mkdir(dir_output_subname)

    print()
    print()
    print('Random - N: ' + str(N))

    output = open(dir_output_subname + '/' + 'parameters.pkl', 'wb')
    pickle.dump({'temperature_parameters':temperature_parameters, 'no_simulations':no_simulations, 'thermalize_time':thermalize_time}, output)
    output.close()

    for entity in range(no_entities):
        print()
        print('Entity: ' + str(entity + 1))
        print(''.join('*' * temperature_parameters[2]))
        dir_output_subname_entity = dir_output_subname + '/' + 'entity_' + str(entity) + '/'

        if not os.path.exists(dir_output_subname_entity):
            os.mkdir(dir_output_subname_entity)

            G = nx.generators.fast_gnp_random_graph(N, 1, directed=True)

            for (u, v) in G.edges():
                while True:
                    rr = random.random()
                    if rr > 0.9 and rr < 1.0:
                        break

                G.edges[u, v]['weight'] = rr

            matrix = nx.to_numpy_array(G)
            save_graph(dir_output_subname_entity + '/' + 'J_ij.csv', G)



            #J = np.random.rand(N, N)


            #simulated_fc, critical_temperature, E, M, S, H = generalized_ising(J,
            #                                                                   temperature_parameters=temperature_parameters,
            #                                                                   no_simulations=no_simulations,
            #                                                                   thermalize_time=thermalize_time)

            #to_save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, dir_output_subname_entity)