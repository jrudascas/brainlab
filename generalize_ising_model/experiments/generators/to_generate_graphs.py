import numpy as np
from networkx.utils import *
import os
from generalize_ising_model.ising_utils import save_graph, makedir

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/'
simulation_name = 'experiment_1'
default_size = 40
default_no_entities = 100

path_output_data = path_input + simulation_name

makedir(path_input + simulation_name)

# Degree
path_outputexp_degree = path_output_data + '/degree/'
if not os.path.exists(path_outputexp_degree):
    os.mkdir(path_outputexp_degree)

degrees_expected = np.linspace(5, 35, num=20).astype(np.int8)

for degree in degrees_expected:
    path_output_data_degree = path_outputexp_degree + 'degree_' + str(degree)
    if not os.path.exists(path_output_data_degree):
        os.mkdir(path_output_data_degree)

    for entity in range(default_no_entities):
        path_output_data_degree_entity = path_output_data_degree + '/' + 'entity_' + str(entity)
        if not os.path.exists(path_output_data_degree_entity):
            os.mkdir(path_output_data_degree_entity)

        secuence = np.random.randint(1, degree * 2, default_size)
        G = nx.expected_degree_graph(secuence)
        G.remove_edges_from(nx.selfloop_edges(G))

        #degrees = sorted(d for n, d in G.degree())

        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = random.random()

        #print('Expected: ' + str(degree) + ' - ' + 'Estimated: ' + str(np.mean(degrees)))
        save_graph(path_output_data_degree_entity + '/' + 'J_ij.csv', G)

# Density
path_outputexp_density = path_output_data + '/density/'
if not os.path.exists(path_outputexp_density):
    os.mkdir(path_outputexp_density)

densities = np.linspace(default_size, int(default_size*default_size/2), num=20).astype(np.int16)

for density in densities:
    path_output_data_density = path_outputexp_density + '/density_' + str(density)
    if not os.path.exists(path_output_data_density):
        os.mkdir(path_output_data_density)

    for entity in range(default_no_entities):
        path_output_data_density_entity = path_output_data_density + '/' + 'entity_' + str(entity)
        if not os.path.exists(path_output_data_density_entity):
            os.mkdir(path_output_data_density_entity)

        G = nx.dense_gnm_random_graph(default_size, density)
        G.remove_edges_from(nx.selfloop_edges(G))

        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = random.random()

        #print('Len: ' + str(len(G.edges())))
        save_graph(path_output_data_density_entity + '/' + 'J_ij.csv', G)


