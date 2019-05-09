import numpy as np
from networkx.utils import *
import os
from generalize_ising_model.ising_utils import save_graph, makedir

# Average Value
path_input = '/home/user/Desktop/'
simulation_name = 'experiment_1'
path_output_data = path_input + simulation_name
path_outputexp_exp = path_output_data + '/average_value/'
default_size = 60
default_no_entities = 20

if not os.path.exists(path_outputexp_exp):
    os.mkdir(path_outputexp_exp)

ranges = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

previus = 0
for range_ in ranges:
    path_output_data_density = path_outputexp_exp + '/value_' + str(previus) + '-' + str(range_)
    if not os.path.exists(path_output_data_density):
        os.mkdir(path_output_data_density)

    for entity in range(default_no_entities):
        path_output_data_density_entity = path_output_data_density + '/' + 'entity_' + str(entity)
        if not os.path.exists(path_output_data_density_entity):
            os.mkdir(path_output_data_density_entity)

        #G = nx.generators.fast_gnp_random_graph(default_size, 1, directed=False)

        while True:
            G = nx.dense_gnm_random_graph(default_size, default_size*default_size/8)
            G.remove_edges_from(nx.selfloop_edges(G))

            if len(list(nx.isolates(G))) == 0:
                break

        for (u, v) in G.edges():
            while True:
                rr = random.random()
                if rr >= previus and rr < range_:
                    break

            G.edges[u, v]['weight'] = rr

        #print('Len: ' + str(len(G.edges())))
        save_graph(path_output_data_density_entity + '/' + 'J_ij.csv', G)
    previus = range_