from generalize_ising_model.ising_utils import save_graph, makedir
import networkx as nx

default_size = 40
default_no_entities = 20
dim = 3

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/'

path_output_data = path_input + 'geometrics/'
makedir(path_output_data)

#Random geometric graph
path_output_data_type = path_output_data + 'geometricxxx_' + str(dim) + '/'
makedir(path_output_data_type)

for i in range(default_no_entities):

    while True:
        G = nx.random_geometric_graph(default_size, 0.2, dim=dim)
        if len(list(nx.isolates(G))) == 0:
            break

    makedir(path_output_data_type + 'entity_' + str(i))
    save_graph(path_output_data_type + 'entity_' + str(i) + '/' + 'J_ij.csv', G)