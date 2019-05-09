from networkx.utils import *
import networkx as nx
from generalize_ising_model.ising_utils import save_graph, makedir
import numpy as np
import os

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_1/'




path_output_data = path_input + 'biological/'
path = '/home/brainlab/Downloads/celegansneural/celegansneural.mtx'
path_output_data_type = path_output_data + 'celegans_neuronal/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
G.remove_edges_from(nx.selfloop_edges(G))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)



default_size = 40
default_no_entities = 20

# Geometric and Form

path_output_data = path_input + 'geometric/'
makedir(path_output_data)

#Random WaxMan Graph
path_output_data_type = path_output_data + 'waxman/'
makedir(path_output_data_type)
for i in range(default_no_entities):
    G = nx.generators.waxman_graph(default_size, beta=0.5, alpha=0.5)
    makedir(path_output_data_type + 'entity_' + str(i))
    save_graph(path_output_data_type + 'entity_' + str(i) + '/' + 'J_ij.csv', G)

#Random geometric graph
path_output_data_type = path_output_data + 'geometric/'
makedir(path_output_data_type)

for i in range(default_no_entities):
    G = nx.random_geometric_graph(default_size, 0.2)
    makedir(path_output_data_type + 'entity_' + str(i))
    save_graph(path_output_data_type + 'entity_' + str(i) + '/' + 'J_ij.csv', G)

#Random free scale graph
path_output_data_type = path_output_data + 'free_scale/'
makedir(path_output_data_type)

for i in range(default_no_entities):
    G = nx.scale_free_graph(default_size, alpha=0.01, beta=0.84, gamma=0.15, create_using=nx.Graph)
    makedir(path_output_data_type + 'entity_' + str(i))
    save_graph(path_output_data_type + 'entity_' + str(i) + '/' + 'J_ij.csv', G)

#Random Small World Graph
#path_output_data_type = path_output_data + 'small_world/'
#makedir(path_output_data_type)

#for i in range(default_no_entities):
    G = nx.navigable_small_world_graph(np.sqrt(default_size).astype(np.int8))
#    makedir(path_output_data_type + 'entity_' + str(i))
#    save_graph(path_output_data_type + 'entity_' + str(i) + '/' + 'J_ij.csv', G)

# Random Intersection Graph
path_output_data_type = path_output_data + 'intersection/'
makedir(path_output_data_type)

for i in range(default_no_entities):
    G = nx.k_random_intersection_graph(default_size, 10, 2)
    makedir(path_output_data_type + 'entity_' + str(i))
    save_graph(path_output_data_type + 'entity_' + str(i) + '/' + 'J_ij.csv', G)

# Social Graph
path_output_data = path_input + 'social/'
makedir(path_output_data)

# Karate club graph
path_output_data_type = path_output_data + 'karate_club/'
makedir(path_output_data_type)
G = nx.karate_club_graph()
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

# Caveman Graph
path_output_data_type = path_output_data + 'caveman_graph/'
makedir(path_output_data_type)
G = nx.caveman_graph(int(default_size / 10), 10)
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

path_output_data = path_input + 'biological/'
makedir(path_output_data)

#Enzimes structure
#http://networkrepository.com/ENZYMES-g16.php
path = '/home/brainlab/Downloads/ENZYMES_g16 (1)/ENZYMES_g16.edgelist'
path_output_data_type = path_output_data + 'enzimes_g16/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

# #C-Elegan neuronal structure
#http://networkrepository.com/celegansneural.php
path = '/home/brainlab/Downloads/celegansneural/celegansneural.mtx'
path_output_data_type = path_output_data + 'celegans_neuronal/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

#Dolphins relationships
#http://networkrepository.com/dolphins.php
path = '/home/brainlab/Downloads/dolphins/dolphins.mtx'
path_output_data_type = path_output_data + 'dolphins/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

#Protein structure
#http://networkrepository.com/DD-g1065.php
path = '/home/brainlab/Downloads/DD_g1065/DD_g1065.edges'
path_output_data_type = path_output_data + 'proteine_DD_g1065/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

#Mouse visual cortex
#http://networkrepository.com/bn-mouse-visual-cortex-2.php
path = '/home/brainlab/Downloads/bn-mouse_visual-cortex_2/bn-mouse_visual-cortex_2.edges'
path_output_data_type = path_output_data + 'mouse_visual_cortex/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

#Mouse brain network
#http://networkrepository.com/bn-mouse-brain-1.php
path = '/home/brainlab/Downloads/bn-mouse_brain_1/bn-mouse_brain_1.edges'
path_output_data_type = path_output_data + 'mouse_brain/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

#MACAQUE-RHESUS-BRAIN-1
#http://networkrepository.com/bn-macaque-rhesus-brain-1.php
path = '/home/brainlab/Downloads/bn-macaque-rhesus_brain_1/bn-macaque-rhesus_brain_1.edges'
path_output_data_type = path_output_data + 'macaque-rhesus_brain/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)

#CAT-MIXED-SPECIES-BRAIN-1
#http://networkrepository.com/bn-cat-mixed-species-brain-1.php
path = '/home/brainlab/Downloads/bn-cat-mixed-species_brain_1/bn-cat-mixed-species_brain_1.edges'
path_output_data_type = path_output_data + 'cat-mixed-species_brain/'
makedir(path_output_data_type)
G=nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph, data=(('weight',float),))
makedir(path_output_data_type + 'entity_' + str(0))
save_graph(path_output_data_type + 'entity_' + str(0) + '/' + 'J_ij.csv', G)


'''

path_output_data = path_input + 'clustering/'
makedir(path_output_data)

sequence = nx.random_powerlaw_tree_sequence(10, tries=500)
G = nx.random_degree_sequence_graph(sequence)
tri = [1,5,6,2,3,1,1,0,0,2]

joint_degree_sequence=zip(sequence,tri)
G = nx.random_clustered_graph(joint_degree_sequence, create_using=nx.Graph)

G = nx.gnm_random_graph(100,2200)

print(nx.average_clustering(G))
# print(G.number_of_nodes())
# print(G.number_of_edges())
# print(nx.betweenness_centrality(G))
# print(nx.degree(G))
'''