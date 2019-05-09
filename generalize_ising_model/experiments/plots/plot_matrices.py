import numpy as np
from nilearn import plotting

list_graph = ['/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/0_others/moreno_seventh_seventh/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/geometric/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/intersection/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/small_world/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/waxman/entity_1/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/11_social/karate_club/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/14_undirected_weighted_1.0/N_75/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/14_undirected_weighted_1.0/N_25/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/00celegant_new/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/1proteine_DD_g1065/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/2enzimes_g16/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/cat-mixed-species_brain/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/hcp/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/macaque-rhesus_brain/entity_0/J_ij.csv',
              '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/mouse_brain/entity_0/J_ij.csv']


for graph in list_graph:
    print(graph)
    J = np.loadtxt(graph, delimiter=',')
    plotting.plot_matrix(J, colorbar=True, vmax=1., vmin=0.)
    plotting.show()