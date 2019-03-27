from generalize_ising_model.ising_utils import to_normalize, to_save_results, correlation_function, dim, find_nearest
from os import walk
import numpy as np
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from networkx.utils import *


path_simulation_output = ['/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/15_degree_20',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/5_degree_40',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/17_degree_60']
#                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/4_undirected_weighted']

default_size = 40
susceptibility_exp = []
dimensionality_ = []
degre_ = []
max_degre_ = []

'''

for path in path_simulation_output:
    print(path)
    dimensionality_exp = []
    degree_exp = []
    max_degree_exp = []
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

            for entity in natsorted(os.listdir(path_simulation)):
                path_entity = path_simulation + '/' + entity + '/'

                if os.path.isdir(path_entity):
                    #print(entity)

                    simulated_matrix = np.load(path_entity + 'sim_fc.npy')
                    J = np.loadtxt(path_entity + 'J_ij.csv', delimiter=',')
                    critical_temperature = np.loadtxt(path_entity + 'ctem.csv', delimiter=',')
                    ctemp_sim.append(critical_temperature)
                    susceptibility_sim.append(np.loadtxt(path_entity + 'susc.csv', delimiter=','))

                    c, r = correlation_function(simulated_matrix, J)

                    G = nx.Graph(J)
                    degrees = sorted(d for n, d in G.degree())

                    index_ct = find_nearest(ts, critical_temperature)
                    dimensionality = dim(c, r, index_ct)
                    if not np.isinf(r[-1]):
                        dimensionality_exp.append(dimensionality)
                        degree_exp.append(np.mean(degrees))
                        max_degree_exp.append(np.max(degrees))

                        #print(np.mean(degrees))
                        #print(str(dimensionality))
                        #print(str(np.max(degrees)))
#            if dimensionality_sim:
#                dimensionality_sim.remove(np.max(dimensionality_sim)) #Removing maximal dimensionality (Probably is other outlier)
            #dimensionality_exp.append(dimensionality_sim)
            #degree_exp.append(degree_sim)
            #max_degree_exp.append(max_degre_sim)
    dimensionality_.append(dimensionality_exp)
    degre_.append(degree_exp)
    max_degre_.append(max_degree_exp)

fig, ax = plt.subplots(figsize=(10, 7))

colors = ['blue', 'green', 'red', 'black']

cont = 0

for dim, d, m in zip(dimensionality_, degre_, max_degre_):
#    new_dim = []
#    new_d = []
#    new_m = []
#    for dim, d_, m_ in zip(exp, d, m):
#        if dim:
#            new_dim.append(dim)
#            new_d.append(d_)
#            new_m.append(m_)

    #parts = plt.violinplot(new_dim, positions=np.array(new_d), showmeans=True, showmedians=False)
    plt.scatter(d, dim, c=colors[cont])
    #for pc in parts['bodies']:
    #    pc.set_facecolor(colors[cont])
    cont +=1

blue_patch = mpatches.Patch(color='blue', label='Graph Size = 20')
green_patch = mpatches.Patch(color='green', label='Graph Size = 40')
red_patch = mpatches.Patch(color='red', label='Graph Size = 60')
#black_patch = mpatches.Patch(color='black', label='Weighted 80%')

plt.legend(handles=[blue_patch, green_patch, red_patch])
plt.xlabel("Graph degree")
plt.ylabel("Dimensionality")

#plt.xticks(np.array(new_d), list(map(str, new_size)))
plt.show()

'''

path_simulation_output = ['/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/16_density_20',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/6_density_40']
                          #'/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/13_undirected_unweighted_0.8']
#                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/3_undirected_unweighted',
#                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/4_undirected_weighted']

sizes_ = np.linspace(0.05, 100, num=19).astype(np.int16)

#sizes_ = []

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
                    #print(entity)

                    simulated_matrix = np.load(path_entity + 'sim_fc.npy')
                    J = np.loadtxt(path_entity + 'J_ij.csv', delimiter=',')
                    critical_temperature = np.loadtxt(path_entity + 'ctem.csv', delimiter=',')
                    ctemp_sim.append(critical_temperature)
                    susceptibility_sim.append(np.loadtxt(path_entity + 'susc.csv', delimiter=','))

                    c, r = correlation_function(simulated_matrix, J)

                    index_ct = find_nearest(ts, critical_temperature)
                    dimensionality = dim(c, r, index_ct)
                    if not np.isinf(r[-1]):
                        dimensionality_sim.append(dimensionality)
#            if dimensionality_sim:
#                dimensionality_sim.remove(
#                    np.max(dimensionality_sim))  # Removing maximal dimensionality (Probably is other outlier)
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

blue_patch = mpatches.Patch(color='blue', label='Graph Size = 20')
green_patch = mpatches.Patch(color='green', label='Graph Size = 40')
red_patch = mpatches.Patch(color='red', label='Graph Size = 60')
# black_patch = mpatches.Patch(color='black', label='Weighted 80%')

plt.legend(handles=[blue_patch, green_patch])
plt.xlabel("Graph density")
plt.ylabel("Dimensionality")

plt.xticks(np.array(new_size)/10, list(map(str, new_size)))
plt.show()