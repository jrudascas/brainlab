from generalize_ising_model.ising_utils import to_normalize, to_save_results, correlation_function, dim, find_nearest
from os import walk
import networkx as nx
import numpy as np
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches


path_simulation_output = ['/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/0_others/moreno_seventh_seventh',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/11_social/karate_club',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/geometric',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/intersection',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/small_world',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/10_geometric/waxman',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/14_undirected_weighted_1.0/N_75',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/14_undirected_weighted_1.0/N_25']
'''

path_simulation_output = ['/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/hcp',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/00celegant_new',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/cat-mixed-species_brain',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/macaque-rhesus_brain',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/2enzimes_g16',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/1proteine_DD_g1065',
                          '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/mouse_brain']
                          ##'/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/1mouse_visual_cortex',
                          ##'/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/11_social/caveman_graph',

                          #]
'''

l = len(path_simulation_output)


labels_ = ['moreno_seventh_seventh',
           'karate_club',
           'geometric',
           'intersection',
           'small_world',
           'waxman',
           'weighted_size_75_density_100',
           'weighted_size_25_density_100']
'''
labels_ = ['hcp',
           'c-elegant',
           'cat-mixed-species_brain',
           'macaque-rhesus_brain',
           'enzimes_g16',
           'proteine_DD_g1065',
           'mouse_brain']
           ##'mouse_visual_cortex',
           ##'caveman_graph',
'''
dimensionality_ = []

results = []
for simulation in path_simulation_output:

    if os.path.isdir(simulation):
        print()
        print(simulation)
        print()

        pkl_file = open(simulation + '/parameters.pkl', 'rb')
        simulation_parameters = pickle.load(pkl_file)
        pkl_file.close()
        ts = np.linspace(simulation_parameters['temperature_parameters'][0],
                         simulation_parameters['temperature_parameters'][1],
                         simulation_parameters['temperature_parameters'][2])

        dimensionality_sim = []
        critical_temperature_sim = []
        size_sim = []
        degree_sim = []
        sparcity = []

        for entity in natsorted(os.listdir(simulation)):
            path_entity = simulation + '/' + entity + '/'

            if os.path.isdir(path_entity):
                print(entity)

                simulated_matrix = np.load(path_entity + 'sim_fc.npy')
                J = np.loadtxt(path_entity + 'J_ij.csv', delimiter=',')
                critical_temperature = np.loadtxt(path_entity + 'ctem.csv', delimiter=',')

                c, r = correlation_function(simulated_matrix, J)

                print(nx.number_of_isolates(nx.Graph(J)))

                index_ct = find_nearest(ts, critical_temperature)
                dimensionality = dim(c, r, index_ct)
                if dimensionality != 3:  # Outliear
                    dimensionality_sim.append(dimensionality)
                    critical_temperature_sim.append(critical_temperature)
                    size_sim.append(J.shape[-1])
                    degrees = sorted(d for n, d in nx.Graph(J).degree())
                    degree_sim.append(np.mean(degrees))
                    sparcity.append(nx.density(nx.Graph(J))*100)

        #dimensionality_.append(np.mean(dimensionality_sim))
        results.append([np.mean(dimensionality_sim), np.mean(size_sim), np.mean(degree_sim), np.mean(sparcity)])
#fig, ax = plt.subplots(figsize=(10, 7))

#plt.scatter(np.linspace(1,l, num=l), dimensionality_)

#plt.xlabel("Graph name")
#plt.ylabel("Dimensionality")


#plt.xticks(np.linspace(1,l, num=l), labels_, rotation=90)
#plt.show()
#plt.savefig('GraphName.png')

import numpy as np
import pylab as pl

class Radar(object):

    def __init__(self, fig, titles, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]

        self.n = len(titles)
        self.angles = np.arange(90, 90+360, 360.0/self.n)
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i)
                         for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=14)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 6), angle=angle, labels=label)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 5)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


    def fill(self, values, alpha, color):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.fill(angle, values, alpha=alpha, color=color)


fig = pl.figure(figsize=(7, 7))

titles = ['', '', '', '']

m = np.array(results)

f1 = np.max(m[:, 0]) / 5
f2 = np.max(m[:, 1]) / 5
f3 = np.max(m[:, 2]) / 5
f4 = np.max(m[:, 3]) / 5
#f5 = np.max(m[:, 4]) / 5

labels = [
    list(map(str, np.round(np.linspace(np.max(m[:, 0])/5, np.max(m[:, 0]), num=5), 1))),
    list(map(str, np.linspace(np.max(m[:, 1])/5, np.max(m[:, 1]), num=5).astype(int))),
    list(map(str, np.linspace(np.max(m[:, 2])/5, np.max(m[:, 2]), num=5).astype(int))),
    list(map(str, np.linspace(np.max(m[:, 3])/5, np.max(m[:, 3]), num=5).astype(int))),
#    list(map(str, np.linspace(np.max(m[:, 4])/5, np.max(m[:, 4]), num=5).astype(int)))
]

radar = Radar(fig, titles, labels)

color = ['red',
         'blue',
         'green',
         'black',
         'purple',
         'orange',
         'cyan',
         'magenta',
         'yellow',
         'fuchsia',
         'teal',
         'pink',
         'brown',
         'gray',
         'navy',
         'knaki',
         'tomato',
         'olive'
         ]
cont = 0

for experiment in results:
    print(experiment)

    val = [experiment[0] / f1, experiment[1] / f2, experiment[2] / f3, experiment[3] / f4]
    radar.plot(val,  "-", lw=2, color=color[cont], alpha=0.7, label=labels_[cont])
    radar.fill(val, color=color[cont], alpha=0.05)
    cont += 1

radar.ax.legend()
fig.savefig('radar2.png', dpi=300)