import time
import numpy as np
from generalize_ising_model.core import generalized_ising
from os import walk
from generalize_ising_model.ising_utils import to_normalize, to_save_results
import os

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/HCP/'

default_Jij_name = 'J_ij.csv'
simulation_name = 'hcp'

# Ising Parameters
temperature_parameters = (0.05, 1, 100)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
no_simulations = 250  # Number of simulation after thermalization
thermalize_time = 0.3  #

dir_output_name = path_input + 'simulation_' + simulation_name
for root, dirs, files in walk(path_input + 'data/'):
    if not os.path.exists(dir_output_name):
        os.mkdir(dir_output_name)

    np.save(dir_output_name + '/' + 'parameters',
            {'temperature_parameters': temperature_parameters,
             'no_simulations': no_simulations,
             'thermalize_time': thermalize_time})

    for dir in sorted(dirs):
        print(dir)
        print (''.join('*' * temperature_parameters[2]))

        sub_dir_output_name = dir_output_name + '/' + dir + '/'
        if not os.path.exists(sub_dir_output_name):
            os.mkdir(sub_dir_output_name)
            J = to_normalize(np.loadtxt(root + dir + '/' + default_Jij_name, delimiter=','))

            start_time = time.time()
            simulated_fc, critical_temperature, E, M, S, H = generalized_ising(J,
                                                                               temperature_parameters=temperature_parameters,
                                                                               no_simulations=no_simulations,
                                                                               thermalize_time=thermalize_time)
            print(time.time() - start_time)

            to_save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, sub_dir_output_name)