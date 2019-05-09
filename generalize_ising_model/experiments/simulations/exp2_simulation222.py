import time
import numpy as np
import os
import gc
import pickle
from generalize_ising_model.ising_utils import to_normalize, to_save_results
from generalize_ising_model.core import generalized_ising
from natsort import natsorted

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/'
default_Jij_name = 'J_ij.csv'

# Ising Parameters

no_temperature = 50
no_simulations = 250                        # Number of simulation after thermalization
thermalize_time = 0.3                       #

dir_output_name = path_input + 'simulation'
if not os.path.exists(dir_output_name):
    os.mkdir(dir_output_name)

path_input_aux = path_input + 'data/'
for dirs in natsorted(os.listdir(path_input_aux)):
    dir_output_name_case = dir_output_name+ '/' + dirs + '/'
    print(dirs)
    if not os.path.exists(dir_output_name_case):
        os.mkdir(dir_output_name_case)

    for dir in natsorted(os.listdir((path_input_aux + '/' + dirs))):
        print(dir)
        print (''.join('*' * no_temperature))

        dir_output_name_case_exp = dir_output_name_case + '/' + dir
        if not os.path.exists(dir_output_name_case_exp):
            os.mkdir(dir_output_name_case_exp)

        for entity in natsorted(os.listdir((path_input_aux + dirs + '/' + dir))):
            sub_dir_output_name = dir_output_name_case_exp + '/' + entity + '/'
            if not os.path.exists(sub_dir_output_name):

                J = to_normalize(np.loadtxt(path_input_aux + dirs + '/' + dir + '/' + entity + '/' + default_Jij_name, delimiter=','))

                temperature_parameters = (0.0, J.shape[-1], no_temperature)  # Temperature parameters (initial tempeture, final tempeture, number of steps)

                if not os.path.exists(dir_output_name_case_exp + '/' + 'parameters.pkl'):
                    output = open(dir_output_name_case_exp + '/' + 'parameters.pkl', 'wb')
                    pickle.dump({'temperature_parameters': temperature_parameters, 'no_simulations': no_simulations,
                                 'thermalize_time': thermalize_time}, output)
                    output.close()

                start_time = time.time()
                simulated_fc, critical_temperature, E, M, S, H = generalized_ising(J,
                                                                                   temperature_parameters=temperature_parameters,
                                                                                   no_simulations=no_simulations,
                                                                                   thermalize_time=thermalize_time)
                print(time.time() - start_time)
                os.mkdir(sub_dir_output_name)
                to_save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, sub_dir_output_name)

                gc.collect()