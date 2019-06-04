import scipy.io
import numpy as np
from generalize_ising_model.core import generalized_ising
import time
import matplotlib.pyplot as plt
import collections
from generalize_ising_model.ising_utils import distance_wei, to_normalize, to_save_results, makedir
from generalize_ising_model.phi_project.utils import to_estimate_tpm_from_ising_model, to_calculate_mean_phi, to_save_phi

dir_output_name = '/home/user/Desktop/phiTest/'
Jij = scipy.io.loadmat(
    '/home/user/Desktop/phiTest/mean_struct_corr.mat')

rsn_index = {'AUD': [33, 34, 29, 30, 21]}#,
           #  'DMN': [9, 25, 15, 24, 7],
           #  'ECL': [7, 18, 17, 3, 19],
            # 'ECR': [67, 66, 56, 75, 52],
          #   'SAL': [66, 83, 51, 68, 75],
            # 'SEN': [16, 21, 22, 23, 33],
           #  'VIL': [59, 55, 64, 61, 77],
            # 'VIM': [4, 20, 12, 9, 73],
            # 'VIO': [20, 10, 12, 4, 6]}

d = collections.OrderedDict(sorted(rsn_index.items()))
Jij = Jij['meanJ_prob']

D, B = distance_wei(1. / Jij)

# Ising Parameters
temperature_parameters = (-1, 5, 500)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
no_simulations = 500  # Number of simulation after thermalization
thermalize_time = 0.3  #

makedir(dir_output_name)

for key, value in d.items():

    i_l = []
    j_l = []
    for v in value:
        for w in value:
            i_l.append(v - 1)
            j_l.append(w - 1)

    J = 1. / D[i_l, j_l]
    J[J == np.inf] = 0
    J[J == -np.inf] = 0
    J = J / np.max(J)
    J = np.reshape(J, (len(value), len(value)))

    J = to_normalize(J)

    start_time = time.time()
    print('Fitting Generalized ising model')
    simulated_fc, critical_temperature, E, M, S, H, spin_mean = generalized_ising(J,
                                                                       temperature_parameters=temperature_parameters,
                                                                       no_simulations=no_simulations,
                                                                       thermalize_time=thermalize_time,
                                                                       temperature_distribution = 'log')

    sub_dir_output_name = dir_output_name + '/' + key + '/'
    makedir(dir_output_name + '/' + key)
    to_save_results(temperature_parameters, J, E, M, H, S, simulated_fc, critical_temperature, sub_dir_output_name)
    print(time.time() - start_time)

    start_time = time.time()

    print('Computing Phi for: ' + key)
    #ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])
    ts = np.logspace(temperature_parameters[0], np.log10(temperature_parameters[1]), temperature_parameters[2])

    phi_temperature = []

    phi_sum = []
    phi_sus = []
    cont = 0

    start_time = time.time()

    for t in ts:
        #print('Temperature: ' + str(t))
        tpm, fpm = to_estimate_tpm_from_ising_model(J, t)
        phi, phiSum, phiSus = to_calculate_mean_phi(fpm, spin_mean[:, cont], t)
        phi_temperature.append(phi)
        phi_sum.append(phiSum)
        phi_sus.append(phiSus)

        cont += 1
    output_path = dir_output_name + '/' + 'phi/' + key + '/'
    makedir(dir_output_name + '/' + 'phi')
    to_save_phi(ts, phi_temperature, phi_sum,phi_sus, S, critical_temperature, key, output_path)
    print(time.time() - start_time)


