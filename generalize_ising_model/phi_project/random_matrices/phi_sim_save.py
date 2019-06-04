def phi_sim_save(size, output_directory,count,temperature_parameters=(-1,5,50), no_simulations=500, thermalize_time=0.3):


    import numpy as np
    from generalize_ising_model.core import generalized_ising
    import time
    from generalize_ising_model.ising_utils import to_save_results, makedir, \
        to_generate_randon_graph, save_graph
    from generalize_ising_model.phi_project.utils import to_estimate_tpm_from_ising_model, to_calculate_mean_phi, \
        to_save_phi

    makedir(output_directory + '/phi/')
    makedir(output_directory + '/ising/')

    size = int(size)
    output_path_phi = output_directory + '/' + 'phi/' + str(count) + '/'
    output_path_ising = output_directory + '/' + 'ising/' + str(count) + '/'

    if makedir(output_path_ising) and makedir(output_path_phi):

        J = save_graph(output_path_phi + 'Jij_' + str(count) + '.csv',
                       to_generate_randon_graph(size, isolate=False, weighted=True))

        # Ising Parameters
        temperature_parameters = temperature_parameters  # Temperature parameters (initial tempeture, final tempeture, number of steps)
        no_simulations = no_simulations  # Number of simulation after thermalization
        thermalize_time = thermalize_time  #

        start_time = time.time()
        print('Fitting Generalized Ising model for a ', size, ' by', size, ' random matrix.')
        simulated_fc, critical_temperature, E, M, S, H, spin_mean = generalized_ising(J,
                                                                                      temperature_parameters=temperature_parameters,
                                                                                      no_simulations=no_simulations,
                                                                                      thermalize_time=thermalize_time,
                                                                                      temperature_distribution='log')

        to_save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, output_path_ising,
                        temperature_distribution='log')
        print('It took ', time.time() - start_time, 'seconds to fit the generalized ising model')

        print('Computing Phi for random ' + str(size) + ' by ' + str(size) + ' matrix')

        ts = np.logspace(temperature_parameters[0], np.log10(temperature_parameters[1]), temperature_parameters[2])

        phi_temperature, phi_sum, phi_sus = [], [], []
        cont = 0

        start_time = time.time()

        for t in ts:
            # print('Temperature: ' + str(t))
            tpm, fpm = to_estimate_tpm_from_ising_model(J, t)
            phi_, phiSum, phiSus = to_calculate_mean_phi(fpm, spin_mean[:, cont], t)
            phi_temperature.append(phi_)
            phi_sum.append(phiSum)
            phi_sus.append(phiSus)

            cont += 1

        to_save_phi(ts, phi_temperature, phi_sum, phi_sus, S, critical_temperature, size, output_path_phi)

        print('It takes ', time.time() - start_time, 'seconds to compute phi for a ', size, 'by', size,
              ' random matrix.')
    else:
        print(str(count) + ' is already done!')
