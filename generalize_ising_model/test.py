import numpy as np
from generalize_ising_model.core import generalized_ising
import matplotlib.pyplot as plt
import time

temperature_parameters = (0.05, 10, 50)
no_simulations = 250
thermalize_time = 0.3
N = 10
J = np.random.rand(N, N)
print(''.join('*' * temperature_parameters[2]))

start_time = time.time()

simulated_fc, critical_temperature, E, M, S, H = generalized_ising(J,
                                                                   temperature_parameters=temperature_parameters,
                                                                   no_simulations=no_simulations,
                                                                   thermalize_time=thermalize_time)

print(time.time() - start_time)
ts = np.linspace(temperature_parameters[0], temperature_parameters[1], num=temperature_parameters[2])
f = plt.figure(figsize=(18, 10))  # plot the calculated values

f.add_subplot(2, 2, 1)
plt.scatter(ts, E, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Energy ", fontsize=20)
plt.axis('tight')

f.add_subplot(2, 2, 2)
plt.scatter(ts, abs(M), s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Magnetization ", fontsize=20)
plt.axis('tight')

f.add_subplot(2, 2, 3)
plt.scatter(ts, H, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Specific Heat", fontsize=20)
plt.axis('tight')

f.add_subplot(2, 2, 4)
plt.scatter(ts, S, s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Susceptibility", fontsize=20)
plt.axis('tight')

plt.show()
