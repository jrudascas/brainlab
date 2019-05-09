import numpy as np
import matplotlib.pyplot as plt

ts = np.logspace(-1, np.log10(4), 20)

print(ts)

plt.scatter(ts, ts, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Susceptibility", fontsize=20)
#plt.xticks(ts)
plt.axis('tight')

plt.show()
