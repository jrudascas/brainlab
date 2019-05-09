import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from generalize_ising_model.ising_utils import corrfun, dim, find_nearest

np.set_printoptions(precision=4)

mat = scipy.io.loadmat('/home/brainlab/Desktop/Rudas/Scripts/ising/dimentionality/wd1/full.mat')

corr = mat['Corr_all']
J = mat['J_count_MS_Det']
tc_subs = np.squeeze(mat['tc_subs'])
temp = np.squeeze(mat['temp'])

print('Starting')
corr_fun, r_all = corrfun(corr, J)

print(corr_fun.shape)

sub = corr_fun.shape[0]

d = []
for i in range(sub):
    print('Sub ' + str(i + 1))
    corr_func = corr_fun[i, :, :]
    idx_ct = find_nearest(temp, tc_subs[i])
    d.append(dim(corr_func, r_all[i, :], idx_ct))

print(d)
plt.scatter(np.linspace(0, len(d), num=len(d)), d)
plt.show()

print('Hola Mundo')
