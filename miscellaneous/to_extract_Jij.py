import scipy.io
import numpy as np
import os

mat = scipy.io.loadmat('/home/brainlab/Desktop/Rudas/Scripts/ising/dimentionality/wd1/toJorge.mat')
path_output = '/home/brainlab/Desktop/Rudas/Data/Ising/HCP/'
Js_ = corr = mat['J_count_MS_Det']

print(Js_.shape)

for i in range(Js_.shape[-1]):
    Jij_sub = Js_[:,:, i]
    os.mkdir(path_output + 'sub_' + str(i + 1))
    np.savetxt(path_output + 'sub_' + str(i + 1) + '/' + 'J_ij.csv', Jij_sub, delimiter=',', fmt='%.2f')