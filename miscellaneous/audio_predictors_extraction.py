from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np

#[Fs, x] = audioBasicIO.readAudioFile("/home/brainlab/Downloads/couchplayin.wav")
[Fs, x] = audioBasicIO.readAudioFile("/home/brainlab/Downloads/Taken-[AudioTrimmer.com].wav")

tr = 2

F, f_names = audioFeatureExtraction.stFeatureExtraction(x[:,0], Fs, tr*Fs, tr*Fs)

np.savetxt('audio_predictors.txt', np.transpose(F[:21]), fmt='%10.6f', delimiter=',')

number_feature = 7
for feature in range(number_feature):
    plt.subplot(number_feature,1,feature+1); plt.plot(F[feature,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[feature])


plt.show()