from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np

audio_path = "/home/brainlab/Desktop/Rudas/Data/Propofol/Taken-[AudioTrimmer.com].wav"
tr = 2
channel = 0
number_features_to_use = 21
number_feature_to_plot = 7
output_filename = 'audio_predictors.txt'

[Fs, x] = audioBasicIO.readAudioFile(audio_path)
F, f_names = audioFeatureExtraction.stFeatureExtraction(x[:, channel], Fs, tr*Fs, tr*Fs)

np.savetxt(output_filename, np.transpose(F[:number_features_to_use]), fmt='%10.6f', delimiter=' ', header=' '.join(f_names[:number_features_to_use]))

#for feature in range(number_feature_to_plot):
#    plt.subplot(number_feature_to_plot,1,feature+1); plt.plot(F[feature,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[feature])


#plt.show()


n = 150
average_signal = []
for i in range(n):
    start = Fs*(2*i)
    end = Fs*(2*i + 2)
    sub_signal = x[start:end, channel]
    print(sub_signal.shape)
    average_signal.append(np.mean(sub_signal))

np.savetxt('audio_average.txt', np.transpose(average_signal), fmt='%10.6f', delimiter=' ')

