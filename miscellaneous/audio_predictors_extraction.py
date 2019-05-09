from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#audio_path = "/home/brainlab/Desktop/Rudas/Data/Propofol/Taken-[AudioTrimmer.com].wav"
audio_path =  "/home/brainlab/Desktop/Rudas/Data/Propofol/Taken-[AudioTrimmer.com].wav"

tr = 2
#number_features_to_use = 10
number_feature_to_plot = 20
output_filename = 'audio_predictors.txt'

[Fs, x] = audioBasicIO.readAudioFile(audio_path)
x = audioBasicIO.stereo2mono(x)

F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, Fs/40, Fs/80)
print(F.shape)

np.savetxt(output_filename, np.transpose(F), fmt='%10.6f', delimiter=' ', header=' '.join(f_names))

plt.plot(x)
plt.show()

from nilearn.signal import clean
#F = clean(signals=F,
#          detrend=False,
#          standardize=True,
#          ensure_finite=False)

#for feature in range(2):
#    plt.subplot(2,1,feature+1);
#    plt.plot(F[feature,:]);
    #plt.xlabel('Frame no');
    #plt.ylabel(f_names[feature])
#plt.show()



n = 150
average_signal = []
average_signal_ = []
signal_portion_average = []
signal_portion_mode = []

for i in range(n):
    start = Fs*(2*i)
    end = Fs*(2*i + 2)

    sub_signal = x[start:end]
    sub_signal_ = x[start:int(start + Fs/10)]


    average_signal.append(np.mean(np.abs(sub_signal)))
    signal_portion_average.append(np.mean(np.abs(sub_signal_)))
    #average_signal_mod.append(np.median(np.abs(sub_signal_)))
    #signal_portion_mode.append(stats.mode(np.abs(sub_signal_)))

np.savetxt('audio_average_complete_tr.txt', np.transpose(average_signal), fmt='%10.6f', delimiter=' ')
np.savetxt('audio_average_portion_tr.txt', np.transpose(signal_portion_average), fmt='%10.6f', delimiter=' ')
#np.savetxt('audio_mode_complete_tr.txt', np.transpose(signal_portion_mode), fmt='%10.6f', delimiter=' ')

