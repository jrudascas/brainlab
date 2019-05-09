from nilearn.image import clean_img
import nibabel as nib
import numpy as np
from scipy.stats import pearsonr
from nilearn import plotting
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind, f_oneway, mannwhitneyu
from nilearn.signal import clean

img = nib.load('/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_02_02CB/wt1_n4bias.nii')
affine = img.affine
data = abs(np.nan_to_num(img.get_data()))
data[np.where(data >= 0.1)] = 1
data[np.where(data <= 0.1)] = 0

mask = nib.Nifti1Image(data.astype(np.float32), affine)
nib.save(mask, '/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_02_02CB/mask.nii')

image_cleaned = clean_img('/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_02_02CB/swfmri_art_removed.nii',
                          sessions=None,
                          detrend=True,
                          standardize=True,
                          high_pass=0.01,
                          t_r=2,
                          confounds='/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_02_02CB/ev_without_gs.csv',
                          ensure_finite=True,
                          mask_img='/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_02_02CB/mask.nii')

nib.save(image_cleaned, '/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_02_02CB/cleaned.nii')

data_cleaned = image_cleaned.get_data()

predictors = np.loadtxt('/home/brainlab/Desktop/Rudas/Scripts/fmri/miscellaneous/audio_predictors.txt')

n = 150
window = 80
new_predictor = np.zeros((150, predictors.shape[1]))
print(predictors.shape)

predictor_cleaned = clean(signals=predictors,
                          detrend=True,
                          standardize=True,
                          #high_pass=0.01,
                          #t_r=window,
                          ensure_finite=True)

for index in range(predictor_cleaned.shape[1]):
    for i in range(n):
        start = window * (2 * i)
        end = window * (2 * i + 2)

        new_predictor[i,index] = np.mean(predictor_cleaned[start:end, index])

data_mask = mask.get_data().astype(bool)

for pred in range(new_predictor.shape[1]):
    stat_map = np.zeros(data_cleaned.shape[:3])
    print(pred)
    for x in range(data_cleaned.shape[0]):
        for y in range(data_cleaned.shape[1]):
            for z in range(data_cleaned.shape[2]):
                if data_mask[x,y,z]:
                    t, p = pearsonr(data_cleaned[x,y,z,:], new_predictor[:, pred])
                    stat_map[x,y,z] = stats.norm.isf(p)

    display = plotting.plot_stat_map(nib.Nifti1Image(stat_map.astype(np.float32), affine), threshold=4)
    display.savefig('A' + str(pred) + '.png')
    display.close()
