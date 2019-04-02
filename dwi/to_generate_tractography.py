import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.streamline import save_trk
from dipy.direction import ProbabilisticDirectionGetter
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.shm import CsaOdfModel

dwi_path = '/home/brainlab/Desktop/Rudas/Data/pubu/diffu/data.nii.gz'
bvec_path = '/home/brainlab/Desktop/Rudas/Data/pubu/diffu/bvecs'
bval_path = '/home/brainlab/Desktop/Rudas/Data/pubu/diffu/bvals'
mask_path = '/home/brainlab/Desktop/Rudas/Data/pubu/nodif_brain_mask.nii.gz'
struct_path = '/home/brainlab/Desktop/Rudas/Data/pubu/struct/T1w_acpc_dc_restore_1.25.nii.gz'

step_size = 3.
max_angle = 30
density = 1
lenght_threshold = 30
sh_order = 6
min_separation_angle = 30
relative_peak_threshold = .5
threshold_tissue_classifier = .2

dwi_img = nib.load(dwi_path)
data = dwi_img.get_data()
affine = dwi_img.affine
mask = nib.load(mask_path).get_data().astype(bool)

struct_img = nib.load(struct_path)
gtab = gradient_table(bval_path, bvec_path)

print('0')
csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=sh_order)
csd_fit = csd_model.fit(data, mask=mask)

tensor_model = dti.TensorModel(gtab)
dti_fit = tensor_model.fit(data, mask=mask)

FA = fractional_anisotropy(dti_fit.evals)
classifier = ThresholdTissueClassifier(FA, threshold_tissue_classifier)

print('1')
detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff, max_angle=max_angle, sphere=default_sphere)
seeds = utils.seeds_from_mask(mask, density=density, affine=affine)
streamlines = LocalTracking(detmax_dg, classifier, seeds, affine, step_size=step_size)

streamlines = [s for s in streamlines if s.shape[0] > lenght_threshold]
streamlines = list(streamlines)
save_trk("deterministic_maximum_shm_coeff.trk", streamlines, struct_img.affine, struct_img.shape)

print('2')
csa_model = CsaOdfModel(gtab, sh_order=sh_order)
csa_peaks = peaks_from_model(csa_model, data, default_sphere, sh_order=sh_order,
                             relative_peak_threshold=relative_peak_threshold,
                             min_separation_angle=min_separation_angle, mask=mask, parallel=True)

classifier = ThresholdTissueClassifier(csa_peaks.gfa, threshold_tissue_classifier)
print('3')

seeds = utils.seeds_from_mask(mask, density=density, affine=affine)
streamlines = LocalTracking(csa_peaks, classifier, seeds, affine, step_size=step_size)
streamlines = [s for s in streamlines if s.shape[0] > lenght_threshold]

streamlines = list(streamlines)
save_trk('tractography_CsaOdf.trk', streamlines, struct_img.affine, struct_img.shape)
print('4')

print('5')
prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff, max_angle=max_angle, sphere=default_sphere)
streamlines_generator = LocalTracking(prob_dg, classifier, seeds, affine, step_size=step_size)

save_trk("probabilistic_shm_coeff.trk", streamlines_generator, affine, mask.shape)

print('5')
peaks = peaks_from_model(csd_model, data, default_sphere, relative_peak_threshold=relative_peak_threshold, min_separation_angle=min_separation_angle, mask=mask, return_sh=True, parallel=True)
prob_dg = ProbabilisticDirectionGetter.from_shcoeff(peaks.shm_coeff, max_angle=max_angle, sphere=default_sphere)
streamlines_generator = LocalTracking(prob_dg, classifier, seeds, affine, step_size=step_size)

save_trk("probabilistic_peaks_from_model.trk", streamlines_generator, affine, mask.shape)
print('6')