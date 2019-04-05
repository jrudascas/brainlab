import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.data import default_sphere, fetch_cenir_multib
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.streamline import save_trk
from dipy.direction import ProbabilisticDirectionGetter
from dipy.direction import peaks_from_model
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.utils import connectivity_matrix
from dipy.tracking.eudx import EuDX
from dipy.reconst import shm
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines
from dipy.data import read_stanford_labels, fetch_stanford_t1, read_stanford_t1


def save(strem_generator, affine_, shape_, path_output, h):
    stream = [s for s in strem_generator if s.shape[0] > h]

    save_trk(path_output, list(stream), affine_, shape_)

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
gtab = gradient_table(bval_path, bvec_path, b0_threshold=50)

#t1 = read_stanford_t1()
#t1_data = t1.get_data()
#mask = t1_data
#mask[mask != 0] = 1
#mask = mask.astype(bool)

labels = nib.load('/home/brainlab/Desktop/Rudas/Data/Parcellation/atlas_NMI_2mm.nii').get_data().astype(int)

csa_model = CsaOdfModel(gtab, sh_order=sh_order)
#csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=sh_order)
#csd_fit = csd_model.fit(data, mask=mask)

seeds = utils.seeds_from_mask(mask, density=density, affine=affine)


csa_peaks = peaks_from_model(model=csa_model,
                             data=data,
                             sphere=default_sphere,
                             relative_peak_threshold=relative_peak_threshold,
                             min_separation_angle=min_separation_angle,
                             mask=mask)

#csd_peaks = peaks_from_model(model=csd_model,
#                             data=data,
#                             sphere=default_sphere,
#                             relative_peak_threshold=relative_peak_threshold,
#                             min_separation_angle=min_separation_angle,
#                             mask=mask)

streamline_eudx = EuDX(csa_peaks.peak_values, csa_peaks.peak_indices,
                            odf_vertices=default_sphere.vertices,
                            a_low=threshold_tissue_classifier, step_sz=step_size, seeds=seeds)

save(streamline_eudx, streamline_eudx.affine, mask.shape, '1.trk', lenght_threshold)

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csa_peaks.shm_coeff, max_angle=max_angle, sphere=default_sphere)
tensor_model = dti.TensorModel(gtab)
dti_fit = tensor_model.fit(data, mask=mask)
FA = fractional_anisotropy(dti_fit.evals)
classifier = ThresholdTissueClassifier(FA, threshold_tissue_classifier)
streamlines_dmdg = LocalTracking(detmax_dg, classifier, seeds, affine, step_size=step_size)

save(streamlines_dmdg, streamline_eudx.affine, mask.shape, '1.trk', lenght_threshold)

classifier = ThresholdTissueClassifier(csa_peaks.gfa, threshold_tissue_classifier)
prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csa_peaks.shm_coeff, max_angle=max_angle, sphere=default_sphere)
streamlines_pdg = LocalTracking(prob_dg, classifier, seeds, affine, step_size=step_size)

save(streamlines_pdg, streamline_eudx.affine, mask.shape, '1.trk', lenght_threshold)

#M, grouping = connectivity_matrix(streamlines, labels, affine=s_affine, symmetric=True, return_mapping=True, mapping_as_streamlines=True)