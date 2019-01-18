def SignalExtraction(in_file, out_path, confounds_file, atlas_identifier, tr, plot=False):
    from nilearn import datasets
    from nilearn.input_data import NiftiLabelsMasker
    import numpy as np
    from os.path import join as opj

    dataset = datasets.fetch_atlas_harvard_oxford(atlas_identifier)
    atlas_filename = dataset.maps

    masker = NiftiLabelsMasker(labels_img=atlas_filename,
                               standardize=True,
                               detrend=False,
                               high_pass=0.1,
                               t_r=tr,
                               memory='nilearn_cache',
                               verbose=5)

    time_series = masker.fit_transform(in_file, confounds=confounds_file)
    print('Time serie shape')
    print(time_series.shape)
    np.savetxt(opj(out_path, 'time_series.csv'), time_series, fmt='%10.5f', delimiter=',')

    if plot:
        from nilearn import plotting
        from nilearn.connectome import ConnectivityMeasure
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]

        # Mask the main diagonal for visualization:
        np.fill_diagonal(correlation_matrix, 0)
        plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=dataset.labels[1:],
                             vmax=0.8, vmin=-0.8, reorder=True)

        plotting.show()

    return opj(out_path, 'time_series.csv')


def ArtifacRemotion(in_file, outlier_files):
    import nibabel as nib
    import numpy as np

    img = nib.load(in_file)
    affine = img.affine
    data = img.get_data()

    artifact_volumens = np.loadtxt(outlier_files)

    print('List of artifact volumens: ' + str(artifact_volumens))
    print(in_file)
    return in_file

def ExtractConfounds(in_file, out_path, out_file_name, delimiter, list_mask, file_concat=None):

    import nibabel as nib
    import numpy as np
    from os.path import join as opj

    threshold = 0.6
    img = nib.load(in_file)
    data = img.get_data()
    confounds = []
    print('/////////////////////////////////')
    if not isinstance(list_mask, list):
        aux = list_mask
        list_mask = [aux]
    print('###########################')
    for mask in list_mask:
        mask_data = nib.load(mask).get_data()

        confounds.append(np.mean(data[mask_data > threshold, :], axis=0))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    ev = np.transpose(np.array(confounds))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(file_concat)
    if file_concat is not None:
        extra_confounds = np.loadtxt(file_concat, delimiter=delimiter)
        print('````````````````````````````````````')
        results = np.concatenate((ev, extra_confounds), axis=1)
    else:
        results = ev
    print('^^^^^^^^^^^^^^^^^')
    print(out_path)
    print(out_file_name)
    print('Saving' + opj(out_path, out_file_name))

    np.savetxt(opj(out_path, out_file_name), results, fmt='%10.5f',delimiter=',')

    return opj(out_path, out_file_name)

from os.path import join as opj
import os
from nipype.interfaces.fsl import (BET, ExtractROI, FAST, FLIRT, ImageMaths,
                                   MCFLIRT, SliceTimer, Threshold)
import nipype.interfaces.spm as spm
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.rapidart import ArtifactDetect
from nipype import Workflow, Node, Function

matlab_cmd = '/home/brainlab/Desktop/Rudas/Tools/spm12_r7487/spm12/run_spm12.sh /home/brainlab/Desktop/Rudas/Tools/MCR/v713/ script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

print('SPM version: ' + str(spm.SPMCommand().version))

base_dir = '/home/brainlab/Desktop/Rudas/Data/Other'
experiment_dir = opj(base_dir, 'output/')
output_dir = 'datasink'
working_dir = 'workingdir'


subject_list = ['1']       # list of subject identifiers
fwhm = [4]                      # Smoothing widths to apply (Gaussian kernel size)
TR = 2                          #Repetition time
init_volume = 0                 #Firts volumen identification which will use in the pipeline
iso_size = 2                    # Isometric resample of functional images to voxel size (in mm)


# ExtractROI - skip dummy scans
extract = Node(ExtractROI(t_min=init_volume, t_size=-1, output_type='NIFTI'), name="extract")

# MCFLIRT - motion correction
mcflirt = Node(MCFLIRT(mean_vol=True, save_plots=True, output_type='NIFTI'), name="motion_correction")

# SliceTimer - correct for slice wise acquisition
slicetimer = Node(SliceTimer(index_dir=False, interleaved=True, output_type='NIFTI', time_repetition=TR), name="slice_timing_correction")

# Smooth - image smoothing
smooth = Node(spm.Smooth(), name="smooth")
smooth.iterables = ("fwhm", fwhm)

# Artifact Detection - determines outliers in functional images
art = Node(ArtifactDetect(norm_threshold=2,
                          zintensity_threshold=3,
                          mask_type='spm_global',
                          parameter_source='FSL', use_differences=[True, False], plot_type='svg'),
           name="artifact_detection")

extract_confounds_ws_csf = Node(Function(input_names=['in_file', 'out_path', 'out_file_name', 'delimiter', 'list_mask', 'file_concat'],
                        output_names=['out_file'],
                        function=ExtractConfounds),
               name='extract_confounds_ws_csf')

extract_confounds_ws_csf.inputs.out_path = base_dir
extract_confounds_ws_csf.inputs.out_file_name = 'ev_without_gs.csv'
extract_confounds_ws_csf.inputs.delimiter = None

extract_confounds_gs = Node(Function(input_names=['in_file', 'out_path', 'out_file_name', 'delimiter', 'list_mask', 'file_concat'],
                        output_names=['out_file'],
                        function=ExtractConfounds),
               name='extract_confounds_global_signal')

extract_confounds_gs.inputs.out_path = base_dir
extract_confounds_gs.inputs.out_file_name = 'ev_with_gs.csv'
extract_confounds_gs.inputs.delimiter = ','

signal_extraction = Node(Function(input_names=['in_file', 'out_path', 'confounds_file', 'atlas_identifier', 'tr', 'plot'],
                        output_names=['out_file'],
                        function=SignalExtraction),
               name='signal_extraction')

signal_extraction.inputs.out_path = base_dir
signal_extraction.inputs.atlas_identifier = 'sub-maxprob-thr0-2mm'
#signal_extraction.inputs.atlas_identifier = 'cort-maxprob-thr25-2mm'

signal_extraction.inputs.tr = TR
signal_extraction.inputs.plot = True

art_remotion = Node(Function(input_names=['in_file', 'outlier_files'],
                        output_names=['out_file'],
                        function=ArtifacRemotion),
               name='artifact_remotion')

# BET - Skullstrip anatomical anf funtional images
bet_t1 = Node(BET(frac=0.75,
                    robust=True,
                    mask=True,
                    output_type='NIFTI_GZ'),
                name="bet_t1")

bet_fmri = Node(BET(frac=0.6,
                    functional = True,
                    output_type='NIFTI_GZ'),
                name="bet_fmri")

# FAST - Image Segmentation
segmentation = Node(FAST(output_type='NIFTI'),
                name="segmentation")


# Select WM segmentation file from segmentation output
def get_latest(files):
    return files[-1]

def get_wm_csf(files):
    print(len(files))
    wm_csf_list = [files[0], files[2]]
    return wm_csf_list

# Threshold - Threshold WM probability image
threshold = Node(Threshold(thresh=0.5,
                           args='-bin',
                           output_type='NIFTI_GZ'),
                name="wm_mask_threshold")

# FLIRT - pre-alignment of functional images to anatomical images
coreg_pre = Node(FLIRT(dof=6, output_type='NIFTI_GZ'), name="linear_warp_estimation")

# FLIRT - coregistration of functional images to anatomical images with BBR
coreg_bbr = Node(FLIRT(dof=6,
                       cost='bbr',
                       schedule=opj(os.getenv('FSLDIR'),
                                    'etc/flirtsch/bbr.sch'),
                       output_type='NIFTI_GZ'),
                 name="nonlinear_warp_estimation")

# Apply coregistration warp to functional images
applywarp = Node(FLIRT(interp='spline',
                       apply_isoxfm=iso_size,
                       output_type='NIFTI'),
                 name="registration_fmri")

# Apply coregistration warp to mean file
applywarp_mean = Node(FLIRT(interp='spline',
                            apply_isoxfm=iso_size,
                            output_type='NIFTI_GZ'),
                 name="registration_mean_fmri")

# Create a coregistration workflow
coregwf = Workflow(name='coreg_fmri_to_t1')
coregwf.base_dir = opj(experiment_dir, working_dir)

# Connect all components of the coregistration workflow
coregwf.connect([(bet_t1, segmentation, [('out_file', 'in_files')]),
                 (segmentation, threshold, [(('partial_volume_files', get_latest),'in_file')]),
                 (bet_t1, coreg_pre, [('out_file', 'reference')]),
                 (threshold, coreg_bbr, [('out_file', 'wm_seg')]),
                 (coreg_pre, coreg_bbr, [('out_matrix_file', 'in_matrix_file')]),
                 (coreg_bbr, applywarp, [('out_matrix_file', 'in_matrix_file')]),
                 (bet_t1, applywarp, [('out_file', 'reference')]),
                 (coreg_bbr, applywarp_mean, [('out_matrix_file', 'in_matrix_file')]),
                 (bet_t1, applywarp_mean, [('out_file', 'reference')]),
                 ])

# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles - to grab the data (alternativ to DataGrabber)
anat_file = opj('sub-{subject_id}', 't1.nii')
func_file = opj('sub-{subject_id}', 'fmri.nii')

templates = {'anat': anat_file, 'func': func_file}

selectfiles = Node(SelectFiles(templates, base_directory=base_dir), name="selectfiles")

# Datasink - creates output folder for important outputs
datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir), name="datasink")

## Use the following DataSink output substitutions
substitutions = [('_subject_id_', 'sub-'),
                 ('_fwhm_', 'fwhm-'),
                 ('_roi', ''),
                 ('_mcf', ''),
                 ('_st', ''),
                 ('_flirt', ''),
                 ('.nii_mean_reg', '_mean'),
                 ('.nii.par', '.par'),
                 ]
subjFolders = [('fwhm-%s/' % f, 'fwhm-%s_' % f) for f in fwhm]

substitutions.extend(subjFolders)
datasink.inputs.substitutions = substitutions

# Create a preprocessing workflow
preproc = Workflow(name='preproc')
preproc.base_dir = opj(experiment_dir, working_dir)

from nipype.interfaces.spm import Normalize12, Normalize
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip

template = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'

# Normalize - normalizes functional and structural images to the MNI template
normalize_fmri = Node(Normalize12(jobtype='estwrite',
                                  tpm=template,
                                  write_voxel_sizes=[2, 2, 2],
                                  write_bounding_box = [[-90, -126, -72], [90, 90, 108]]),
                 name="normalize_fmri")

gunzip = Node(Gunzip(), name="gunzip")
#gunzip2 = Node(Gunzip(), name="gunzip2")

normalize_t1 = Node(Normalize12(jobtype='estwrite',
                                tpm=template,
                                write_voxel_sizes=[2, 2, 2],
                                write_bounding_box = [[-90, -126, -72], [90, 90, 108]]),
                 name="normalize_t1")

normalize_masks = Node(Normalize12(jobtype='estwrite',
                                tpm=template,
                                write_voxel_sizes=[2, 2, 2],
                                write_bounding_box = [[-90, -126, -72], [90, 90, 108]]),
                    name="normalize_masks")

# Connect all components of the preprocessing workflow
preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                 (selectfiles, extract, [('func', 'in_file')]),
                 (extract, bet_fmri, [('roi_file', 'in_file')]),

                 (bet_fmri, mcflirt, [('out_file', 'in_file')]),


                 (mcflirt, slicetimer, [('out_file', 'in_file')]),
                 (selectfiles, coregwf, [('anat', 'bet_t1.in_file'),
                                         ('anat', 'nonlinear_warp_estimation.reference')]),
                 (mcflirt, coregwf, [('mean_img', 'linear_warp_estimation.in_file'),
                                     ('mean_img', 'nonlinear_warp_estimation.in_file'),
                                     ('mean_img', 'registration_mean_fmri.in_file')]),
                 (slicetimer, coregwf, [('slice_time_corrected_file', 'registration_fmri.in_file')]),

                 #(coregwf, smooth, [('applywarp.out_file', 'in_files')]),

                 (coregwf, art, [('registration_fmri.out_file', 'realigned_files')]),
                 (mcflirt, art, [('par_file', 'realignment_parameters')]),
                 #(bet_fmri, art, [('mask_file', 'mask_file')]),

                 (art, art_remotion, [('outlier_files', 'outlier_files')]),
                 (coregwf, art_remotion, [('registration_fmri.out_file', 'in_file')]),

                 (coregwf, gunzip, [('bet_t1.out_file', 'in_file')]),

                 #(coregwf, normalize_fmri, [('registration_fmri.out_file', 'image_to_align')]),
                 (gunzip, normalize_fmri, [('out_file', 'image_to_align')]),
                 (art_remotion, normalize_fmri, [('out_file', 'apply_to_files')]),

                 (gunzip, normalize_t1, [('out_file', 'image_to_align')]),
                 (gunzip, normalize_t1, [('out_file', 'apply_to_files')]),

                 (gunzip, normalize_masks, [('out_file', 'image_to_align')]),
                 (coregwf, normalize_masks, [(('segmentation.partial_volume_files', get_wm_csf), 'apply_to_files')]),

                 (normalize_fmri, smooth, [('normalized_files', 'in_files')]),

                 (smooth, extract_confounds_ws_csf, [('smoothed_files', 'in_file')]),
                 (normalize_masks, extract_confounds_ws_csf, [('normalized_files', 'list_mask')]),
                 (mcflirt, extract_confounds_ws_csf, [('par_file', 'file_concat')]),

                 (smooth, extract_confounds_gs, [('smoothed_files', 'in_file')]),
                 (normalize_t1, extract_confounds_gs, [('normalized_files', 'list_mask')]),
                 (extract_confounds_ws_csf, extract_confounds_gs, [('out_file', 'file_concat')]),

                 (smooth, signal_extraction, [('smoothed_files', 'in_file')]),
                 (extract_confounds_gs, signal_extraction, [('out_file', 'confounds_file')]),

                 (normalize_fmri, datasink, [('normalized_files', 'norm_spm.@files'),
                                             ('normalized_image', 'norm_spm.@image'),
                                             ]),
                 (mcflirt, datasink, [('par_file', 'preproc.@par')]),
                 (smooth, datasink, [('smoothed_files', 'preproc.@smooth')]),
                 (coregwf, datasink, [('nonlinear_warp_estimation.out_matrix_file', 'preproc.@mat_file'),
                                      ('registration_mean_fmri.out_file', 'preproc.@mean'),
                                      ('bet_t1.out_file', 'preproc.@brain')]),
                 (art, datasink, [('outlier_files', 'preproc.@outlier_files'),
                                  ('plot_files', 'preproc.@plot_files')]),
                 ])

preproc.write_graph(graph2use='colored', format='png', simple_form=True)
preproc.run()