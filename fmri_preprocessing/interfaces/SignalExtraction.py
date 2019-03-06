from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os

class SignalExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    time_series_out_file = File(mandatory=True)
    correlation_matrix_out_file = File(genfile=True)
    fmri_cleaned_out_file = File(genfile=True)
    atlas_identifier = traits.String(mandatory=True)
    tr = traits.Float(mandatory=True)
    confounds_file = File(mandatory=True)
    plot = traits.Bool(default_value=False, mandatory = False)

class SignalExtractionOutputSpec(TraitedSpec):
    time_series_out_file = File(genfile=True)
    correlation_matrix_out_file = File(genfile=True)
    fmri_cleaned_out_file = File(genfile=True)


class SignalExtraction(BaseInterface):
    input_spec = SignalExtractionInputSpec
    output_spec = SignalExtractionOutputSpec

    def _run_interface(self, runtime):

        from nilearn import datasets
        from nilearn.input_data import NiftiLabelsMasker
        import numpy as np

        #dataset = datasets.fetch_atlas_harvard_oxford(self.inputs.atlas_identifier)
        #atlas_filename = dataset.maps

        masker = NiftiLabelsMasker(labels_img='/home/brainlab/Desktop/Rudas/Data/Parcellation/atlas_NMI_2mm.nii',
                                   standardize=True,
                                   detrend=True,
                                   low_pass=0.1,
                                   high_pass=0.01,
                                   t_r=self.inputs.tr,
                                   memory='nilearn_cache',
                                   verbose=5)

        file_labels = open('/home/brainlab/Desktop/Rudas/Data/Parcellation/AAL from Freesourfer/fs_default.txt', 'r')
        labels = []
        for line in file_labels.readlines():
            labels.append(line)
        file_labels.close()

        time_series = masker.fit_transform(self.inputs.in_file, confounds=self.inputs.confounds_file)

        np.savetxt(self.inputs.time_series_out_file, time_series, fmt='%10.2f', delimiter=',')

        if self.inputs.plot:
            from nilearn import plotting
            from nilearn.connectome import ConnectivityMeasure
            import matplotlib
            import matplotlib.pyplot as plt
            fig, ax = matplotlib.pyplot.subplots()

            font = {'family': 'normal',
                    'size': 5}

            matplotlib.rc('font', **font)

            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]

            # Mask the main diagonal for visualization:
            np.fill_diagonal(correlation_matrix, 0)
            plotting.plot_matrix(correlation_matrix, figure=fig, labels=labels, vmax=0.8, vmin=-0.8, reorder=True)

            fig.savefig(self.inputs.correlation_matrix_out_file, dpi=1200)

        return runtime

    def _list_outputs(self):
        return {'time_series_out_file': os.path.abspath(self.inputs.time_series_out_file),
                'correlation_matrix_out_file':os.path.abspath(self.inputs.correlation_matrix_out_file)}

    def _gen_filename(self, name):
        if name == 'time_series_out_file':
            return os.path.abspath(self.inputs.time_series_out_file)
        if name == 'correlation_matrix_out_file':
            return os.path.abspath(self.inputs.correlation_matrix_out_file)
        return None