from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os

class SignalExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    out_file = File(mandatory=True)
    atlas_identifier = traits.String(mandatory=True)
    tr = traits.Float(mandatory=True)
    confounds_file = File(mandatory=True)
    plot = traits.Bool(default_value=False, mandatory = False)

class SignalExtractionOutputSpec(TraitedSpec):
    out_file = File(genfile=True)

class SignalExtraction(BaseInterface):
    input_spec = SignalExtractionInputSpec
    output_spec = SignalExtractionOutputSpec

    def _run_interface(self, runtime):

        from nilearn import datasets
        from nilearn.input_data import NiftiLabelsMasker
        import numpy as np

        dataset = datasets.fetch_atlas_harvard_oxford(self.inputs.atlas_identifier)
        atlas_filename = dataset.maps

        masker = NiftiLabelsMasker(labels_img=atlas_filename,
                                   standardize=True,
                                   detrend=True,
                                   low_pass=0.1,
                                   high_pass=0.01,
                                   t_r=self.inputs.tr,
                                   memory='nilearn_cache',
                                   verbose=5)

        time_series = masker.fit_transform(self.inputs.in_file, confounds=self.inputs.confounds_file)

        np.savetxt(self.inputs.out_file, time_series, fmt='%10.2f', delimiter=',')

        if self.inputs.plot:
            from nilearn import plotting
            from nilearn.connectome import ConnectivityMeasure
            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]

            # Mask the main diagonal for visualization:
            np.fill_diagonal(correlation_matrix, 0)
            plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=dataset.labels[1:],
                                 vmax=0.8, vmin=-0.8, reorder=True)

            plotting.show()

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath(self.inputs.out_file)}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None