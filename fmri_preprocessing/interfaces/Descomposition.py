from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface, OutputMultiPath
import os


class DescompositionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    n_components = traits.Int(mandatory=True)
    low_pass = traits.Float(mandatory=True)
    high_pass = traits.Float(mandatory=True)
    tr = traits.Float(mandatory=True)
    confounds_file = File(exists=True, mandatory=True)


class DescompositionOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(genfile=True))
    plot_files = OutputMultiPath(File(genfile=True))
    time_series = File(genfile=True)

class Descomposition(BaseInterface):
    input_spec = DescompositionInputSpec
    output_spec = DescompositionOutputSpec

    def _run_interface(self, runtime):

        from nilearn.decomposition import CanICA, DictLearning
        from nilearn.plotting import plot_stat_map, plot_prob_atlas
        from nilearn.image import iter_img
        from nilearn.input_data import NiftiMapsMasker
        import numpy as np

        canica = CanICA(n_components=self.inputs.n_components,
                        smoothing_fwhm=None,
                        low_pass=self.inputs.low_pass,
                        high_pass=self.inputs.high_pass,
                        t_r=self.inputs.tr,
                        do_cca=True,
                        detrend=True,
                        standardize=True,
                        threshold='auto',
                        random_state=0,
                        n_jobs=2,
                        memory_level=2,
                        memory="nilearn_cache")

        canica.fit(self.inputs.in_file, confounds=self.inputs.confounds_file)
        # canica.fit(self.inputs.in_file)

        components_img = canica.components_img_
        components_img.to_filename('descomposition_canica.nii.gz')

        # plot_prob_atlas(components_img, title='All ICA components')


        masker = NiftiMapsMasker(maps_img=components_img, standardize=True, memory='nilearn_cache', verbose=5)
        time_series_ica = masker.fit_transform(self.inputs.in_file)

        np.savetxt('time_series_ica.csv', np.asarray(time_series_ica), delimiter=',')

        for i, cur_img in enumerate(iter_img(components_img)):
            display = plot_stat_map(cur_img, display_mode="ortho", colorbar=True)
            display.savefig('ica_ic_' + str(i + 1) + '.png')
            display.close()

        dict_learning = DictLearning(n_components=self.inputs.n_components,
                                     smoothing_fwhm=None,
                                     low_pass=self.inputs.low_pass,
                                     high_pass=self.inputs.high_pass,
                                     t_r=self.inputs.tr,
                                     detrend=True,
                                     standardize=True,
                                     random_state=0,
                                     n_jobs=-2,
                                     memory_level=2,
                                     memory="nilearn_cache",
                                     mask_strategy = 'template')

        dict_learning.fit(self.inputs.in_file, confounds=self.inputs.confounds_file)

        components_img = dict_learning.components_img_
        components_img.to_filename('descomposition_dict.nii.gz')

        for i, cur_img in enumerate(iter_img(components_img)):
            display = plot_stat_map(cur_img, display_mode="ortho", colorbar=True)
            display.savefig('dic_ic_' + str(i + 1) + '.png')
            display.close()

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = []
        outputs['out_file'].append(os.path.abspath('descomposition_canica.nii.gz'))
        outputs['out_file'].append(os.path.abspath('descomposition_dict.nii.gz'))

        outputs['plot_files'] = []
        for i in range(self.inputs.n_components):
            outputs['plot_files'].append(os.path.abspath('ica_ic_' + str(i + 1) + '.png'))
            outputs['plot_files'].append(os.path.abspath('dic_ic_' + str(i + 1) + '.png'))

        outputs['time_series'] = os.path.abspath('time_series_ica.csv')

        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None
