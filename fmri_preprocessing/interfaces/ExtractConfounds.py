from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class ExtractConfoundsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    out_file = File(mandatory=True)
    delimiter = traits.String(mandatory=False)
    list_mask = traits.List(mandatory=True)
    file_concat = File(mandatory=False)


class ExtractConfoundsOutputSpec(TraitedSpec):
    out_file = File(genfile=True)


class ExtractConfounds(BaseInterface):
    input_spec = ExtractConfoundsInputSpec
    output_spec = ExtractConfoundsOutputSpec

    def _run_interface(self, runtime):

        import nibabel as nib
        import numpy as np

        delimiter = None
        if isinstance(self.inputs.delimiter, str):
            delimiter = self.inputs.delimiter

        threshold = 0.8
        img = nib.load(self.inputs.in_file)
        data = img.get_data()
        confounds = []

        if not isinstance(self.inputs.list_mask, list):
            aux = self.inputs.list_mask
            list_mask = [aux]

        for mask in self.inputs.list_mask:
            mask_data = np.nan_to_num(nib.load(mask).get_data())

            confounds.append(np.mean(data[mask_data > threshold, :], axis=0))

        ev = np.transpose(np.array(confounds))

        if self.inputs.file_concat is not None:
            extra_confounds = np.loadtxt(self.inputs.file_concat, delimiter=delimiter)
            results = np.concatenate((ev, extra_confounds), axis=1)
        else:
            results = ev

        np.savetxt(self.inputs.out_file, results, fmt='%10.5f', delimiter=',')

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath(self.inputs.out_file)}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None