from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, BaseInterface
import os

class ArtifacRemotionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    out_file = File(mandatory=True)
    outlier_files = File(mandatory=True)

class ArtifacRemotionOutputSpec(TraitedSpec):
    out_file = File(genfile=True)

class ArtifacRemotion(BaseInterface):
    input_spec = ArtifacRemotionInputSpec
    output_spec = ArtifacRemotionOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        img = nib.load(self.inputs.in_file)
        affine = img.affine
        data = img.get_data()


        nib.save(nib.Nifti1Image(data, affine), self.inputs.out_file)
        artifact_volumens = np.loadtxt(self.inputs.outlier_files)

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath(self.inputs.out_file)}