from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class N4BiasInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    out_file = File(mandatory=True)


class N4BiasOutputSpec(TraitedSpec):
    out_file = File(genfile=True)


class N4Bias(BaseInterface):
    input_spec = N4BiasInputSpec
    output_spec = N4BiasOutputSpec

    def _run_interface(self, runtime):
        import SimpleITK as sitk

        inputImage = sitk.ReadImage(self.inputs.in_file)
        inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        # numberFittingLevels = 4
        # numberOfIterations = 4
        # corrector.SetMaximumNumberOfIterations([numberOfIterations] *numberFittingLevels)

        output = corrector.Execute(inputImage)
        sitk.WriteImage(output, self.inputs.out_file)
        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath(self.inputs.out_file)}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None