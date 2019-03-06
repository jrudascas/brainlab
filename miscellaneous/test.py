from nilearn import datasets, image
import numpy as np

niimg = datasets.load_mni152_template()
print(niimg.get_data().shape)
result = image.coord_transform(54, 10, 38, niimg.affine)
print(np.asarray(result)/1000)