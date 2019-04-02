import scipy.io
from nibabel.affines import apply_affine
import nibabel as nib
import numpy as np
from numpy.linalg import inv
from nilearn import datasets, image
from nilearn import plotting


def distance(pt_1, pt_2):
    return np.linalg.norm(pt_1-pt_2)


def to_find_closer(data_atlas, voxel_coordinate):
    d_min = 9999999

    x, y, z = np.where(data_atlas != 0)

    for x1, y1, z1 in zip(x,y,z):
        d = distance(np.array([x1, y1, z1]), np.asarray(voxel_coordinate))
        if d <= d_min:
            d_min = d
            id = data_atlas[x1, y1, z1]

    return id

path_atlas = '/home/brainlab/Desktop/Rudas/Data/Parcellation/atlas_NMI_2mm.nii'
#path_atlas = '/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
path_mat_file = '/home/brainlab/Downloads/mni_coord_stats.mat'

mat = scipy.io.loadmat(path_mat_file)

img_atlas = nib.load(path_atlas)
data_atlas = img_atlas.get_data().astype(int)
affine_atlas = img_atlas.affine

coordinate = mat['mni_coord']
activations = mat['activations']

outlier = []

new_image = np.zeros(data_atlas.shape)
image_t_values = np.zeros(data_atlas.shape)
#image_t_values = image_t_values + 4

with open('labels_with_closer.txt', 'w') as f:
    for index in range(coordinate.shape[0]):
        point_coordinate = coordinate[index, :]*1000

        voxel_coordinate = tuple(np.rint(apply_affine(inv(affine_atlas), point_coordinate)).astype(int))

        if data_atlas[voxel_coordinate] == 0:
            f.write("%s\n" % to_find_closer(data_atlas, voxel_coordinate))
            #f.write("%s\n" % data_atlas[voxel_coordinate])
        else:
            f.write("%s\n" % data_atlas[voxel_coordinate])

        if data_atlas[voxel_coordinate] == 0:
            new_image[voxel_coordinate] = 5
            #outlier.append(coordinate[index, :])
        else:
            image_t_values[voxel_coordinate] = activations[index]
            #print(activations[index])

nib.save(nib.Nifti1Image(new_image, affine_atlas), 'new_image.nii')
#nib.save(nib.Nifti1Image(image_t_values, affine_atlas), 'new_image_t_values.nii')

n = len(outlier)
print(n)

#from nilearn import surface
#big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
#big_texture = surface.vol_to_surf(nib.Nifti1Image(image_t_values, affine_atlas), big_fsaverage.pial_right)

#plotting.plot_surf_stat_map(big_fsaverage.infl_right,
#                            big_texture, hemi='right', colorbar=True,
#                            title='Surface right hemisphere: fine mesh',
#                            threshold=4.1, bg_map=big_fsaverage.sulc_right)

#plotting.show()

#plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=path_atlas,
#                       hemi='left', view='lateral',
#                       bg_map=fsaverage['sulc_left'], bg_on_data=True)

#plotting.show()

#plotting.plot
plotting.plot_stat_map(nib.Nifti1Image(image_t_values, affine_atlas), display_mode='z', cut_coords=[-30,-20,-10,0,10,20,30],threshold=3.5, bg_img=path_atlas)
plotting.show()

plotting.plot_stat_map(nib.Nifti1Image(new_image, affine_atlas), display_mode='z', cut_coords=[-30,-20,-10,0,10,20,30], vmax=10, threshold=3.5, bg_img=path_atlas)
plotting.show()
#view = plotting.view_img_on_surf(nib.Nifti1Image(image_t_values, affine_atlas), threshold=1, vmax=8)
#view.open_in_browser()

#dis = plotting.plot_anat(path_atlas)

#dis.add_graph(np.zeros((n, n)).astype(np.int8), np.array(outlier)*1000,
#                      node_color='auto', node_size=10,
#                      edge_vmin=None, edge_vmax=None,
#                      edge_threshold="80%",
#                      edge_kwargs=None, node_kwargs=None,
#                      colorbar=False)

#plotting.plot_connectome(np.zeros((n, n)).astype(np.int8), np.array(outlier)*1000, node_size=5, edge_threshold="80%")
#plotting.show()