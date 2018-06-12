import os
import numpy as np
import matplotlib.pyplot as plt  # for visualising and debugging
from scipy.ndimage.morphology import distance_transform_edt
from skimage.io import imsave, imread
from skimage.segmentation import find_boundaries
from demos.unet.file_sorter import get_user_args

W_0, SIGMA = 10, 5


def construct_weights_and_mask(img):
    seg_boundaries = find_boundaries(img, mode='inner')

    bin_img = img > 0
    # take segmentations, ignore boundaries
    binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)

    foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
    background_weight = 1 - foreground_weight

    # build euclidean distances maps for each cell:
    cell_ids = [x for x in np.unique(img) if x > 0]
    distances = np.zeros((img.shape[0], img.shape[1], len(cell_ids)))

    for i, cell_id in enumerate(cell_ids):
        distances[..., i] = distance_transform_edt(img != cell_id)

    # we need to look at the two smallest distances
    distances.sort(axis=-1)

    weight_map = W_0 * np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[..., 0] + distances[..., 1]) ** 2))
    weight_map[binary_with_borders] = foreground_weight
    weight_map[~binary_with_borders] += background_weight

    return weight_map, binary_with_borders


def main():
    args = get_user_args()
    for experiment_name in args.experiment_names:
        file_dir = os.path.join(args.file_dir, experiment_name, 'niftynet_data')
        for f_name in [f for f in os.listdir(file_dir) if f.startswith('seg') and f.endswith('.tif')]:
            img = imread(os.path.join(file_dir, f_name))
            weight_map, binary_seg = construct_weights_and_mask(img)
            imsave(os.path.join(file_dir, f_name).replace('seg', 'weight'), weight_map)
            imsave(os.path.join(file_dir, f_name).replace('seg', 'bin_seg'), binary_seg.astype(np.uint8))


if __name__ == "__main__":
    main()
