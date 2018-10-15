import sys
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from niftynet.io.image_reader import ImageReader
from niftynet.engine.image_window_dataset import ImageWindowDataset
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator as IA
from niftynet.layer.pad import PadLayer


def vis_coordinates(image, coordinates=None, saving_name='image.png', dpi=50):
    """
    Plot image, and on top of it, draw boxes with the window coordinates
    the figure is saved at `saving_name`
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    all_patch = []
    if coordinates is not None:
        for win in coordinates[::-1]:
            patch = patches.Rectangle(
                xy=(win[2], win[1]),
                width=win[5] - win[2],
                height=win[4] - win[1],
                linewidth=1)
            all_patch.append(patch)
    if all_patch:
        all_pc = PatchCollection(
            all_patch, alpha=0.6, edgecolor='r', facecolor='#f5e44c')
        ax.add_collection(all_pc)
    #  plt.show()
    if saving_name:
        fig.savefig(saving_name, bbox_inches='tight', pad_inches=0, dpi=dpi)
        return


###
# config parameters
###
spatial_window_size = (100, 100)
border = (12, 12)
volume_padding_size = (50, 50)
data_param = \
    {'MR': {
        'path_to_search': '~/Desktop/useful_scripts/visualise_windows',
        'filename_contains': 'example.png'}}

###
# create an image reader
###
reader = ImageReader().initialise(data_param)
reader.add_preprocessing_layers(  # add volume padding layer
    [PadLayer(image_name=['MR'],
              border=volume_padding_size, mode='constant')])

###
# show 'volume' -- without window sampling
###
image_2d = ImageWindowDataset(reader)()['MR'][0, :, :, 0, 0, 0]
vis_coordinates(image_2d, saving_name='output/image.png')

###
# create & show uniform random samples
###
uniform_sampler = UniformSampler(
    reader, spatial_window_size, windows_per_image=100)
next_window = uniform_sampler.pop_batch_op()
coords = []
with tf.Session() as sess:
    for _ in range(20):
        uniform_windows = sess.run(next_window)
        coords.append(uniform_windows['MR_location'])
coords = np.concatenate(coords, axis=0)
vis_coordinates(image_2d, coords, 'output/uniform.png')

###
# create & show all grid samples
###
grid_sampler = GridSampler(
    reader, spatial_window_size, window_border=border)
next_grid = grid_sampler.pop_batch_op()
coords = []
with tf.Session() as sess:
    while True:
        window = sess.run(next_grid)
        if window['MR_location'][0, 0] == -1:
            break
        coords.append(window['MR_location'])
coords = np.concatenate(coords, axis=0)
vis_coordinates(image_2d, coords, 'output/grid.png')

###
# create & show cropped grid samples (in aggregator)
###
n_window = coords.shape[0]
dummy_window = np.zeros((n_window, 800, 800, 1, 1))
_, coords = IA.crop_batch(dummy_window, coords, border=border)
vis_coordinates(image_2d, coords, 'output/grid_cropped.png')

