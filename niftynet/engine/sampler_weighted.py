# -*- coding: utf-8 -*-
"""
Generating image window by weighted sampling map from input image
This can also be considered as a `weighted random cropping` layer of the
input image
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import N_SPATIAL
from niftynet.engine.sampler_uniform import UniformSampler


class WeightedSampler(UniformSampler):
    """
    This class generators samples from a user provided
    frequency map for each input volume
    The sampling likelihood of each voxel (and window arround)
    is proportional to its frequency

    This is implemented in a closed form using commulative histograms
    for efficiency purposes i.e., the first three dims of image.

    This layer can be considered as a `weighted random cropping` layer of the
    input image.
    """

    def __init__(self,
                 reader,
                 data_param,
                 batch_size,
                 windows_per_image,
                 queue_length=10):
        UniformSampler.__init__(self,
                                reader=reader,
                                data_param=data_param,
                                batch_size=batch_size,
                                windows_per_image=windows_per_image,
                                queue_length=queue_length)
        tf.logging.info('Initialised weighted sampler window instance')

    def layer_op(self):
        """
        This function generates sampling windows to the input buffer
        image data are from self.reader()
        it first completes window shapes based on image data,
        then finds random coordinates based on the window shapes
        finally extract window with the coordinates and output
        a dictionary (required by input buffer)
        :return: output data dictionary {placeholders: data_array}
        """
        while True:
            image_id, data, _ = self.reader(idx=None, shuffle=True)
            if not data:
                break
            image_shapes = {
                name: data[name].shape for name in self.window.names}
            static_window_shapes = self.window.match_image_shapes(image_shapes)

            # find random coordinates based on window and image shapes
            coordinates = weighted_spatial_coordinates(
                image_id, image_shapes,
                static_window_shapes, data['sampler'], self.window.n_samples)

            # initialise output dict, placeholders as dictionary keys
            # this dictionary will be used in
            # enqueue operation in the form of: `feed_dict=output_dict`
            output_dict = {}
            # fill output dict with data
            for name in list(data):
                coordinates_key = self.window.coordinates_placeholder(name)
                image_data_key = self.window.image_data_placeholder(name)

                # fill the coordinates
                location_array = coordinates[name]
                output_dict[coordinates_key] = location_array

                # fill output window array
                image_array = []
                for window_id in range(self.window.n_samples):
                    # tf.logging.info("locations  %s",
                    #                 location_array[window_id, 1:])
                    x_start, y_start, z_start, x_end, y_end, z_end = \
                        location_array[window_id, 1:]
                    try:
                        image_window = data[name][
                            x_start:x_end, y_start:y_end, z_start:z_end, ...]
                        image_array.append(image_window[np.newaxis, ...])
                    except ValueError:
                        tf.logging.fatal(
                            "dimensionality miss match in input volumes, "
                            "please specify spatial_window_size with a "
                            "3D tuple and make sure each element is "
                            "smaller than the image length in each dim.")
                        raise
                # [tf.logging.info('%s', item.shape) for item in image_array]
                if len(image_array) > 1:
                    output_dict[image_data_key] = \
                        np.concatenate(image_array, axis=0)
                else:
                    output_dict[image_data_key] = image_array[0]
            # the output image shape should be
            # [enqueue_batch_size, x, y, z, time, modality]
            # where enqueue_batch_size = windows_per_image
            yield output_dict


def weighted_spatial_coordinates(subject_id,
                                 img_sizes,
                                 win_sizes,
                                 data,
                                 n_samples=1):
    """
    This is the function that actually does the cumulative histogram
    and sampling.

    also, note that win_sizes could be different,
    for example in segmentation network
    input image window size is 32x32x10,
    training label window is 16x16x10, the network reduces x-y plane
    spatial resolution.
    This function handles this situation by first find the largest
    window across these window definitions, and generate the coordinates.
    These coordinates are then adjusted for each of the
    smaller window sizes (the output windows are concentric).
    """
    n_samples = max(n_samples, 1)
    uniq_spatial_size = set([img_size[:N_SPATIAL]
                             for img_size in list(img_sizes.values())])
    if len(uniq_spatial_size) > 1:
        tf.logging.fatal("Don't know how to generate sampling "
                         "locations: Spatial dimensions of the "
                         "grouped input sources are not "
                         "consistent. {}".format(uniq_spatial_size))
        raise NotImplementedError
    uniq_spatial_size = uniq_spatial_size.pop()

    # find spatial window location based on the largest spatial window
    spatial_win_sizes = [win_size[:N_SPATIAL]
                         for win_size in win_sizes.values()]
    spatial_win_sizes = np.asarray(spatial_win_sizes, dtype=np.int32)
    max_spatial_win = np.max(spatial_win_sizes, axis=0)

    # testing window size
    for i in range(0, N_SPATIAL):
        assert uniq_spatial_size[i] >= max_spatial_win[i], \
            "window size {} is larger than image size {}".format(
                max_spatial_win[i], uniq_spatial_size[i])

    # get cropped version of the input image where the centre of
    # the window might be. If the centre of the window was outside of
    # this crop area, the patch would be outside of the field of view
    half_win = np.floor(max_spatial_win / 2).astype(int)
    windowed_data = data[
        half_win[0]:-half_win[0] if max_spatial_win[0] > 1 else 1,
        half_win[1]:-half_win[1] if max_spatial_win[1] > 1 else 1,
        half_win[2]:-half_win[2] if max_spatial_win[2] > 1 else 1, 0, 0]

    # Get the cumulative sum of the normalised sorted intensities
    # i.e. first sort the sampling frequencies, normalise them
    # to sum to one, and then accumulate them in order
    flat_window = windowed_data.flatten()
    sorted_data = np.cumsum(np.divide(np.sort(flat_window), flat_window.sum()))
    # get the sorting indexes to that we can invert the sorting later on.
    sorted_indexes = np.argsort(flat_window)

    middle_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    for sample in range(0, n_samples):
        # get n_sample from the comulative histogram, spaced by 1/n_samples,
        # plus a random perturbation to give us a stochastic sampler
        sample_ratio = 1 - (np.random.random() + sample) / (n_samples + 1)
        # find the index where the comulative it above the sample threshold
        sample_index = np.argmax(sorted_data >= sample_ratio)
        # inver the sample index to the pre-sorted index
        inverted_sample_index = sorted_indexes[sample_index]
        # get the x,y,z coordinates on the cropped windowed_data
        # (note: we need to re-shift it later due to the crop)
        middle_coords[sample, :N_SPATIAL] = np.unravel_index(
            inverted_sample_index, windowed_data.shape)[:N_SPATIAL]

    # adjust max spatial coordinates based on each mod spatial window size
    all_coordinates = {}
    for mod in list(win_sizes):
        win_size = win_sizes[mod][:N_SPATIAL]
        half_win_diff = np.floor((max_spatial_win - win_size) / 2.0)

        # shift starting coordinates of the window
        # Note that we did not shift the centre coordinates
        # above to the corner of the window
        # because the shift is the same as the cropping amount
        # Also, we need to add half_win_diff/2 so that smaller windows
        # are centred within the large windows
        spatial_coords = np.zeros((n_samples, N_SPATIAL * 2), dtype=np.int32)
        spatial_coords[:, :N_SPATIAL] = \
            middle_coords[:, :N_SPATIAL] + half_win_diff[:N_SPATIAL]

        # the opposite corner of the window is
        # just adding the mod specific window size
        spatial_coords[:, N_SPATIAL:] = \
            spatial_coords[:, :N_SPATIAL] + win_size[:N_SPATIAL]
        # include the subject id
        subject_id = np.ones((n_samples,), dtype=np.int32) * subject_id
        spatial_coords = np.append(subject_id[:, None], spatial_coords, axis=1)
        all_coordinates[mod] = spatial_coords

    return all_coordinates
