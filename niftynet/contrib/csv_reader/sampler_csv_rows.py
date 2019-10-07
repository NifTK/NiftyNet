import numpy as np
from niftynet.engine.image_window_dataset import ImageWindowDataset
from niftynet.engine.image_window import N_SPATIAL, LOCATION_FORMAT


class ImageWindowDatasetCSV(ImageWindowDataset):
    """
    Extending the default sampler to include csv data
    """

    def __init__(self,
                 reader,
                 csv_reader=None,
                 window_sizes=None,
                 batch_size=10,
                 windows_per_image=1,
                 shuffle=True,
                 queue_length=10,
                 num_threads=4,
                 epoch=-1,
                 smaller_final_batch_mode='pad',
                 name='random_vector_sampler'):
        self.csv_reader = csv_reader
        print("assigned csv_reader")
        ImageWindowDataset.__init__(
            self,
            reader=reader,
            window_sizes=window_sizes,
            batch_size=batch_size,
            windows_per_image=windows_per_image,
            shuffle=shuffle,
            queue_length=queue_length,
            epoch=epoch,
            smaller_final_batch_mode=smaller_final_batch_mode,
            name=name)
        print("initialised IWD")
        self.set_num_threads(num_threads)

    def layer_op(self, idx=None):
        """
        Generating each image as a window.
        Overriding this function to create new image sampling strategies.

        This function should either yield a dictionary
        (for single window per image)::

            yield a dictionary

            {
             'image_name': a numpy array,
             'image_name_location': (image_id,
                                     x_start, y_start, z_start,
                                     x_end, y_end, z_end)
            }

        or return a dictionary (for multiple windows per image)::

            return a dictionary:
            {
             'image_name': a numpy array,
             'image_name_location': [n_samples, 7]
            }

        where the 7-element location vector encode the image_id,
        starting and ending coordinates of the image window.

        Following the same notation, the dictionary can be extended
        to multiple modalities; the keys will be::

            {'image_name_1', 'image_name_location_1',
             'image_name_2', 'image_name_location_2', ...}

        :param idx: image_id used to load the image at the i-th row of
            the input
        :return: a image data dictionary
        """

        # dataset: from a window generator
        # assumes self.window.n_samples == 1
        # the generator should yield one window at each iteration

        if self.window.n_samples == 1:
            assert self.window.n_samples == 1, \
                'image_window_dataset.layer_op() requires: ' \
                'windows_per_image should be 1.'

            image_id, image_data, _ = self.reader(idx=idx)
            print(image_id, idx)
            for mod in list(image_data):
                spatial_shape = image_data[mod].shape[:N_SPATIAL]
                coords = self.dummy_coordinates(image_id, spatial_shape, 1)
                image_data[LOCATION_FORMAT.format(mod)] = coords
                image_data[mod] = image_data[mod][np.newaxis, ...]
            if self.csv_reader is not None:
                _, label_dict, _ = self.csv_reader(subject_id=image_id)
                print(label_dict, image_id, idx)
                image_data.update(label_dict)
                for name in self.csv_reader.names:
                    image_data[name + '_location'] = \
                        image_data['image_location']
            return image_data
        else:
            print("Warning, it may not be ready yet")
            image_id, image_data, _ = self.reader(idx=idx)
            print(image_id, idx)
            for mod in list(image_data):
                spatial_shape = image_data[mod].shape[:N_SPATIAL]
                coords = self.dummy_coordinates(image_id, spatial_shape, 1)
                image_data[LOCATION_FORMAT.format(mod)] = coords
                image_data[mod] = image_data[mod][np.newaxis, ...]
            if self.csv_reader is not None:
                _, label_dict, _ = self.csv_reader(subject_id=image_id)
                print(label_dict, image_id, idx)
                image_data.update(label_dict)
                for name in self.csv_reader.names:
                    image_data[name + '_location'] = image_data[
                        'image_location']
            return image_data

    @property
    def tf_shapes(self):
        """
        returns a dictionary of sampler output tensor shapes
        """
        assert self.window, 'Unknown output shapes: self.window not initialised'
        shape_dict = self.window.tf_shapes
        if self.csv_reader is not None:
            shape_dict.update(self.csv_reader.tf_shapes)
        return shape_dict

    @property
    def tf_dtypes(self):
        """
        returns a dictionary of sampler output tensorflow dtypes
        """
        assert self.window, 'Unknown output shapes: self.window not initialised'
        shape_dict = self.window.tf_dtypes
        if self.csv_reader is not None:
            shape_dict.update(self.csv_reader.tf_dtypes)
        return shape_dict
