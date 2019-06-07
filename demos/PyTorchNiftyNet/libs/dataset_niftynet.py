import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetNiftySampler(Dataset):
    """
    A simple adapter
    converting NiftyNet sampler's output into PyTorch Dataset properties
    """
    def __init__(self, sampler):
        super(DatasetNiftySampler, self).__init__()
        self.sampler = sampler

    def __getitem__(self, index):
        data = self.sampler(idx=index)

        # Transpose to PyTorch format
        image = np.transpose(data['image'], (0, 5, 1, 2, 3, 4))
        label = np.transpose(data['label'], (0, 5, 1, 2, 3, 4))

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

    def __len__(self):
        return len(self.sampler.reader.output_list)
