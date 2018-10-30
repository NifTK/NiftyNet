import torch
from torch.utils.data import Dataset


class DatasetNiftySampler(Dataset):
    def __init__(self, sampler):
        super(DatasetNiftySampler, self).__init__()
        self.sampler = sampler

    def __getitem__(self, index):
        data = self.sampler(idx=index)
        image = torch.from_numpy(data['image'][..., 0, :]).float()
        label = torch.from_numpy(data['label'][..., 0, :]).float()
        return image, label

    def __len__(self):
        return len(self.sampler.reader.output_list)
