## PyTorchNiftyNet

This demo provides a minimal working example of segmentation network training/validation/inference,
combining:
- NiftyNet (`v0.4`)'s image reader, window sampler, window aggregator classes; 

and

- neural networks and optimisation methods implemented in PyTorch (tested with `v0.4`).

The PyTorch loss function and 3D U-net are adapted from 
https://github.com/pykao/Modified-3D-UNet-Pytorch.


#### Summary
The key element of the demo is a tiny ["adapter" class](libs/dataset_niftynet.py). 
It wraps NiftyNet's sampler instance in a PyTorch Dataset object:
```python
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

```
Instance of this adapter class could then be used with `torch.utils.data.DataLoader`:
```python
adapter_sampler = DatasetNiftySampler(sampler=niftynet_sampler)
torch.utils.data.DataLoader(
    adapter_sampler, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=4)
```
where `niftynet_sampler` is an instance of NiftyNet window sampler.
It could be initialised to benefit from NiftyNet's:
- training/validation dataset splitting,
- image volume preprocessors, 
- image filename search and matching

functionality. 

In this demo, multiple image readers and samplers are created:
```python

# Dictionary with data parameters for NiftyNet Reader
data_param = {
    'image': {'path_to_search': opt.image_path, 'filename_contains': 'CC'},
    'label': {'path_to_search': opt.label_path, 'filename_contains': 'CC'}}

image_sets_partitioner = ImageSetsPartitioner().initialise(
    data_param=data_param,
    data_split_file=opt.data_split_file,
    new_partition=False,
    ratios=opt.ratios
)

readers = {x: get_reader(data_param, image_sets_partitioner, x)
           for x in ['training', 'validation', 'inference']}
samplers = {x: get_sampler(readers[x], opt.patch_size, x)
            for x in ['training', 'validation', 'inference']}
```

_Check out [the IO module demo](../module_examples) 
for more information about the readers and samplers._
