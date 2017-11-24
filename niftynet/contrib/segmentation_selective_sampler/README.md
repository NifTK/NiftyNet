### Training segmentation network with selective sampling of image windows

To run this application (from NiftyNet source code root folder)
with a config file (e.g., `niftynet/contrib/segmentation_selective_sampler/selective_seg.ini`)
```bash
python net_run.py train -a niftynet.contrib.segmentation_selective_sampler.ss_app.SelectiveSampling \
                        -c niftynet/contrib/segmentation_selective_sampler/selective_seg.ini
```
