## To load Jupyter Notebooks

1) Install conda
2) in conda terminal run the following commands:

```conda create -n tensorflow python=3.5```

If you are using bash terminal:
```
source activate tensorflow
```
otherwise:
```
activate tensorflow
```

If you are using gpus:
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-win_amd64.whl
```
otherwise:
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.3.0-cp35-cp35m-win_amd64.whl
```

```
conda install nb_conda_kernels
```
```
jupyter notebook
```


_This demo only supports NiftyNet cloned from [GitHub](https://github.com/NifTK/NiftyNet) or [CMICLab](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet)_

Please see also the demo running on [Microsoft Azure Notebooks](https://notebooks.azure.com/eligibson/libraries/NiftyNetTesting_Public)
