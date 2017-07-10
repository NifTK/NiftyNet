### Structure
The basic picture of a training procedure (data parallelism) is:
```
<Multi-GPU training>
                                     (engine/training.py)
                                            |>----------------------+
                                            |>---------------+      |
                                            |^|              |      |
                   (engine/input_buffer.py) |^|     sync   GPU_1  GPU_2   ...
                                            |^|     +----> model  model (network/*.py)
with multiple threads:                      |^|     |        |      |
            (layer/input_normalisation.py)  |^|    CPU       v      v (layer/*.py)
image&label ->> (engine/*_sampler.py)   >>>   |   model <----+------+
(*.nii.gz)        (layer/rand_*.py)     >>>   |  update    stochastic gradients
```
