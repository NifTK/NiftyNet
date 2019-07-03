
Buffer queue in NiftyNet
====================
This example ilustrates 
how the `buffer_queue` is created 
and modified after each iteration during the network training.


```python
from random import shuffle
import random
import string
```


```python
num_volumes = 4
windows_per_volume = 10
num_iterations = 10
batch_size = 5
```

As mentioned in the [config file](.), image window samplers fill the `buffer_queue`
and networks read the `buffer_queue`. In the following two lines,
sampled windows in the `buffer_queue ` are randomly generated and written as
`volume_id-window_id`.



```python
volumes_ids = [random.choice(string.ascii_uppercase) for _ in range(num_volumes)] 
windows_ids = list(range(windows_per_volume))
```


```python
def generate_ids(volumes_ids, windows_ids):
    return ''.join(random.choice(volumes_ids))\
           + '-' + ''.join(str(random.choice(windows_ids)))
```


```python
# Defaut configuration
queue_length_default = 10
queue_length = int(max(queue_length_default, round(batch_size * 2.5)))
buffer_queue = [generate_ids(volumes_ids, windows_ids) for _ in range(queue_length)]
print("[INFO] Initial buffer_queue: {}".format(buffer_queue))
```

    [INFO] Initial buffer_queue: ['K-2', 'P-5', 'A-4', 'K-8', 'A-3', 'J-9', 'K-6', 'A-4', 'K-1', 'J-9', 'A-6', 'K-6']


Inside the `for` loop, a snippet code shows the `buffer_queue` being updated during the network training process. At first, the `buffer_queue` is shuffled, secondly `batch_size` image windows are read from the queue for the training computations. Lastly, the `buffer_queue` is updated with new `batch_size` image windows for the next iteration.


```python
for iterations in range(num_iterations):
    shuffle(buffer_queue)
    image_windows = [buffer_queue.pop(0) for ii in range(batch_size)]
    print("[INFO] Iteration: {} Image windows: {}".format(iterations + 1, image_windows))

    '''
    Do something ...
    '''
    # Update the buffer queue with new image windows for the next iteration    
    buffer_queue += [generate_ids(volumes_ids, windows_ids) for _ in range(batch_size)]
    print("[INFO] Updated buffer {}".format(buffer_queue, iterations + 1))
    print("\n\n")
```

    [INFO] Iteration: 1 Image windows: ['J-9', 'K-6', 'A-3', 'A-6', 'A-4']
    [INFO] Updated buffer ['A-4', 'K-1', 'J-9', 'K-6', 'P-5', 'K-8', 'K-2', 'A-6', 'A-0', 'A-3', 'J-7', 'A-2']
    
    
    
    [INFO] Iteration: 2 Image windows: ['K-1', 'A-6', 'A-3', 'J-7', 'K-2']
    [INFO] Updated buffer ['P-5', 'A-4', 'K-6', 'J-9', 'K-8', 'A-2', 'A-0', 'J-7', 'A-6', 'J-7', 'J-7', 'J-0']
    
    
    
    [INFO] Iteration: 3 Image windows: ['A-0', 'K-8', 'J-7', 'A-6', 'A-2']
    [INFO] Updated buffer ['J-7', 'J-7', 'P-5', 'J-9', 'K-6', 'J-0', 'A-4', 'K-4', 'K-3', 'K-1', 'P-5', 'K-2']
    
    
    
    [INFO] Iteration: 4 Image windows: ['J-7', 'K-2', 'K-6', 'K-1', 'P-5']
    [INFO] Updated buffer ['J-0', 'K-4', 'J-9', 'A-4', 'J-7', 'K-3', 'P-5', 'P-8', 'P-5', 'P-5', 'K-2', 'J-7']
    
    
    
    [INFO] Iteration: 5 Image windows: ['P-5', 'P-5', 'J-9', 'J-7', 'J-0']
    [INFO] Updated buffer ['K-3', 'P-8', 'K-4', 'A-4', 'J-7', 'K-2', 'P-5', 'A-2', 'A-1', 'A-6', 'J-3', 'J-1']
    
    
    
    [INFO] Iteration: 6 Image windows: ['P-8', 'J-3', 'K-4', 'J-7', 'A-4']
    [INFO] Updated buffer ['A-6', 'J-1', 'A-2', 'A-1', 'P-5', 'K-2', 'K-3', 'J-8', 'A-4', 'K-1', 'J-7', 'J-0']
    
    
    
    [INFO] Iteration: 7 Image windows: ['K-2', 'A-2', 'J-7', 'P-5', 'J-1']
    [INFO] Updated buffer ['A-4', 'A-6', 'J-0', 'A-1', 'K-3', 'J-8', 'K-1', 'J-7', 'A-6', 'J-5', 'A-7', 'P-0']
    
    
    
    [INFO] Iteration: 8 Image windows: ['J-0', 'J-8', 'J-5', 'P-0', 'K-1']
    [INFO] Updated buffer ['A-6', 'A-4', 'A-1', 'A-7', 'A-6', 'J-7', 'K-3', 'K-8', 'K-5', 'P-0', 'A-7', 'J-3']
    
    
    
    [INFO] Iteration: 9 Image windows: ['K-5', 'A-7', 'J-3', 'A-4', 'A-1']
    [INFO] Updated buffer ['K-8', 'A-6', 'A-7', 'K-3', 'A-6', 'J-7', 'P-0', 'P-9', 'J-2', 'J-4', 'P-1', 'K-6']
    
    
    
    [INFO] Iteration: 10 Image windows: ['P-0', 'A-7', 'P-1', 'K-8', 'A-6']
    [INFO] Updated buffer ['P-9', 'J-2', 'K-6', 'K-3', 'J-4', 'J-7', 'A-6', 'K-1', 'P-2', 'K-0', 'J-0', 'P-9']
    
    
    

