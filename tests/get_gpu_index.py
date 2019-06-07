# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from subprocess import Popen, PIPE

bash_string = "nvidia-smi --query-gpu=memory.free --format=csv"
p1 = Popen(bash_string.split(), stdout=PIPE)
bash_string = "tail -n 2"
p2 = Popen(bash_string.split(), stdin=p1.stdout, stdout=PIPE)
p3 = Popen(['sed', 's: MiB::'], stdin=p2.stdout, stdout=PIPE)
output, error = p3.communicate()

free_memory = [float(x) for x in output.decode('utf-8').split('\n')[:-1]]
if free_memory[1] > free_memory[0]:
    print('1')
else:
    print('0')
