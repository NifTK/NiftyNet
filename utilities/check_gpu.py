from __future__ import absolute_import, print_function
import sys
import os
from subprocess import Popen, PIPE

if __name__ == "__main__":
    minimum = int(sys.argv[1])
    bash_string = "nvidia-smi --query-gpu=memory.free --format=csv"
    p1 = Popen(bash_string.split(), stdout=PIPE)
    p3 = Popen(['sed', 's: MiB::'], stdin=p1.stdout, stdout=PIPE)
    output, error = p3.communicate()

    free_memory = [float(x) for x in output.split('\n')[1:-1]]
    for (i, x) in enumerate(free_memory):
        if x > minimum:
            print(i)
            break;
