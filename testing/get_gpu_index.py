import os
import subprocess

rc = subprocess.call('/home/wenqi/list_gpu_memory.sh', shell=True)
gpu_info = []
with open('./gpu_tmp_log', 'r') as f:
    for line in f.readlines():
        gpu_info.append(line[:-1])
free_memory = [float(x) for x in gpu_info[1:]]
if free_memory[0] > free_memory[1]:
    print('1')
else:
    print('0')

if os.path.exists('./gpu_tmp_log'):
    os.remove('./gpu_tmp_log')
