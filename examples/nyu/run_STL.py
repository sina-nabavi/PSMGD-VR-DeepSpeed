

import subprocess
import os
import time
from itertools import product
import random

# Parameters
weightings = ['STL']
main_tasks = [0,1,2]
n_values = [8]
#seeds = [random.randint(0, 2**32 - 1) for i in range(1,4)]

# GPUs and processes per GPU
gpus = ['0','7']
max_processes_per_gpu = 2

# Track running processes and GPU occupancy
running_processes = []
gpu_occupancy = {gpu: 0 for gpu in gpus}

while main_tasks or running_processes:
    # Start new processes if GPUs have free slots and combinations are left
    started_process = False
    for gpu in gpus:
        while gpu_occupancy[gpu] < max_processes_per_gpu and main_tasks:
            main_task = main_tasks.pop(0)

            # Set the environment variable for this process
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu

            # Output filename
            output_file = f'./PSMGDVR/STL_main_task_{main_task}_.txt'

            # Construct command
            cmd = [
                'python', 'main.py',
                '--weighting', 'STL',
                '--arch', 'HPS',
                '--dataset_path', './data/',
                '--scheduler', 'step',
                '--mode', 'train',
                '--save_path', './out/',
                '--epochs', '200',
                '--step_size', '100',
                '--main_task', str(main_task),
            ]

            # Open output file
            f = open(output_file, 'w')

            # Start subprocess
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

            # Keep track of the process
            running_processes.append({'process': process, 'gpu': gpu, 'file': f})
            gpu_occupancy[gpu] += 1
            started_process = True

            print(f"Started: STL main task {main_task}, on GPU {gpu}")

    # Check for completed processes
    for proc_info in running_processes[:]:
        if proc_info['process'].poll() is not None:
            proc_info['file'].close()
            gpu_occupancy[proc_info['gpu']] -= 1
            running_processes.remove(proc_info)
            print(f"Completed: Process on GPU {proc_info['gpu']}")

    # Short delay to avoid busy waiting if nothing new started
    if not started_process:
        time.sleep(1)