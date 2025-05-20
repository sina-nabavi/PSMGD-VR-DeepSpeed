import subprocess
import os
import time
from itertools import product
import random

# Parameters
weightings = ['EW', 'UW', 'GradNorm', 'GLS', 'RLW', 'MGDA', 'IMTL',
                            'PCGrad', 'GradVac', 'CAGrad', 'GradDrop', 'DWA', 
                            'Nash_MTL', 'MoCo', 'Aligned_MTL', 'DB_MTL', 'STCH', 
                            'ExcessMTL', 'FairGrad']
n_values = [8]
seeds = [random.randint(0, 2**32 - 1) for i in range(1,4)]

# GPUs and processes per GPU
gpus = ['2','3','4']
max_processes_per_gpu = 2

# All combinations of parameters
combinations = list(product(seeds, n_values, weightings))

# Track running processes and GPU occupancy
running_processes = []
gpu_occupancy = {gpu: 0 for gpu in gpus}

while combinations or running_processes:
    # Start new processes if GPUs have free slots and combinations are left
    started_process = False
    for gpu in gpus:
        while gpu_occupancy[gpu] < max_processes_per_gpu and combinations:
            seed, n_val, weighting = combinations.pop(0)

            # Set the environment variable for this process
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu

            # Output filename
            output_file = f'./PSMGDVR_lower_lr/weighting_{weighting}_seed_{seed}.txt'

            # Construct command
            cmd = [
                'python', 'main_base.py',
                '--weighting', weighting,
                '--arch', 'HPS',
                '--dataset_path', './data/',
                '--scheduler', 'step',
                '--lr', str(1e-5),
                '--mode', 'train',
                '--save_path', './out/',
                '--epochs', '300',
                '--step_size', '100',
                '--seed', str(seed),
            ]

            # Open output file
            f = open(output_file, 'w')

            # Start subprocess
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

            # Keep track of the process
            running_processes.append({'process': process, 'gpu': gpu, 'file': f})
            gpu_occupancy[gpu] += 1
            started_process = True

            print(f"Started: PSMGDVR n {n_val} weighting {weighting} seed {seed}, on GPU {gpu}")

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