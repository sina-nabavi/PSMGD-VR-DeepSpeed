import subprocess
import os
import time
from itertools import product

# List of weighting methods
weighting_methods = ['PSMGD', 'method2', 'method3']  # Replace with your actual methods

# List of learning rates
learning_rates = [1e-3, 1e-4, 1e-5]  # Replace with your desired learning rates

# Additional parameters
batch_sizes = [32, 64]  # Example batch sizes
optimizers = ['adam', 'sgd']  # Example optimizers

# List of GPUs available
gpus = ['0', '1', '2', '3']

# Create a list of all combinations of parameters
combinations = list(product(weighting_methods, learning_rates, batch_sizes, optimizers))

# List to keep track of running processes and their assigned GPUs
running_processes = []

# List of available GPUs
available_gpus = gpus.copy()

while combinations or running_processes:
    # Start new processes if GPUs are available and combinations are left
    while combinations and available_gpus:
        weighting, lr, batch_size, optimizer = combinations.pop(0)
        gpu = available_gpus.pop(0)  # Assign a GPU

        # Set the environment variable for this process
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu

        # Define the output file name based on the current settings
        output_file = f'output_weighting_{weighting}_lr_{lr}_bs_{batch_size}_opt_{optimizer}.txt'

        # Construct the command to run
        cmd = [
            'python', 'main.py',
            '--weighting', weighting,
            '--arch', 'HPS',
            '--dataset_path', './data/',
            '--scheduler', 'step',
            '--mode', 'train',
            '--save_path', './out/',
            '--epochs', '200',
            '--step_size', '100',
            '--lr', str(lr),
            '--batch_size', str(batch_size),
            '--optimizer', optimizer,
            '--n', '50'
        ]

        # Open the output file
        f = open(output_file, 'w')

        # Start the subprocess
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

        # Keep track of the process, assigned GPU, and output file
        running_processes.append({'process': process, 'gpu': gpu, 'file': f})

        print(f"Started: Weighting={weighting}, LR={lr}, Batch Size={batch_size}, Optimizer={optimizer}, on GPU {gpu}")

    # Check for any processes that have completed
    for proc_info in running_processes[:]:  # Iterate over a copy of the list
        retcode = proc_info['process'].poll()
        if retcode is not None:  # Process has completed
            # Close the output file
            proc_info['file'].close()

            # Release the GPU
            available_gpus.append(proc_info['gpu'])

            # Remove the process from the running list
            running_processes.remove(proc_info)

            print(f"Completed: Process on GPU {proc_info['gpu']}")

    # Sleep briefly to prevent busy waiting
    time.sleep(1)