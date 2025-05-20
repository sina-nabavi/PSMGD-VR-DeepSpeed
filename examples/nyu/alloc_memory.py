import torch
import time

def allocate_max_memory_on_device(device):
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.empty(1).cuda() #<<< This is so it starts the allocated memory tracker. Weird behaviour without this. 
    reserved_memory = torch.cuda.memory_allocated(device)

    safe_memory = (total_memory - reserved_memory) *0.95 
    num_elements = int(safe_memory / 4)

    large_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
    print(f"Allocated a tensor with {large_tensor.nelement() * large_tensor.element_size() / (1024 ** 2):.2f} MB on {device}")
    return large_tensor

def allocate_max_memory_all_devices(hold_time=60):
    tensors = []
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available.")
    
    # Allocate memory on all GPUs
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        tensor = allocate_max_memory_on_device(device)
        tensors.append(tensor)
    
    # Hold the memory by sleeping
    print(f"Holding memory for {hold_time} seconds...")
    time.sleep(hold_time)
    print("Releasing memory...")

try:
    allocate_max_memory_all_devices(100000000)  # Adjust hold time as needed
except Exception as e:
    print(f"An error occurred: {str(e)}")
