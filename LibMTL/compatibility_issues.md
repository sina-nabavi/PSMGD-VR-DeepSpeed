`torch` and `nvcc` both show `CUDA 11`. I do not know why `nvidia-smi` shows `12.04`.
```
(libmtl) bash-4.4$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:09:46_PDT_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.TC455_06.29190527_0
(libmtl) bash-4.4$ pip freeze | grep torch
torch==1.8.1+cu111
torchvision==0.9.1+cu111
```

Came to fix [this](https://github.com/databrickslabs/dolly/issues/119), and then had to change the setup.config. 
[this](https://github.com/microsoft/DeepSpeed/issues/4889#issuecomment-1876243679) might be a fix for the Adam Fuse error but needs request@cs.fiu.edu to be online.

* Request a feature update on DeepSpeed for supporting MTL or MOO

* This help from o1 resolved the ADAM FUSE error:
```
Even though `gcc --version` shows GCC 8.5.0, the error suggests that **`nvcc` is using a different GCC version**, likely one that's newer than 10. This can happen if `nvcc` picks up a different compiler in your environment.

**To resolve this:**

1. **Check which GCC `nvcc` is using:**

   Run:

   ```bash
   nvcc --compiler-bindir /usr/bin/gcc --forward-unknown-to-host-compiler -v
   ```

   Or compile a simple CUDA file with verbose output:

   ```bash
   nvcc -v test.cu -o test
   ```

   This will show which compiler `nvcc` is invoking.

2. **Specify the GCC version for `nvcc`:**

   Force `nvcc` to use GCC 8.5.0 by setting the `-ccbin` flag:

   ```bash
   export NVCC='/usr/local/cuda/bin/nvcc -ccbin /usr/bin/gcc'
   ```

   Or when building DeepSpeed, specify the compiler:

   ```bash
   CXX=/usr/bin/g++ CC=/usr/bin/gcc python setup.py install
   ```

3. **Ensure environment variables point to GCC 8.5.0:**

   ```bash
   export CC=/usr/bin/gcc
   export CXX=/usr/bin/g++
   export PATH=/usr/bin:$PATH
   ```

4. **Verify GCC version within `nvcc`:**

   Check that `nvcc` now uses the correct GCC:

   ```bash
   nvcc --ccbin=$CC -V
   ```

5. **Rebuild DeepSpeed:**

   After ensuring `nvcc` uses GCC 8.5.0, rebuild your code:

   ```bash
   python setup.py install
   ```

**Summary:** The error occurs because `nvcc` is detecting a GCC version later than 10, despite `gcc --version` showing 8.5.0. By explicitly specifying the GCC version for `nvcc` using the `-ccbin` flag and ensuring your environment variables point to GCC 8.5.0, you align the compiler versions and resolve the issue.
```


*** After finding the right `gcc` version  (>=9) and a torch that has `torch.cuda.is_bf16_supported()`, I found another error about crypt.h that I fixed using [this](https://github.com/stanford-futuredata/ColBERT/issues/309).


Got this error when assigned 3 GPUS:
```
Traceback (most recent call last):
  File "cifar/cifar10_deepspeed.py", line 402, in <module>
    main(args)
  File "cifar/cifar10_deepspeed.py", line 337, in main
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/__init__.py", line 179, in initialize
    config_class = DeepSpeedConfig(config, mpu, mesh_device=mesh_device)
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 798, in __init__
    self._configure_train_batch_size()
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 981, in _configure_train_batch_size
    self._batch_assertion()
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 929, in _batch_assertion
    assert train_batch == micro_batch * grad_acc * self.world_size, (
AssertionError: Check batch related parameters. train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size 16 != 5 * 1 * 3
Traceback (most recent call last):
  File "cifar/cifar10_deepspeed.py", line 402, in <module>
    main(args)
  File "cifar/cifar10_deepspeed.py", line 337, in main
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/__init__.py", line 179, in initialize
    config_class = DeepSpeedConfig(config, mpu, mesh_device=mesh_device)
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 798, in __init__
    self._configure_train_batch_size()
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 981, in _configure_train_batch_size
    self._batch_assertion()
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 929, in _batch_assertion
    assert train_batch == micro_batch * grad_acc * self.world_size, (
AssertionError: Check batch related parameters. train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size 16 != 5 * 1 * 3
Traceback (most recent call last):
  File "cifar/cifar10_deepspeed.py", line 402, in <module>
    main(args)
  File "cifar/cifar10_deepspeed.py", line 337, in main
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/__init__.py", line 179, in initialize
    config_class = DeepSpeedConfig(config, mpu, mesh_device=mesh_device)
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 798, in __init__
    self._configure_train_batch_size()
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 981, in _configure_train_batch_size
    self._batch_assertion()
  File "/aul/homes/snaba002/miniconda3/envs/libmtl/lib/python3.8/site-packages/deepspeed/runtime/config.py", line 929, in _batch_assertion
    assert train_batch == micro_batch * grad_acc * self.world_size, (
AssertionError: Check batch related parameters. train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size 16 != 5 * 1 * 3
```