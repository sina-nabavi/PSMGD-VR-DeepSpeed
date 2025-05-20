For this, we have to edit a lot of parts

* Model loading
* Data loading
* Optimizer Loading
* Backward calculation

* There is a preprocessing phase for the dataloaders, but DeepSpeed requires the dataset itself to perform parallelism. Therefore, we need to send it the pre-processed data or modify deepspeed which is not desirable.

* Therefore, the only challenge is not calling backward, but how to load data as well. 

* Currently trying to omit prepare_dataLoader part without fragility. Investigating the data loading part: `multi_input=False` in happy path.

* `return_weight=False`

* No `validation` dataset. 

```      
self.model.epoch = epoch
self.model.train()
```

I assume that I can still hook model and change attributes. (may be wrong)

* In testing and doing hooks, I'm only interested in knowing whether the idea works and as we do not want to present this code, I do not intend to spend much time to refactor it. Therefore I comment-out any part regarding `multi-input` or `return_weight`

* `main.py` loads `nyuv2_train_loader` as the `train_dataloaders` arguments. As `nyuv2_train_loader` is not pre-processed at all, and `_prepare_dataloaders` does not do anything special (I suppose it only handles `multi-input`), we may be able to transfer the dataloading part completely to DeepSpeed.

*`--scheduler default='step' but we use deepspeed's optimizer`

```
naba002/miniconda3/envs/libmtl/include/python3.8 -c _configtest.c -o _configtest.o
      /aul/homes/snaba002/miniconda3/envs/libmtl/bin/mpicc: line 282: x86_64-conda_cos6-linux-gnu-cc: command not found
      failure.
      removing: _configtest.c _configtest.o
      error: Cannot compile MPI programs. Check your configuration!!!
      Installing mpi4py requires a working MPI implementation.
      If you are running on a supercomputer or cluster, check with
      the system administrator or refer to the system user guide.
      Otherwise, if you are running on a laptop or desktop computer,
      your may be missing the MPICH or Open MPI development package:
      * On Fedora/RHEL systems, run:
        $ sudo dnf install mpich-devel     # for MPICH
        $ sudo dnf install openmpi-devel   # for Open MPI
      * On Debian/Ubuntu systems, run:
        $ sudo apt install libmpich-dev    # for MPICH
        $ sudo apt install libopenmpi-dev  # for Open MPI
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for mpi4py
Failed to build mpi4py
```

To parallelize data go [here](https://discuss.pytorch.org/t/parallel-multi-task-training/154630/2y).

ADAM_FUSE ERROR: update gcc to 9 or higher through CONDA