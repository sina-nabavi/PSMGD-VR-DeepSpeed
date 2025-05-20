`Grad overflow at Rank 0`

` TEST: nan 0.0332 0.1314 |` What is `nan`?

Make sure that the GPU requirement is devided now. It does not matter if we occupy 3 GPUs yet use the same amount on each as a single GPU scenario! So, find a borderline where the single GPU fails OOM and deepspeed runs perfectly.

According to DeepSpeed's [tutorial](https://www.deepspeed.ai/getting-started/):


*Learning Rate Scheduler: when using a DeepSpeed’s learning rate scheduler (specified in the ds_config.json file), DeepSpeed calls the step() method of the scheduler at every training step (when model_engine.step() is executed). When not using DeepSpeed’s learning rate scheduler:*

*if the schedule is supposed to execute at every training step, then the user can pass the scheduler to deepspeed.initialize when initializing the DeepSpeed engine and let DeepSpeed manage it for update or save/restore.
if the schedule is supposed to execute at any other interval (e.g., training epochs), then the user should NOT pass the scheduler to DeepSpeed during initialization and must manage it explicitly.*

I deactivated the DS scheduler in the config!

watch -n 5 nvidia-smi --query-compute-apps=pid,name,gpu_bus_id,used_gpu_memory --format=csv,noheader