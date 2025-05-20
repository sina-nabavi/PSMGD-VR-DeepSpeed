* For now, edited two lines:

`43, 59` in `_compute_grad` 

* I do not know if parameter zeroing by oursevles would be okay

* I do not know if MTL becomes the bottleneck after gradient calculation: In this case we have to also manually distribute the Tensors.

* Cannot figure out `111` and `117` yet as I do not know if they are actually caluclating gradients, or saving it for later. 