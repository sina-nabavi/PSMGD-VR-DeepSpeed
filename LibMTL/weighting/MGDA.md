```python 
def backward(self, losses, **kwargs):
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]
        ...
        if self.rep_grad:
            self._backward_new_grads(sol, per_grads=per_grads)
        else:
            self._backward_new_grads(sol, grads=grads)
```
It seems that the backward pass either processes the shared gradients or task-specific ones and one can not train both. 

What is per_grad i.e. representation gradients? 

```python
grad_mat = grads.mm(grads.t())
init_sol = _min_norm_2d(grad_mat)
```
- `init_sol` is $\gamma$ in Algorithm 1 in page 5 of paper.

```python
_parser.add_argument('--rep_grad', action='store_true', default=False, 
                    help='computing gradient for representation or sharing parameters')
```
Saw this in the config file. Does this indicate that by default we do not share parameters among tasks? That is not a reasonable assumption.

* Why is the implementation so complicated? The representation of MGDA (2012) is very straight-forward. For reaching the minimum, they argue that it is an optimization problem equivalent to **finding a minimum-norm point in the convex hull which has been well-studied in computational geometry.** 

* It's bad practice that MGDA takes `mgda_gn = kwargs['mgda_gn']` for every `backward`.