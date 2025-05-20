* What is the difference between *architecture* and the *encoder-decoder* they later use? Each architecture includes encoders and decoders. 

* How come the model does the backward? 
```
w = self.model.backward(train_losses, **self.kwargs['weight_args'])
```

* As MTLmodel inherits from architecture and from weighting, it inherits `forward` from architecture and `backward` from weighting.

* Orignially, `backward` is a `Torch.tensor` or `autograd` method. Implementing it for `Model` here is just for name conventions. Therefore, for optimizing the performance, we only consider the inner `backward` for now! 
