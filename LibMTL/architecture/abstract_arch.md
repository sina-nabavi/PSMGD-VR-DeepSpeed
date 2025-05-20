* Is it a correct assumption that decoders are task specific? Do we have that assumption across all MTL works? Because every `forward` implementation in this work assumes that. 

* The implementation of `architecture` is dependent on model. For example, `cross\_stich` assumes the encoders have the overal `resnet` architecture. 