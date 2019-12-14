# VAE mxnet

Adapted from 

[PyTorch Examples](https://github.com/pytorch/examples/tree/master/vae)

and

[Gluon - The Straight Dope](https://gluon.mxnet.io/chapter13_unsupervised-learning/vae-gluon.html)


## Quick start

```
python main.py --hybrid
```


## Improvement

The original model defined in gluon has `batch_size` attribute

```
# note to self: requring batch_size in model definition is sad, not sure how to deal with this otherwise though
```

My implementation removes this limitation.