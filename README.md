# torch-lsuv

This is an attempt to reproduce the experiments from [arXiv:1511.06422](http://arxiv.org/abs/1511.06422).

## Work in progress.

## How to use

### MNIST

```bash
th mnist-example.lua --lsuv
```

### CIFAR-10 (In progress).
code taken from [here](https://github.com/szagoruyko/cifar.torch), thanks to [@szagoruyko](https://github.com/szagoruyko))

Download and preprocess data first:

```bash
cd cifar.torch
OMP_NUM_THREADS=2 th -i provider.lua
```
```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```

Train with lsuv:

```bash
CUDA_VISIBLE_DEVICES=0 th train.lua --model vgg_bn_drop -s logs/vgg --lsuv
```

## Results

### test accuracy for MNIST
(with -f key for training on full dataset)

epoch |with lsuv (lr=0.1)| with lsuv (lr=0.05) | without lsuv (lr=0.001) | with lsuv (lr=0.001)
------------ | ------------ | ------------- | ------------- | -------------
1|97.77%|96.69%|83.39%|78.28%
2| 98.45%| 97.94%|89.25%|87.75%
3| 98.63%|98.37%|91.23%|91.19%
4| 98.74%|98.57|92.46%|92.82%
5| 98.88%|98.72%|93.23%|93.81%
6| 98.97%|98.75%| 93.88%|94.53%
7| 99.03%|98.86%| 94.44%|95.06%
8| 99.01%|98.86%|94.81%|95.4%
9| 99.01%|98.9%|95.03%|95.87%
10|98.96%|98.91|95.29%|96.15%

Training without LSUV with learning rates 0.05 and 0.01 did not converge after 10 epochs (the accuracy was the same 11.35% along 10 epochs).


## TODO
* Results on CIFAR

## References
* http://arxiv.org/abs/1511.06422 *LSUV paper*
* http://arxiv.org/abs/1312.6120 *ortonormal init which is used here too*
* https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py *python code of orthogonal init*

Thanks for debugging and help to [@ikostrikov](https://github.com/ikostrikov)
