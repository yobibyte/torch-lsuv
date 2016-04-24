# torch-lsuv

This is an attempt to reproduce the experiments from [arXiv:1511.06422](http://arxiv.org/abs/1511.06422).

##Work in progress.

##How to use

You can run experiment on MNIST.

```bash
th mnist-example.lua --lsuv
```

CIFAR-10 experiment code taken from [here](https://github.com/szagoruyko/cifar.torch), thanks to [@szagoruyko](https://github.com/szagoruyko)

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

##TODO
* Results on MNIST upcoming
* Results on CIFAR upcoming

## References
* http://arxiv.org/abs/1511.06422 *LSUV paper*
* http://arxiv.org/abs/1312.6120 *ortonormal init which we use here to*
* https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py *python code of orthogonal init*

Thanks for debugging and help to [@ikostrikov](https://github.com/ikostrikov)
