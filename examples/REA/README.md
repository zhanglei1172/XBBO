# REA


## Introduction

[Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548)

## Abstract

> The effort devoted to hand-crafting neural network image classifiers has motivated the use of architecture search to discover them automatically. Although evolutionary algorithms have been repeatedly applied to neural network topologies, the image classifiers thus discovered have remained inferior to human-crafted ones. Here, we evolve an image classifier—AmoebaNet-A—that surpasses hand-designs for the first time. To do this, we modify the tournament selection evolutionary algorithm by introducing an age property to favor the younger genotypes. Matching size, AmoebaNet-A has comparable accuracy to current state-of-the-art ImageNet models discovered with more complex architecture-search methods. Scaled to larger size, AmoebaNet-A sets a new state-of-theart 83.9% top-1 / 96.6% top-5 ImageNet accuracy. In a controlled comparison against a well known reinforcement learning algorithm, we give evidence that evolution can obtain results faster with the same hardware, especially at the earlier stages of the search. This is relevant when fewer compute resources are available. Evolution is, thus, a simple method to effectively discover high-quality architectures.

## Usage

E.g. `PYTHONPATH='./' python examples/REA/rosenbrock_rea.py`


## benchmark

Modify the following section of `comparison/xbbo_benchmark.py` :

```python
test_algs = ["bore"]
```
And run `PYTHONPATH='./' python comparison/xbbo_benchmark.py` in the command line.

## Results


### Branin

|   Method  |    Minimum    | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| :--------: | :-----------: | :----------: | :-----------------: | :----------------: | :--------------------: |
| XBBO(rea) | 0.420+/-0.026 |    0.404     |        111.2        |       51.911       |           29           |
