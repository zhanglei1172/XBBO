# LaMCTS


## Introduction

[Latent Action Monte Carlo Tree Search (LA-MCTS)](https://arxiv.org/abs/2007.00708)

## Abstract

> Since LaNAS works very well on NAS datasets, e.g. NASBench-101, and the core of the algorithm can be easily generalized to other problems, we extend it to be a generic solver for black-box function optimization. LA-MCTS further improves by using a nonlinear classifier at each decision node in MCTS and use a surrogate (e.g., a function approximator) to evaluate each sample in the leaf node. The surrogate can come from any existing Black-box optimizer (e.g., Bayesian Optimization). The details of LA-MCTS can be found in the following paper.



## Usage

E.g. `PYTHONPATH='./' python examples/LaMCTS/rosenbrock_LaMCTS.py`


## benchmark

Modify the following section of `comparison/xbbo_benchmark.py` :

```python
test_algs = ["lamcts"]
```
And run `PYTHONPATH='./' python comparison/xbbo_benchmark.py` in the command line.


## Results


### Branin

|   Method  |    Minimum    | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| :--------: | :-----------: | :----------: | :-----------------: | :----------------: | :--------------------: |
| XBBO(lamcts) | 0.440+/-0.040 |    0.407     |         81.7        |       39.666       |           4            |
