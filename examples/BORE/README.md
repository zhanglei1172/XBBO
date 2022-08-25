# BORE


## Introduction

[BORE: Bayesian Optimization by Density-Ratio Estimation](https://arxiv.org/abs/2102.09009) [ICML2021]

[Official Repo](https://github.com/ltiao/bore)

## Abstract

> Bayesian optimization (BO) is among the most effective and widely-used blackbox optimization methods. BO proposes solutions according to an explore-exploit trade-off criterion encoded in an acquisition function, many of which are computed from the posterior predictive of a probabilistic surrogate model. Prevalent among these is the expected improvement (EI) function. The need to ensure analytical tractability of the predictive often poses limitations that can hinder the efficiency and applicability of BO. In this paper, we cast the computation of EI as a binary classification problem, building on the link between class-probability estimation and density-ratio estimation, and the lesser-known link between density-ratios and EI. By circumventing the tractability constraints, this reformulation provides numerous advantages, not least in terms of expressiveness, versatility, and scalability.

## Usage

E.g. `PYTHONPATH='./' python examples/BORE/rosenbrock_bore.py`


## benchmark

Modify the following section of `comparison/xbbo_benchmark.py` :

```python
test_algs = ["bore"]
```
And run `PYTHONPATH='./' python comparison/xbbo_benchmark.py` in the command line.

## Results


### Branin

|   Method   |    Minimum    | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| :--------: | :-----------: | :----------: | :-----------------: | :----------------: | :--------------------: |
| XBBO(bore) | 0.412+/-0.016 |    0.399     |         93.9        |       51.829       |           38           |

