# Bayesian Optimization


## Introduction

[A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)

## Abstract

> Bayesian optimization is an approach to optimizing objective functions that take a long time (minutes or hours) to evaluate. It is best-suited for optimization over continuous domains of less than 20 dimensions, and tolerates stochastic noise in function evaluations. It builds a surrogate for the objective and quantifies the uncertainty in that surrogate using a Bayesian machine learning technique, Gaussian process regression, and then uses an acquisition function defined from this surrogate to decide where to sample. In this tutorial, we describe how Bayesian optimization works, including Gaussian process regression and three common acquisition functions: expected improvement, entropy search, and knowledge gradient. We then discuss more advanced techniques, including running multiple function evaluations in parallel, multi-fidelity and multi-information source optimization, expensive-to-evaluate constraints, random environmental conditions, multi-task Bayesian optimization, and the inclusion of derivative information. We conclude with a discussion of Bayesian optimization software and future research directions in the field. Within our tutorial material we provide a generalization of expected improvement to noisy evaluations, beyond the noise-free setting where it is more commonly applied. This generalization is justified by a formal decision-theoretic argument, standing in contrast to previous ad hoc modifications.

## Usage

E.g. `PYTHONPATH='./' python examples/BayesianOptimization/rosenbrock_bo.py`


## benchmark

Modify the following section of `comparison/xbbo_benchmark.py` :

```python
test_algs = ["basic-bo"]
```
And run `PYTHONPATH='./' python comparison/xbbo_benchmark.py` in the command line.

## Results


### Branin

|     Method     |    Minimum    | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| :--------: | :-----------: | :----------: | :-----------------: | :----------------: | :--------------------: |
| XBBO(basic-bo) | 0.398+/-0.000 |    0.398     |        158.4        |       37.492       |           95           |

