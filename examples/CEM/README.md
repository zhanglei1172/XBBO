# CEM


## Introduction

[The Cross-Entropy Method for Optimization](https://www.sciencedirect.com/science/article/abs/pii/B9780444538598000035)

## Abstract

> The cross-entropy method is a versatile heuristic tool for solving diﬃcult estima-tion and optimization problems, based on Kullback–Leibler (or cross-entropy)minimization. As an optimization method it uniﬁes many existing population-based optimization heuristics. In this chapter we show how the CE method canbe applied to a diverse range of combinatorial, continuous, and noisy optimiza-tion problems.

## Usage

E.g. `PYTHONPATH='./' python examples/CEM/rosenbrock_CEM.py`


## benchmark

Modify the following section of `comparison/xbbo_benchmark.py` :

```python
test_algs = ["cem"]
```
And run `PYTHONPATH='./' python comparison/xbbo_benchmark.py` in the command line.

## Results


### Branin

|   Method  |    Minimum    | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| :--------: | :-----------: | :----------: | :-----------------: | :----------------: | :--------------------: |
| XBBO(cem) | 1.217+/-1.378 |    0.398     |        129.4        |       65.521       |           21           |
