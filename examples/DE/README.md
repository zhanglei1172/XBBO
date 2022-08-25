# Differential Evolution


## Introduction

[A tutorial on Differential Evolution with Python](https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/)

## Abstract (from Wikipedia)

> In evolutionary computation, differential evolution (DE) is a method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. Such methods are commonly known as metaheuristics as they make few or no assumptions about the problem being optimized and can search very large spaces of candidate solutions. However, metaheuristics such as DE do not guarantee an optimal solution is ever found.

## Usage

E.g. `PYTHONPATH='./' python examples/DE/rosenbrock_de.py`


## benchmark

Modify the following section of `comparison/xbbo_benchmark.py` :

```python
test_algs = ["de"]
```
And run `PYTHONPATH='./' python comparison/xbbo_benchmark.py` in the command line.

## Results


### Branin

|  Method  |    Minimum    | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| :--------: | :-----------: | :----------: | :-----------------: | :----------------: | :--------------------: |
| XBBO(de) | 0.436+/-0.054 |    0.398     |        159.8        |       29.892       |           97           |

