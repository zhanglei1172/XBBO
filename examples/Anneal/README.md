# Anneal


## Introduction

[Courseea Video](https://www.coursera.org/lecture/rengong-zhineng/4-3-simulated-annealing-sf7Nr)

## Abstract (from Wikipedia)

> Simulated annealing (SA) is a probabilistic technique for approximating the global optimum of a given function. Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem. It is often used when the search space is discrete (for example the traveling salesman problem, the boolean satisfiability problem, protein structure prediction, and job-shop scheduling). For problems where finding an approximate global optimum is more important than finding a precise local optimum in a fixed amount of time, simulated annealing may be preferable to exact algorithms such as gradient descent or branch and bound.

> The name of the algorithm comes from annealing in metallurgy, a technique involving heating and controlled cooling of a material to alter its physical properties. Both are attributes of the material that depend on their thermodynamic free energy. Heating and cooling the material affects both the temperature and the thermodynamic free energy or Gibbs energy. Simulated annealing can be used for very hard computational optimization problems where exact algorithms fail; even though it usually achieves an approximate solution to the global minimum, it could be enough for many practical problems.

## Usage

E.g. `PYTHONPATH='./' python examples/Anneal/rosenbrock_anneal.py`


## benchmark

Modify the following section of `comparison/xbbo_benchmark.py` :

```python
test_algs = ["anneal"]
```
And run `PYTHONPATH='./' python comparison/xbbo_benchmark.py` in the command line.

## Results


### Branin

|    Method    |    Minimum    | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
+--------------+---------------+--------------+---------------------+--------------------+------------------------+
| XBBO(anneal) | 0.402+/-0.005 |    0.398     |        166.4        |       33.788       |           81           |