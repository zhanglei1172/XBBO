# Random Search


## Introduction


## Abstract

>

## Usage

E.g. `PYTHONPATH='./' python examples/RandomSearch/rosenbrock_rs.py`


## benchmark

Modify the following section of `comparison/xbbo_benchmark.py` :

```python
test_algs = ["rs"]
```
And run `PYTHONPATH='./' python comparison/xbbo_benchmark.py` in the command line.

## Results


### Branin

|  Method  |    Minimum    | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| :--------: | :-----------: | :----------: | :-----------------: | :----------------: | :--------------------: |
| XBBO(rs) | 0.618+/-0.217 |    0.421     |        126.7        |       47.221       |           15           |
