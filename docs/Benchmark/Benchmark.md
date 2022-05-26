# Benchmark

Run `comparison/xbbo_benchmark.py` to benchmark general BBO optimizer.

| Method        | Minimum       | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| ------------- | ------------- | ------------ | ------------------- | ------------------ | ---------------------- |
| XBBO(rs)      | 0.684+/-0.248 | 0.399        | 110.4               | 60.511             | 17                     |
| XBBO(bo-gp)   | 0.398+/-0.000 | 0.398        | 138.5               | 33.685             | 90                     |
| XBBO(tpe)     | 0.519+/-0.119 | 0.398        | 191.4               | 12.035             | 162                    |
| XBBO(anneal)  | 0.404+/-0.005 | 0.399        | 164.5               | 29.032             | 92                     |
| XBBO(cma-es)  | 0.398+/-0.000 | 0.398        | 191.3               | 8.391              | 174                    |
| XBBO(rea)     | 0.425+/-0.026 | 0.399        | 115.8               | 47.743             | 56                     |
| XBBO(de)      | 0.465+/-0.065 | 0.399        | 163.5               | 27.969             | 99                     |
| XBBO(turbo-1) | 0.398+/-0.000 | 0.398        | 110.3               | 46.596             | 46                     |
| XBBO(turbo-2) | 0.398+/-0.000 | 0.398        | 130.7               | 48.57              | 68                     |
| XBBO(bore)    | 0.408+/-0.006 | 0.401        | 117.4               | 58.114             | 38                     |
| XBBO(cem)     | 1.875+/-2.090 | 0.398        | 144.8               | 60.834             | 36                     |


## Compare other bbo library

Here you can **comparison** with commonly used and well-known Hyperparameter Optimization (HPO) packages:

<!-- [SMAC3](comparison/SMAC3.md) -->

[hyperopt](comparison/hyperopt.md)

[scikit-optimize](comparison/skopt.md)

[TuRBO](comparison/turbo.md)

[Bayesian Optimization](comparison/bayes_opt.md)

[DEHB、HpBandSter](comparison/mf.md)

[OpenBox](comparison/openbox.md)

[Hypermapper](comparison/hypermapper.md)