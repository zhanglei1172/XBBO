# XBBO

**XBBO** is an an effective, modular, reproducible and flexible black-box optimization (BBO) codebase, which aims to provide a common framework and benchmark for the BBO community.

## Installation

`Python >= 3.7` is required.

### For pip

```bash
pip install xbbo
```

### For Development

```bash
git clone REPO_URL
cd XBBO
# install requirements
pip install -r ./requirements.txt
# pip install -r ./requirements_optional.txt 
# set root path
export PYTHONPATH=$PYTHONPATH:/Path/to/XBBO
```

Install full version:

```bash
conda install gxx_linux-64 gcc_linux-64 swig
pip install 'xbbo[dev]'
```

## Quick Start

`note:`XBBO default **minimize** black box function.

### Bayesian Optimization example

Script path is `./examples/optimize_api_rosenbrock_bo.py`

```python
import numpy as np

from xbbo.search_space.fast_example_problem import build_space_hard, rosenbrock_2d_hard
from xbbo.search_algorithm.bo_optimizer import BO
from xbbo.utils.constants import MAXINT

if __name__ == "__main__":
  MAX_CALL = 30
  rng = np.random.RandomState(42)

  # define black box function
  blackbox_func = rosenbrock_2d_hard
  # define search space
  cs = build_space_hard(rng)
  # define black box optimizer
  hpopt = BO(space=cs,
              objective_function=blackbox_func,
              seed=rng.randint(MAXINT),
              suggest_limit=MAX_CALL,
              initial_design='sobol',
              surrogate='gp',
              acq_opt='rs_ls')

  # ---- Use minimize API ----
  hpopt.optimize()
  best_value, best_config = hpopt.trials.get_best()
  print('Find best value:{}'.format(best_value))
  print('Best Config:{}'.format(best_config))
```

> This example shows how to use this `.optimize()` api to easily and quickly optimize a black box function.

Script path is `./examples/rosenbrock_bo.py`


```python
def build_space(rng):
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])
    return cs

rng = np.random.RandomState(42)
# define black box function
blackbox_func = rosenbrock_2d
# define search space
cs = build_space(rng)
# define black box optimizer
hpopt = BO(config_spaces=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL)
# Example call of the black-box function
def_value = blackbox_func(cs.get_default_configuration())
print("Default Value: %.2f" % def_value)
# ---- Begin BO-loop ----
for i in range(MAX_CALL):
    # suggest
    trial_list = hpopt.suggest()
    # evaluate 
    value = blackbox_func(trial_list[0].config_dict)
    # observe
    trial_list[0].add_observe_value(observe_value=value)
    hpopt.observe(trial_list=trial_list)
  
    print(value)  
```

> This example shows how to use `.ask()`、`.tell()` api to quickly optimize a black box function.

All examples can be found in `examples/` folder.

## Supported Algorithms

- [X] Transfer

  - [X] TST-R
  - [X] TAF
  - [X] TAF(RGPE)
  - [X] RMoGP
  - [X] RGPE(mean)
- [X] Optimizer

  - [X] BORE optimizer
  - [X] Anneal
  - [X] DE
  - [X] CMA
  - [X] NSGA
  - [X] Regularized EA
  - [X] PBT
  - [X] TuRBO
- [X] multi-fidelity

  - [X] HyperBand
  - [X] BOHB
  - [X] DEHB
  - [x] MFES-BO

## Benchmark

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

[SMAC3](comparison/smac3/SMAC3.md)

[hyperopt](comparison/hyperopt/hyperopt.md)

[scikit-optimize](comparison/scikit_optimize/skopt.md)

[TuRBO](comparison/turbo/turbo.md)

[Bayesian Optimization](comparison/BayesianOptimization/bayes_opt.md)

[DEHB、HpBandSter](comparison/multifidelity/mf.md)

[OpenBox](comparison/openbox/openbox.md)

[Hypermapper](comparison/hypermapper/hypermapper.md)

## Algorithms notes

- [BORE](docs/BBO_paper_reading/BORE_BayesianOptimization_by_Density-Ratio_Estimation.pdf)
- [CEM](docs/BBO_paper_reading/cem.md)
- [CMA-ES](docs/BBO_paper_reading/cma-es.md)
- [DE](docs/BBO_paper_reading/de.md)
- [NSGA-II](docs/BBO_paper_reading/NSGA.md)
- [PBT](docs/BBO_paper_reading/pbt.md)
- [REA](docs/BBO_paper_reading/rea.md)
- [TuRBO](docs/BBO_paper_reading/Scalable_Global_Optimization_via_Local_Bayesian_Optimization.pdf)
- [TPE](docs/BBO_paper_reading/toy_tpe.pdf)
- [TST-R](docs/BBO_paper_reading/Two-stage_transfer_surrogate_model_for_automatic_hyperparameter_optimization.pdf)
- [RGPE、RMoGP、TAF](docs/BBO_paper_reading/Practical_Transfer_Learning_for_Bayesian_Optimization.pdf)
- [TAF](docs/BBO_paper_reading/Transfer_Bayesian_Optimization.pdf)
- [HyperBand](docs/BBO_paper_reading/Hyperband.pdf)
- [BOHB](docs/BBO_paper_reading/BOHB_Robust_and_Efficient_Hyperparameter_Optimization_at_Scale.pdf)

[review](docs/BBO_paper_reading/Hyper-Parameter_Optimization_A_Review_of_Algorithms_and_Applications.pdf)

## TODO

- [ ] parallel
- [X] multi-fidelity
