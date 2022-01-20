# XBBO

**XBBO** is an an effective, modular and flexible black-box optimization (BBO) codebase, which aims to provide a common framework and benchmark for the BBO community.

## Installation

`Python >= 3.8` is required.

```bash
git clone REPO_URL
cd XBBO
# install requirements
pip install -r ./requirements.txt
# set root path
export PYTHONPATH=$PYTHONPATH:/Path/to/XBBO
```

## Quick Start

### Bayesian Optimization test

`python ./examples/rosenbrock_bo.py`

`note:`XBBO default **minimize** black box function.

```python
def build_space(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
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
hpopt = BO(config_spaces=cs, seed=rng.randint(10000), total_limit=MAX_CALL)
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

## Benchmark

Run `tests/xbbo_benchmark.py` to benchmark general BBO optimizer.

| Method        | Minimum       | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| ------------- | ------------- | ------------ | ------------------- | ------------------ | ---------------------- |
| XBBO(rs)      | 0.684+/-0.248 | 0.399        | 110.4               | 60.511             | 17                     |
| XBBO(bo-gp)   | 0.398+/-0.000 | 0.398        | 42.0                | 5.0398             | 30                     |
| XBBO(tpe)     | 0.519+/-0.119 | 0.398        | 191.4               | 12.035             | 162                    |
| XBBO(anneal)  | 0.403+/-0.004 | 0.398        | 161.1               | 18.839             | 126                    |
| XBBO(rea)     | 0.425+/-0.026 | 0.399        | 115.8               | 47.743             | 56                     |
| XBBO(de)      | 0.412+/-0.038 | 0.398        | 148.2               | 35.21              | 100                    |
| XBBO(turbo-1) | 0.398+/-0.000 | 0.398        | 110.3               | 46.596             | 46                     |
| XBBO(turbo-2) | 0.398+/-0.000 | 0.398        | 130.7               | 48.57              | 68                     |
| XBBO(bore)    | 0.408+/-0.006 | 0.401        | 117.4               | 58.114             | 38                     |

## Compare other bbo library

Here you can **comparison** with commonly used and well-known Hyperparameter Optimization (HPO) packages:

[SMAC3](tests/smac3/SMAC3.md)

[hyperopt](tests/hyperopt/hyperopt.md)

[scikit-optimize](tests/scikit_optimize/skopt.md)

[TuRBO](tests/turbo/turbo.md)

[Bayesian Optimization](tests/BayesianOptimization/bayes_opt.md)

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
- [ ] multi-fidelity