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

## Compare other bbo library

Here you can **comparision** with commonly used and well-known Hyperparameter Optimization (HPO) packages:

[SMAC3](tests/smac3/SMAC3.md)
[hyperopt](tests/hyperopt/hyperopt.md)

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

[review](docs/BBO_paper_reading/Hyper-Parameter_Optimization_A_Review_of_Algorithms_and_Applications.pdf)

## TODO

- [ ] parallel
- [ ] multi-fidelity