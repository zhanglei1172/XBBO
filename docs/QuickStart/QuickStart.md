# Quick Start

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
              suggest_limit=MAX_CALL)

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