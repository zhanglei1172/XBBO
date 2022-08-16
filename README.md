<div align="center">

<img src="./docs/_static/XBBO_logo.png" width="200">

<p>
	<a href="https://img.shields.io/badge/Python-%3E%3D3.7-blue"><img src="https://img.shields.io/badge/Python-%3E%3D3.7-blue"></a>
	<a href="https://img.shields.io/badge/License-MIT-brightgreen"><img src="https://img.shields.io/badge/License-MIT-brightgreen"></a>
      <a href="https://pypi.org/project/XBBO/"><img src="https://img.shields.io/badge/PyPI-XBBO-yellowgreen.svg"></a>
  <a href="https://xbbo.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/xbbo/badge/?version=latest"></a>
  <!-- <a href="https://img.shields.io/badge/Docs-latest-yellowgreen"><img src="https://img.shields.io/badge/Docs-latest-yellowgreen"></a> -->
</p>
</div>

<br>

**XBBO** is an an effective, modular, reproducible and flexible black-box optimization (BBO) codebase, which aims to provide a common framework and benchmark for the BBO community.

This project is now supported by PengCheng Lab.

---

[**Overview**](#overview) | [**Links**](#links) |[**Installation**](#installation) | [**Quick Start**](#quick-start) | [**Benchmark**](#benchmark) |[**Contributing**](#contributing) | [**License**](#license)

**For more information and API usages, please refer to our** [**Documentation**](https://xbbo.readthedocs.io).

<br>

## Overview

**XBBO** decouples the **search algorithm** from the **search space** and provides a unified search space interface, allowing developers to focus on the search algorithm.

We provide these black box optimization algorithms as follows:

|    Search Algorithm    | Docs  |                 Official Links                  | multi-fideility | transfer | multi-obj |
| :--------------------: | :---: | :---------------------------------------------: | :-------------: | :------: | :-------: |
|         Random         |       |                                                 |                 |          |           |
|          BORE          |       | [ltiao/bore](https://github.com/ltiao/bore) |                 |          |           |
|         Anneal         |       |                                                 |                 |          |           |
| Diffenential Evolution |       |                                                 |                 |          |           |
|         CMA-ES         |       |                                                 |                 |          |           |
|        NSGA-II         |       |                                                 |                 |          |     √     |
|     Regularized EA     |       |                                                 |                 |          |           |
|          PBT           |       |                                                 |                 |          |           |
|         TuRBO          |       |                                                 |                 |          |           |
|         LaMCTS         |       |                                                 |                 |          |           |
|       HyperBand        |       |                                                 |        √        |          |           |
|          BOHB          |       |                                                 |        √        |          |           |
|          DEHB          |       |                                                 |        √        |          |           |
|        MFES-BO         |       |                                                 |        √        |          |           |
|         TST-R          |       |                                                 |                 |    √     |           |
|          TAF           |       |                                                 |                 |    √     |           |
|       TAF(RGPE)        |       |                                                 |                 |    √     |           |
|         RMoGP          |       |                                                 |                 |    √     |           |
|       RGPE(mean)       |       |                                                 |                 |    √     |           |
|          PSO           |       |                                                 |                 |          |           |
|          XNES          |       |                                                 |                 |          |           |
|          LFBO          |       | [lfbo-ml/lfbo](https://github.com/lfbo-ml/lfbo) |                 |          |           |

# Links

- [Documentation](https://xbbo.readthedocs.io)
- [Pypi package](https://pypi.org/project/XBBO/)

## Installation

`Python >= 3.7` is required.

### Installation from PyPI

To install XBBO from [PyPI](https://pypi.org/project/XBBO/):

```bash
pip install xbbo
```

For detailed instructions, please refer to [**Installation.md**](./docs/Installation/Installation.md)

## Search Space

XBBO uses **ConfigSpace** as a tool to define search space. **Please see [ConfigSpace](https://automl.github.io/ConfigSpace/master/API-Doc.html) for how to define a search space.**

## Quick Start

`note:`XBBO default **minimize** black box function. All examples can be found in `examples/` folder.


Here we take optimizing a quadratic function as a toy example:

```python
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from xbbo.search_algorithm.bo_optimizer import BO

def custom_black_box_func(config):
    '''
    define black box function:
    y = x^2
    '''
    return config['x'] ** 2

def custom_search_space():
    '''
    define search space
    '''
    configuration_space = ConfigurationSpace()
    configuration_space.add_hyperparameter(UniformFloatHyperparameter('x', -10, 10, default_value=-3))
    return configuration_space

if __name__ == "__main__":
    MAX_CALL = 30

    cs = custom_search_space()

    # specify black box optimizer
    hpopt = BO(space=cs, suggest_limit=MAX_CALL)
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest() # defalut suggest one trial
        # evaluate 
        obs = custom_black_box_func(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(obs)
        hpopt.observe(trial_list=trial_list)
        
        print(obs)
    
    print('find best (value, config):{}'.format(hpopt.trials.get_best()))
```

Please refer to [**Quick Start.md**](./docs/QuickStart/QuickStart.md) for more information.

## Benchmark

XBBO provides an easy-to-use benchmark tool, users can easily and quickly test the performance of the variety black-box algorithms on each test problem. Clik [**here**](./docs/Benchmark/Benchmark.md) for more information.


## Contributing

We welcome contributions to the library along with any potential issues or suggestions.

Please refer to [**Contributing.md**](./docs/Contributing/Contributing.md) in our docs for more information.

## License

This project is released under the [MIT license](https://mit-license.org).

## TODO

- [ ] 文档完善
- [ ] Logger
- [ ] parallel
