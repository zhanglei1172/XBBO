<div align="center">

<img src="./docs/_static/XBBO_logo.png" width="200">

<p>
	<a href="https://img.shields.io/badge/Python-%3E%3D3.7-blue"><img src="https://img.shields.io/badge/Python-%3E%3D3.7-blue"></a>
	<a href="https://img.shields.io/badge/License-MIT-brightgreen"><img src="https://img.shields.io/badge/License-MIT-brightgreen"></a>
  <a href="https://xbbo.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/xbbo/badge/?version=latest"></a>
  <!-- <a href="https://img.shields.io/badge/Docs-latest-yellowgreen"><img src="https://img.shields.io/badge/Docs-latest-yellowgreen"></a> -->
</p>
</div>

<br>

**XBBO** is an an effective, modular, reproducible and flexible black-box optimization (BBO) codebase, which aims to provide a common framework and benchmark for the BBO community.

This project is now supported by PengCheng Lab.

---

[**Overview**](#Overview) | [**Links**](#Links) |[**Installation**](#Installation) | [**Quick Start**](#quick-start) | [**Benchmark**](#benchmark) |[**Contributing**](#Contributing) | [**License**](#License)

**For more information and API usages, please refer to our** [**Documentation**](https://xbbo.readthedocs.io).

<br>

## Overview

**XBBO** decouples the **search algorithm** from the **search space** and provides a unified search space interface, allowing developers to focus on the search algorithm.

We provide these black box optimization algorithms as follows:

| Search Spaces  | Docs  | Official Links |
| :------------: | :---: | :------------: |
| BORE optimizer |       |                |
|     Anneal     |       |                |
|       DE       |       |                |
|      CMA       |       |                |
|      NSGA      |       |                |
| Regularized EA |       |                |
|      PBT       |       |                |
|     TuRBO      |       |                |
|     LaMCTS     |       |                |
|   HyperBand    |       |                |
|      BOHB      |       |                |
|      DEHB      |       |                |
|    MFES-BO     |       |                |
|     TST-R      |       |                |
|      TAF       |       |                |
|   TAF(RGPE)    |       |                |
|     RMoGP      |       |                |
|   RGPE(mean)   |       |                |


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

## Quick Start

`note:`XBBO default **minimize** black box function. All examples can be found in `examples/` folder.

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
- [ ] parallel
