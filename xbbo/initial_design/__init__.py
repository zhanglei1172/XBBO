from typing import DefaultDict
from .default_configuration_design import DefaultConfiguration
from .factorial_design import FactorialInitialDesign
from .latin_hypercube_design import LHDesign
from .random_design import RandomDesign
from .sobol_design import SobolDesign

ALL_avaliable_design = {
    'default': DefaultConfiguration,
    'fac': FactorialInitialDesign,
    'lh':LHDesign,
    'random': RandomDesign,
    'sobol': SobolDesign
}