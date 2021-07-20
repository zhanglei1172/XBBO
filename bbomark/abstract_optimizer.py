# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Abstract base class for the optimizers in the benchmark. This creates a common API across all packages.
"""
from abc import ABC, abstractmethod

from importlib_metadata import version

import space

def validate_space(config_spaces):
    for config_space in config_spaces:
        config_space.is_valid_configuration()

class AbstractOptimizer(ABC):
    """Abstract base class for the optimizers in the benchmark. This creates a common API across all packages.
    """

    # Every implementation package needs to specify this static variable, e.g., "primary_import=opentuner"
    primary_import = None

    def __init__(self, api_config, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        # self.api_config = api_config
        self.space = space.build_space(api_config)

    @classmethod
    def get_version(cls):
        """Get the version for this optimizer.

        Returns
        -------
        version_str : str
            Version number of the optimizer. Usually, this is equivalent to ``package.__version__``.
        """
        assert (cls.primary_import is None) or isinstance(cls.primary_import, str)
        # Should use x.x.x as version if sub-class did not specify its primary import
        # version_str = "x.x.x" if cls.primary_import is None else version(cls.primary_import)
        version_str = "1.0.0"
        return version_str

    def suggest(self, n_suggestions): # output [meta param]
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """

        next_guess = [self._suggest() for _ in range(n_suggestions)]
        self._post_suggest(next_guess)

        return next_guess

    @abstractmethod
    def _suggest(self, n_suggestions): # output [meta param]
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        pass


    # def _pre_suggest(self):
    #     """Get a suggestion from the optimizer.
    #
    #     Parameters
    #     ----------
    #     n_suggestions : int
    #         Desired number of parallel suggestions in the output
    #
    #     Returns
    #     -------
    #     next_guess : list of dict
    #         List of `n_suggestions` suggestions to evaluate the objective
    #         function. Each suggestion is a dictionary where each key
    #         corresponds to a parameter being optimized.
    #     """
    #     pass

    def _post_suggest(self, next_guess):
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        validate_space(next_guess)


    def observe(self, X, y): # input [meta param]
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        for xx, yy in zip(X, y):
            xx = self.warp(xx)
            self.space.
            self._observe(xx, yy)


    @abstractmethod
    def _observe(self, X, y): # input [meta param]
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        pass

    def warp(self, parms):
        pass

    def unwarp(self, cs):
        pass
