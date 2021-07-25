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
import nevergrad.optimization as optimization
import numpy as np
# from nevergrad import instrumentation as inst
from scipy.stats import norm

from bbomark.core import AbstractOptimizer

from bbomark.configspace.space import Configurations
from bbomark.optimizers.feature_space import FeatureSpace_nevergrad

class NevergradOptimizer(AbstractOptimizer):
    primary_import = "nevergrad"

    def __init__(self, config_spaces, feature_spaces, tool, budget):
        """Build wrapper class to use nevergrad optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        budget : int
            Expected number of max function evals
        """
        super().__init__(config_spaces, feature_spaces)
        # self.random = random
        opt_class = optimization.registry[tool]
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.optim = opt_class(dimension=self.dense_dimension, budget=budget)
        self.x = []


    def suggest(self, n_suggestions=1):
        """Get suggestion from nevergrad.

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
        x_guess_data = [self.optim.ask() for _ in range(n_suggestions)]

        x_guess = [None] * n_suggestions
        for ii, xx in enumerate(x_guess_data):
            x_array = self.feature_spaces.feature_to_array(xx, self.sparse_dimension)




            dict_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            x_guess[ii] = dict_unwarped

        return x_guess, x_guess_data

    def observe(self, features, y):
        """Feed an observation back to nevergrad.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        for xx, yy in zip(features, y):
            self.optim.tell(xx, yy)


opt_wrapper = NevergradOptimizer
feature_space = FeatureSpace_nevergrad
