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
# from nevergrad import instrumentation as inst
import numpy as np

from xbbo.core import AbstractOptimizer

from xbbo.configspace.space import DenseConfiguration
from xbbo.configspace.feature_space import FeatureSpace_gaussian

class NevergradOptimizer(AbstractOptimizer, FeatureSpace_gaussian):
    primary_import = "nevergrad"

    def __init__(self, config_spaces, tool, budget):
        """Build wrapper class to use nevergrad optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        budget : int
            Expected number of max function evals
        """
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        # self.dtypes_idx_map = self.space.dtypes_idx_map
        # self.random = random
        self.opt_name = 'nevergrad'

        opt_class = optimization.registry[tool]
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.optim = opt_class(dimension=self.dense_dimension, budget=budget)
        self.x = []

    def transform_sparseArray_to_optSpace(self, sparse_array):
        return [self.array_to_feature(x, self.dense_dimension) for x in sparse_array]

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
            x_array = self.feature_to_array(np.asarray(xx), self.sparse_dimension)




            dict_unwarped = DenseConfiguration.array_to_dict(self.space, x_array)
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


opt_class = NevergradOptimizer
# feature_space = FeatureSpace_gaussian
