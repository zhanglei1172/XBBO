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
# from bbomark import np_util
from bbomark.core.abstract_optimizer import AbstractOptimizer
from bbomark.optimizers.feature_space import FeatureSpace_nevergrad

class RandomOptimizer(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bbomark"

    def __init__(self, config_spaces, feature_spaces, random=None):
        self.opt_name = 'radom-search'
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        super().__init__(config_spaces, feature_spaces)
        self.random = random

    def transform_sparseArray_to_optSpace(self, sparse_array):
        return sparse_array

    def suggest(self, n_suggestions=1):
        """Get suggestion.

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
        # x_guess = rs.suggest_dict([], [], self.configspace, n_suggestions=n_suggestions, random=self.random)
        x_guess_configs = self.space.sample_configuration(size=n_suggestions)
        x_guess = [x_guess_config.get_dict_unwarped() for x_guess_config in x_guess_configs]
        features = [x_guess_config.get_array() for x_guess_config in x_guess_configs]        # x_guess = self.configspace.sample_configuration_and_unwarp(size=n_suggestions)
        # x_guess = self.configspace.sample_configuration_and_unwarp(size=n_suggestions)
        return x_guess, features

    def observe(self, features, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Random search so don't do anything
        pass


# All optimizer wrappers need to assign their wrapper to the name opt_wrapper because experiment always tries to import
# opt_wrapper regardless of the optimizer it is importing.
opt_wrapper = RandomOptimizer
feature_space = FeatureSpace_nevergrad