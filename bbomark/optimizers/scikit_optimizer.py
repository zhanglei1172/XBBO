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
import numpy as np
from scipy.interpolate import interp1d
from skopt import Optimizer as SkOpt
from skopt.space import Categorical, Integer, Real
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter
)

from bbomark.core.abstract_optimizer import AbstractOptimizer
from bbomark.configspace.space import Configurations

class ScikitOptimizer(AbstractOptimizer):
    primary_import = "scikit-optimize"

    def __init__(self, config_spaces, feature_spaces, base_estimator="GP", acq_func="gp_hedge", n_initial_points=5, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        base_estimator : {'GP', 'RF', 'ET', 'GBRT'}
            How to estimate the objective function.
        acq_func : {'LCB', 'EI', 'PI', 'gp_hedge', 'EIps', 'PIps'}
            Acquisition objective to decide next suggestion.
        n_initial_points : int
            Number of points to sample randomly before actual Bayes opt.
        """
        self.opt_name = 'scikit-optimize'
        super().__init__(config_spaces, feature_spaces)
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        # self.param_names = self.space.get_hyperparameter_names()
        # self.all_types = self.space.warp.all_types
        configs = self.space.get_hyperparameters()
        feature_space = []
        for config in configs:
            if isinstance(config, CategoricalHyperparameter):
                feature_space.append(Categorical(range(config.num_choices), name=config.name, transform="label"))
            elif isinstance(config, OrdinalHyperparameter):
                feature_space.append(Categorical(range(config.num_elements), name=config.name, transform="label"))
            elif isinstance(config, (UniformFloatHyperparameter,UniformIntegerHyperparameter)):
                feature_space.append(Real(0, 1, name=config.name, transform="identity"))
            # elif isinstance(config, UniformIntegerHyperparameter):
            #     feature_space.append(Integer(config.lower, config.upper, name=config.name, transform="identity"))
            else:
                raise TypeError()

        # for name in self.param_names:
        #     if self.all_types[name] == 'cat':
        #         feature_space.append(Categorical(self.space.get, name, transform="label"))

        # Older versions of skopt don't copy over the dimensions names during
        # normalization and hence the names are missing in
        # self.skopt.space.dimensions. Therefore, we save our own copy of
        # dimensions list to be safe. If we can commit to using the newer
        # versions of skopt we can delete self.dimensions.

        # Undecided where we want to pass the kwargs, so for now just make sure
        # they are blank
        assert len(kwargs) == 0

        self.skopt = SkOpt(
            feature_space,
            n_initial_points=n_initial_points,
            base_estimator=base_estimator,
            acq_func=acq_func,
            acq_optimizer="auto",
            acq_func_kwargs={},
            acq_optimizer_kwargs={},
        )

    def transform_sparseArray_to_optSpace(self, sparse_array):
        return sparse_array

    def suggest(self, n_suggestions=1):
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
        # First get list of lists from skopt.ask()
        features = self.skopt.ask(n_points=n_suggestions)
        # Then convert to list of dicts

        # Now do the rounding, custom rounding is not supported in skopt. Note
        # that there is not nec a round function for each dimension here.
        x_guess = [None] * n_suggestions
        for ii, xx in enumerate(features):
            dict_unwarped = Configurations.array_to_dictUnwarped(self.space, xx)
            x_guess[ii] = dict_unwarped
        return x_guess, features

    def observe(self, features, y):
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
        # Supposedly skopt can handle blocks, but not sure about interface for
        # that. Just do loop to be safe for now.
        for xx, yy in zip(features, y):
            # Just ignore, any inf observations we got, unclear if right thing
            if np.isfinite(yy):
                if isinstance(xx, np.ndarray):
                    xx = xx.tolist()
                self.skopt.tell(xx, yy)


opt_wrapper = ScikitOptimizer
feature_space = None