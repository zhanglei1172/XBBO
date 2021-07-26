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
from hyperopt import hp, tpe
from hyperopt.base import JOB_STATE_DONE, JOB_STATE_NEW, STATUS_OK, Domain, Trials
from scipy.interpolate import interp1d
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter
)

from bbomark.core import AbstractOptimizer
from bbomark.utils.util import random as np_random
from bbomark.utils.util import random_seed
from bbomark.configspace.space import Configurations

# Sklearn prefers str to unicode:
DTYPE_MAP = {"real": float, "int": int, "bool": bool, "cat": str, "ordinal": str}


def dummy_f(x):
    assert False, "This is a placeholder, it should never be called."


def only(x):
    y, = x
    return y


class HyperoptOptimizer(AbstractOptimizer):
    primary_import = "hyperopt"

    def __init__(self, config_spaces, feature_spaces, random=np_random):
        """Build wrapper class to use hyperopt optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        super().__init__(config_spaces, feature_spaces)
        self.opt_name = 'hyperopt'

        self.random = random
        feature_space = {}
        configs = self.space.get_hyperparameters()
        for config in configs:
            if isinstance(config, CategoricalHyperparameter):
                feature_space[config.name] = (hp.choice(config.name, range(config.num_choices)))
            elif isinstance(config, OrdinalHyperparameter):
                feature_space[config.name] = (hp.choice(config.name, range(config.num_elements)))
            elif isinstance(config, (UniformFloatHyperparameter,UniformIntegerHyperparameter)):
                feature_space[config.name] = (hp.uniform(config.name, 0, 1))
            # elif isinstance(config, UniformIntegerHyperparameter):
            #     feature_space.append(Integer(config.lower, config.upper, name=config.name, transform="identity"))
            else:
                raise TypeError()
        self.domain = Domain(dummy_f, feature_space, pass_expr_memo_ctrl=None)
        self.trials = Trials()

        # Some book keeping like opentuner wrapper
        self.trial_id_lookup = {}

        # Store just for data validation
        self.param_set_chk = frozenset(self.space.get_hyperparameter_names())

    @staticmethod
    def hashable_dict(d):
        """A custom function for hashing dictionaries.

        Parameters
        ----------
        d : dict or dict-like
            The dictionary to be converted to immutable/hashable type.

        Returns
        -------
        hashable_object : frozenset of tuple pairs
            Bijective equivalent to dict that can be hashed.
        """
        hashable_object = frozenset(d.items())
        return hashable_object

    def get_trial(self, trial_id):
        for trial in self.trials._dynamic_trials:
            if trial["tid"] == trial_id:
                assert isinstance(trial, dict)
                # Make sure right kind of dict
                assert "state" in trial and "result" in trial
                assert trial["state"] == JOB_STATE_NEW
                return trial
        assert False, "No matching trial ID"

    def cleanup_guess(self, x_guess):
        assert isinstance(x_guess, dict)
        # Also, check the keys are only the vars we are searching over:
        assert frozenset(list(x_guess.keys())) == self.param_set_chk

        # Do the rounding
        # Make a copy to be safe, and also unpack singletons
        # We may also need to consider clip_chk at some point like opentuner
        x_guess = {k: only(x_guess[k]) for k in x_guess}
        # for param_name, round_f in self.round_to_values.items():
        #     x_guess[param_name] = round_f(x_guess[param_name])
        # Also ensure this is correct dtype so sklearn is happy
        # x_guess = {k: DTYPE_MAP[self.api_config[k]["type"]](x_guess[k]) for k in x_guess}
        return x_guess

    def _suggest(self):
        """Helper function to `suggest` that does the work of calling
        `hyperopt` via its dumb API.
        """
        new_ids = self.trials.new_trial_ids(1)
        assert len(new_ids) == 1
        self.trials.refresh()

        seed = random_seed(self.random)
        new_trials = tpe.suggest(new_ids, self.domain, self.trials, seed)
        assert len(new_trials) == 1

        self.trials.insert_trial_docs(new_trials)
        self.trials.refresh()

        new_trial, = new_trials  # extract singleton
        return new_trial

    def suggest(self, n_suggestions=1):
        """Make `n_suggestions` suggestions for what to evaluate next.

        This requires the user observe all previous suggestions before calling
        again.

        Parameters
        ----------
        n_suggestions : int
            The number of suggestions to return.

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        assert n_suggestions >= 1, "invalid value for n_suggestions"

        # Get the new trials, it seems hyperopt either uses random search or
        # guesses one at a time anyway, so we might as welll call serially.
        new_trials = [self._suggest() for _ in range(n_suggestions)]

        X = []
        features = []
        for trial in new_trials:
            x_guess = self.cleanup_guess(trial["misc"]["vals"])
            dict_unwarped = Configurations.array_to_dictUnwarped(
                self.space, self.unorder_dict2ord_array(self.space.get_hyperparameter_names(), x_guess)
            )
            X.append(dict_unwarped)
            features.append(x_guess)

            # Build lookup to get original trial object
            x_guess_ = HyperoptOptimizer.hashable_dict(x_guess)
            assert x_guess_ not in self.trial_id_lookup, "the suggestions should not already be in the trial dict"
            self.trial_id_lookup[x_guess_] = trial["tid"]

        assert len(X) == n_suggestions
        return X, features

    def unorder_dict2ord_array(self, keys, dct):
        array = np.zeros(len(keys))
        for i, k in enumerate(keys):
            array[i] = dct[k]
        return array

    def observe(self, X, y):
        """Feed the observations back to hyperopt.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated.
        """
        assert len(X) == len(y)

        for x_guess, y_ in zip(X, y):
            x_guess_ = HyperoptOptimizer.hashable_dict(x_guess)
            assert x_guess_ in self.trial_id_lookup, "Appears to be guess that did not originate from suggest"

            trial_id = self.trial_id_lookup.pop(x_guess_)
            trial = self.get_trial(trial_id)
            assert self.cleanup_guess(trial["misc"]["vals"]) == x_guess, "trial ID not consistent with x values stored"

            # Cast to float to ensure native type
            result = {"loss": float(y_), "status": STATUS_OK}
            trial["state"] = JOB_STATE_DONE
            trial["result"] = result
        # hyperopt.fmin.FMinIter.serial_evaluate only does one refresh at end
        # of loop of a bunch of evals, so we will do the same thing here.
        self.trials.refresh()


opt_wrapper = HyperoptOptimizer
feature_space = None