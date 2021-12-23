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
"""
In opentuner, many search techniques are already available. All the names of
the techniques can be found as follows:
```
>>> import opentuner
>>> techniques, generators = opentuner.search.technique.all_techniques()
>>> for t in techniques:
...     print t.name
```
A user can also create new search techniques
(http://opentuner.org/tutorial/techniques/).

Opentuner will create a multi-arm bandit of multiple techniques if more than
one technique is specified in `args.technique`.

Some bandits with pre-defined techniques are already registered in:
`opentuner.search.bandittechniques`

By default, we use a pre-defined bandit called `'AUCBanditMetaTechniqueA'` of 4
techniques:
```
register(AUCBanditMetaTechnique([
        differentialevolution.DifferentialEvolutionAlt(),
        evolutionarytechniques.UniformGreedyMutation(),
        evolutionarytechniques.NormalGreedyMutation(mutation_rate=0.3),
        simplextechniques.RandomNelderMead()],
        name='AUCBanditMetaTechniqueA'))
```
The other two bandits used in our experiments are: PSO_GA_DE and PSO_GA_Bandit.
Specifying a list of multiple techniques will use a multi-arm bandit over them.
"""
import warnings
from argparse import Namespace
import numpy as np

import opentuner.tuningrunmain
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface as DMI
from opentuner.resultsdb.models import DesiredResult, Result
from opentuner.search.manipulator import (
    ConfigurationManipulator,
    EnumParameter,
    FloatParameter,
    IntegerParameter,
    LogFloatParameter,
    LogIntegerParameter,
    ScaledNumericParameter,
)
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter
)

from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration

DEFAULT_TECHNIQUES = ("AUCBanditMetaTechniqueA",)
MEMORY_ONLY_DB = "sqlite://"

# Monkey patch here! Opentuner is messed up, TuningRunMain changes global log
# settings. We should file in issue report here and have them fix it.
opentuner.tuningrunmain.init_logging = lambda: None



class OpentunerOptimizer(AbstractOptimizer):
    primary_import = "opentuner"

    def __init__(self, config_spaces, techniques=DEFAULT_TECHNIQUES, n_suggestions=1):
        """Build wrapper class to use opentuner optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.

        techniques : iterable of strings
            A list or tuple of techniques to use in opentuner. If the list
            has only one technique, then that technique will be used. If the
            list has multiple techniques a bandit over those techniques
            will be used.

        n_suggestions : int
            Default number of suggestions to be made in parallel.
        """
        AbstractOptimizer.__init__(self, config_spaces)
        self.opt_name = 'opentuner'
        # Opentuner requires DesiredResult to reference suggestion when making
        # its observation. x_to_dr maps the dict suggestion to DesiredResult.
        self.x_to_dr = {}
        # Keep last suggested x and repeat it whenever opentuner gives up.
        self.dummy_suggest = None
        self.dummy_features = None

        """Setting up the arguments for opentuner. You can see all possible
        arguments using:
        ```
        >>> import opentuner
        >>> opentuner.default_argparser().parse_args(['-h'])
        ```
        We only change a few arguments (other arguments are set to defaults):
        * database = MEMORY_ONLY_DB: to use an in-memory sqlite database
        * parallelism = n_suggestions: num of suggestions to give in parallel
        * technique = techniques: a list of techniques to be used by opentuner
        * print_params = False: to avoid opentuner from exiting after printing
            param spaces
        """
        args = Namespace(
            bail_threshold=500,
            database=MEMORY_ONLY_DB,
            display_frequency=10,
            generate_bandit_technique=False,
            label=None,
            list_techniques=False,
            machine_class=None,
            no_dups=False,
            parallel_compile=False,
            parallelism=n_suggestions,
            pipelining=0,
            print_params=False,
            print_search_space_size=False,
            quiet=False,
            results_log=None,
            results_log_details=None,
            seed_configuration=[],
            stop_after=None,
            technique=techniques,
            test_limit=5000,
        )
        configs = self.space.get_hyperparameters()
        # feature_space = []
        manipulator = ConfigurationManipulator()
        # Setup some dummy classes required by opentuner to actually run.
        for config in configs:
            if isinstance(config, CategoricalHyperparameter):
                manipulator.add_parameter(EnumParameter(config.name, range(config.num_choices)))
            elif isinstance(config, OrdinalHyperparameter):
                manipulator.add_parameter(EnumParameter(config.name, range(config.num_elements)))
            elif isinstance(config, (UniformFloatHyperparameter,UniformIntegerHyperparameter)):
                manipulator.add_parameter(FloatParameter(config.name, 0, 1))
            # elif isinstance(config, UniformIntegerHyperparameter):
            #     feature_space.append(Integer(config.lower, config.upper, name=config.name, transform="identity"))
            else:
                raise TypeError()
        interface = DMI(args=args, manipulator=manipulator)
        self.api = TuningRunManager(interface, args)

    def transform_sparseArray_to_optSpace(self, sparse_array):
        '''
        opt space is a dict
        '''
        optFeatures = []
        for obs_num in range(len(sparse_array)):
            optFeature = {}
            for i, name in enumerate(self.space.get_hyperparameter_names()):
                optFeature[name] = sparse_array[obs_num][i]
            optFeatures.append(optFeature)
        return optFeatures

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
        # hashable_object = frozenset(d)
        return hashable_object

    def unorder_dict2ord_array(self, keys, dct):
        array = np.zeros(len(keys))
        for i, k in enumerate(keys):
            array[i] = dct[k]
        return array

    # def inv_unorder_dict2ord_array(self, keys, array):
    #     # array = np.zeros(len(keys))
    #     dct = {}
    #     for i, k in enumerate(keys):
    #         dct[k] = array[i]
    #     return dct

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

        # Update the n_suggestions if it is different from the current setting.
        if self.api.search_driver.args.parallelism != n_suggestions:
            self.api.search_driver.args.parallelism = n_suggestions
            warnings.warn("n_suggestions changed across suggest calls")

        # Require the user to already observe all previous suggestions.
        # Otherwise, opentuner will just recycle old suggestions.
        assert len(self.x_to_dr) == 0, "all the previous suggestions should have been observed by now"

        # The real meat of suggest from opentuner: Get next `n_suggestions`
        # unique suggestions.
        desired_results = [self.api.get_next_desired_result() for _ in range(n_suggestions)]

        # Save DesiredResult object in dict since observe will need it.
        X = []
        features = []
        using_dummy_suggest = False
        for ii in range(n_suggestions):
            # Opentuner can give up, but the API requires guessing forever.
            if desired_results[ii] is None:
                assert self.dummy_suggest is not None, "opentuner gave up on the first call!"
                # Use the dummy suggestion in this case.
                X.append(self.dummy_suggest)
                features.append(self.dummy_features)
                using_dummy_suggest = True
                continue

            # Get the simple dict equivalent to suggestion.
            x_guess = desired_results[ii].configuration.data
            dict_unwarped = DenseConfiguration.array_to_dict(
                self.space, self.unorder_dict2ord_array(self.space.get_hyperparameter_names(), x_guess)
            )
            X.append(dict_unwarped)
            features.append(x_guess)

            # Now save the desired result for future use in observe.
            x_guess_ = OpentunerOptimizer.hashable_dict(x_guess)
            assert x_guess_ not in self.x_to_dr, "the suggestions should not already be in the x_to_dr dict"
            self.x_to_dr[x_guess_] = desired_results[ii]
            # This will also catch None from opentuner.
            assert isinstance(self.x_to_dr[x_guess_], DesiredResult)

        assert len(X) == n_suggestions, "incorrect number of suggestions provided by opentuner"
        # Log suggestion for repeating if opentuner gives up next time. We can
        # only do this when it is not already being used since it we will be
        # checking guesses against dummy_suggest in observe.
        if not using_dummy_suggest:
            self.dummy_suggest = X[-1]
            self.dummy_features = features[-1]
        return X, features

    def observe(self, X, y):
        """Feed the observations back to opentuner.

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
            x_guess_ = OpentunerOptimizer.hashable_dict(x_guess)

            # If we can't find the dr object then it must be the dummy guess.
            if x_guess_ not in self.x_to_dr:
                assert x_guess == self.dummy_features, "Appears to be guess that did not originate from suggest"
                continue

            # Get the corresponding DesiredResult object.
            dr = self.x_to_dr.pop(x_guess_, None)
            # This will also catch None from opentuner.
            assert isinstance(dr, DesiredResult), "DesiredResult object not available in x_to_dr"

            # Opentuner's arg names assume we are minimizing execution time.
            # So, if we want to minimize we have to pretend y is a 'time'.
            result = Result(time=y_)
            self.api.report_result(dr, result)


opt_class = OpentunerOptimizer
# feature_space = None