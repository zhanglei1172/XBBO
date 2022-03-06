from abc import ABC, abstractmethod

import numpy as np
import ConfigSpace as CS
from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.core.trials import Trials
# from xbbo.configspace.space import Configurations


class AbstractOptimizer(ABC):
    """Abstract base class for the optimizers in the benchmark. This creates a common API across all packages.
    """

    # Every implementation package needs to specify this static variable, e.g., "primary_import=opentuner"
    primary_import = None

    def __init__(self,
                 space: CS.ConfigurationSpace,
                 encoding_cat='round',
                 encoding_ord='round',
                 seed=42,
                 total_time_limit: float = np.inf,
                 learner_time_limit: float = np.inf,
                 **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        assert isinstance(space, CS.ConfigurationSpace)
        self.space = DenseConfigurationSpace(space,
                                             encoding_cat=encoding_cat,
                                             encoding_ord=encoding_ord)
        self.rng = np.random.RandomState(seed)
        self.total_time_limit = total_time_limit
        self.learner_time_limit = learner_time_limit
        self.learner_time_recoder = 0
        self.time_limit_recoder = 0

    def fix_boundary(self, individual):
        if self.fix_type == 'random':
            return np.where(
                (individual > self.bounds.lb) & (individual < self.bounds.ub),
                individual,
                self.rng.uniform(self.bounds.lb, self.bounds.ub,
                                 self.dimension))  # FIXME
        elif self.fix_type == 'clip':
            return np.clip(individual, self.bounds.lb, self.bounds.ub)

    def suggest(self, n_suggestions=1):
        return self._suggest(n_suggestions)

    @abstractmethod
    def _suggest(self, n_suggestions):  # output [meta param]
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
        x_guess_configs = self.space.sample_configuration(size=n_suggestions)
        x_guess = [
            x_guess_config.get_dictionary()
            for x_guess_config in x_guess_configs
        ]
        return x_guess

    def _observe(self, trial_list: Trials):  # input [meta param]
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        features : list of features
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        pass

    def observe(self, trial_list: Trials):
        return self._observe(trial_list)