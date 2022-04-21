from abc import ABC, abstractmethod
import time

import numpy as np
import ConfigSpace as CS
from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.core.trials import Trials
from xbbo.utils.constants import Key
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
                 suggest_limit: float = np.inf,
                 total_time_limit: float = np.inf,
                 cost_limit: float = np.inf,
                 learner_time_limit: float = np.inf,
                 budget_limit: float= np.inf,
                 objective_function=None,
                 **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        assert isinstance(space, CS.ConfigurationSpace)
        if not isinstance(space, DenseConfigurationSpace):
            self.space = DenseConfigurationSpace(space,
                                             encoding_cat=encoding_cat,
                                             encoding_ord=encoding_ord)
        else:
            self.space = space
        self.rng = np.random.RandomState(seed)
        self.suggest_limit = suggest_limit
        self.budget_limit = budget_limit
        self.cost_limit = cost_limit
        self.total_time_limit = total_time_limit
        self.learner_time_limit = learner_time_limit
        self.learner_time_recoder = 0
        self.time_limit_recoder = 0
        self.total_time_recoder = 0
        self.suggest_counter = 0
        self.budget_recoder = 0
        self.cost_recoder = 0
        self.objective_function = objective_function

    def fix_boundary(self, individual):
        if self.fix_type == 'random':
            return np.where(
                (individual >= self.bounds.lb) & (individual <= self.bounds.ub),
                individual,
                self.rng.uniform(self.bounds.lb, self.bounds.ub,
                                 self.dimension))  # FIXME
        elif self.fix_type == 'clip':
            return np.clip(individual, self.bounds.lb, self.bounds.ub)

    def suggest(self, n_suggestions=1):
        st = time.time()
        ret = self._suggest(n_suggestions)
        self.total_time_recoder += time.time() - st
        self.suggest_counter += 1
        return ret

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
        learner_train_time = 0
        for trial in trial_list:
            job_info = trial.info
            learner_train_time += job_info.get(Key.EVAL_TIME, 0)
            self.cost_recoder += job_info.get(Key.COST, 0)
            self.budget_recoder += job_info.get(Key.BUDGET, 0)

        st = time.time()
        ret = self._observe(trial_list)
        self.total_time_recoder += time.time() - st + learner_train_time
        self.learner_time_recoder += learner_train_time
        return ret

    def check_stop(self, ):
        if self.learner_time_recoder >= self.learner_time_limit or self.total_time_recoder >= self.total_time_limit or self.suggest_counter >= self.suggest_limit or self.budget_recoder >= self.budget_limit:
            return True
        else:
            return False

    def optimize(self):
        assert self.objective_function is not None
        while not self.check_stop():
            trial_list = self.suggest()
            for trial in (trial_list):
                print('Current suggest count={}.'.format(self.suggest_counter))
                info = trial.info.copy()
                res = self.objective_function(trial.config_dict, **info)
                if not isinstance(res, dict):
                    res = {Key.FUNC_VALUE: res}
                info.update(res)
                trial.add_observe_value(observe_value=info[Key.FUNC_VALUE],
                                        obs_info=info)
            self.observe(trial_list)
