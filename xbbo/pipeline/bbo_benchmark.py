import os
import pandas as pd
import numpy as np


from xbbo.core.trials import Trial, Trials

from xbbo.search_algorithm import alg_register
from xbbo.utils.constants import MAXINT, Key
from xbbo.utils.util import dumpJson, dumpOBJ


class BBObenchmark:
    def __init__(self, cfg, seed):
        # setup TestProblem
        self.cfg = cfg
        # self.max_budget = cfg.OPTM.max_budget
        # self.min_budget = cfg.OPTM.min_budget
        self.expdir = cfg.GENERAL.exp_dir
        self.out_dir = os.path.join(self.expdir, self.cfg.OPTM.name)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.rng = np.random.RandomState(seed)

        self.problem, self.config_spaces = self._build_problem(
            cfg.TEST_PROBLEM.name,
            seed=self.rng.randint(MAXINT),
            **cfg.TEST_PROBLEM.kwargs)

        # Setup optimizer
        opt_class = alg_register[cfg.OPTM.name]
        self.optimizer_instance = opt_class(
            self.config_spaces,
            seed=self.rng.randint(MAXINT),
            budget_bound=[self.min_budget, self.max_budget],
            **dict(cfg.OPTM.kwargs))

        self.n_suggestions = cfg.OPTM.n_suggestions
        self.n_obj = cfg.OPTM.n_obj

        assert self.n_suggestions >= 1, "batch size must be at least 1"
        assert self.n_obj == 1, "Must one objective"

    def _build_problem(self, problem_name: str, seed: int, **kwargs):
        if problem_name == 'countingones':
            from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes
            problem = CountingOnes(seed)
            n_continuous = kwargs.get("n_continuous", 4)
            n_categorical = kwargs.get("n_categorical", 4)
            cs = problem.get_configuration_space(n_continuous=n_continuous,
                                                 n_categorical=n_categorical)
            dimensions = len(cs.get_hyperparameters())
            self.min_budget = 576 / dimensions
            self.max_budget = 93312 / dimensions
            self.y_star_test = -dimensions
        else:
            pass

        return problem, cs

    def _call_obj(self, trial: Trial, **kwargs):
        budget = kwargs.get('budget')
        r = {}
        if budget is None:
            kwargs['budget'] = self.max_budget
            res = self.problem.objective_function(trial.configuration,
                                                  **kwargs)
        else:
            res = self.problem.objective_function(trial.configuration,
                                                  **kwargs)
        r.update(kwargs)
        r.update(res)
        return r

    def _call_obj_test(self, trial: Trial, **kwargs):
        budget = kwargs.get('budget')
        r = {}
        if budget is None:
            kwargs['budget'] = self.max_budget
            res = self.problem.objective_function_test(trial.configuration,
                                                       **kwargs)
        else:
            res = self.problem.objective_function_test(trial.configuration,
                                                       **kwargs)
        r.update(kwargs)
        r.update(res)
        return r

    def _observe(self, trial_list):
        for trial in (trial_list):
            info = trial.info.copy()
            res = self._call_obj(trial, **info)  # TODO 2
            res_test = self._call_obj_test(trial, **info)  # TODO 2
            info["regret_test"] = res_test["function_value"]
            # info["function_value"] = res["function_value"]
            info.update(res)
            trial.add_observe_value(observe_value=info['function_value'],
                                    obs_info=info)
        self.optimizer_instance.observe(trial_list)  # TODO 3

    def _suggest(self):
        return self.optimizer_instance.suggest(self.n_suggestions)  # TODO 1

    def run_one_exp(self):
        while not self.optimizer_instance.check_stop():
            trial_list = self._suggest()
            self._observe(trial_list)

    def save_to_file(self, run_id):
        trials: Trials = self.optimizer_instance.trials
        dumpOBJ(self.out_dir, 'trials_{}.pkl'.format(run_id), trials)
        res = {}
        tmp = np.minimum.accumulate(trials._his_observe_value)
        res[Key.REGRET_VAL] = tmp.tolist()
        res[Key.REGRET_TEST] = np.array(
            [_dict['regret_test'] for _dict in trials.infos])
        res[Key.REGRET_TEST][1:][np.diff(tmp) == 0] = np.nan
        res[Key.REGRET_TEST] = pd.Series(
            res[Key.REGRET_TEST]).fillna(method='ffill').to_list()
        #  = ([_dict['regret_test'] for _dict in trials.infos]).tolist()
        res[Key.COST] = np.cumsum([_dict['budget']
                                   for _dict in trials.infos]).tolist()

        dumpJson(self.out_dir, 'res_{}.json'.format(run_id), res)


