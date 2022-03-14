import importlib
import os
from ConfigSpace import Configuration
import pandas as pd
import numpy as np

from xbbo.core.trials import Trial, Trials

from xbbo.search_algorithm import alg_register
from xbbo.utils.constants import MAXINT, Key
from xbbo.utils.util import dumpJson, dumpOBJ


class BBObenchmark:
    def __init__(self, cfg, seed):
        self.min_budget = None
        self.max_budget = None
        # setup TestProblem
        self.cfg = cfg
        self.problem_name = cfg.TEST_PROBLEM.name
        # self.max_budget = cfg.OPTM.max_budget
        # self.min_budget = cfg.OPTM.min_budget
        self.expdir = cfg.GENERAL.exp_dir
        self.out_dir = os.path.join(self.expdir, self.cfg.OPTM.name)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.problem, self.config_spaces = self._build_problem(
            self.problem_name,
            seed=seed,
            **cfg.TEST_PROBLEM.kwargs)
        self.reset(seed)

    def reset(self, seed):
        self.rng = np.random.RandomState(seed)
        # Setup optimizer
        if self.cfg.OPTM.name in alg_register:
            opt_class = alg_register[self.cfg.OPTM.name]
        else:
            module = importlib.import_module('ext_algs.{}'.format(self.cfg.OPTM.name))
            opt_class = module.opt_class
        
        
        
        self.optimizer_instance = opt_class(
            self.config_spaces,
            seed=self.rng.randint(MAXINT),
            budget_bound=[self.min_budget, self.max_budget],
            objective_function=self._call_obj,
            **dict(self.cfg.OPTM.kwargs))

        self.n_suggestions = self.cfg.OPTM.n_suggestions
        self.n_obj = self.cfg.OPTM.n_obj

        assert self.n_suggestions >= 1, "batch size must be at least 1"
        assert self.n_obj == 1, "Must one objective"

    def _build_problem(self, problem_name: str, seed: int, **kwargs):
        if problem_name == 'countingones':
            from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes
            problem = CountingOnes(seed)
            n_continuous = kwargs.get("n_continuous", 4)
            n_categorical = kwargs.get("n_categorical", 4)
            cs = problem.get_configuration_space(n_continuous=n_continuous,
                                                 n_categorical=n_categorical,seed=seed)
            dimensions = len(cs.get_hyperparameters())
            self.min_budget = kwargs.get("min_budget", 576 / dimensions)
            self.max_budget = kwargs.get("max_budget", 93312 / dimensions)
            self.y_star_test = -dimensions
            self._objective_function = problem.objective_function
            self._objective_function_test = problem.objective_function_test
        elif problem_name == 'cc18':
            from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
            task_id = kwargs.get("task_id", 189906)
            # self.min_budget = kwargs.get("min_budget", 0.1)
            # self.max_budget = kwargs.get("max_budget", 1)
            problem = Benchmark(task_id=task_id, rng=seed)
            # Parameter space to be used by DE
            cs = problem.get_configuration_space(seed=seed)
            
            def f_trees(config, budget=None, **kwargs):
                if budget is None:
                    budget = self.max_budget
                fidelity = {
                    "n_estimators": np.round(budget).astype(int),
                    "subsample": 1
                }
                res = problem.objective_function(config,
                                                 fidelity,
                                                 rng=self.rng)
                fitness = res[Key.FUNC_VALUE]
                cost = res[Key.COST]
                res[Key.COST] = res['info']['fidelity']["n_estimators"]
                return res

            def f_dataset(config, budget=None, **kwargs):
                if budget is None:
                    budget = self.max_budget
                fidelity = {"n_estimators": 128, "subsample": budget}
                res = problem.objective_function(config,
                                                 fidelity,
                                                 rng=self.rng)
                fitness = res[Key.FUNC_VALUE]
                cost = res[Key.COST]
                res[Key.COST] = res['info']['fidelity']["subsample"]

                return res

            fidelity_kind = kwargs.get("fidelity_kind", "trees")
            if fidelity_kind == "trees":
                self.min_budget = 50
                self.max_budget = 2000
                self._objective_function = f_trees
            else:
                self.min_budget = 0.1
                self.max_budget = 1.0
                self._objective_function = f_dataset
            self._objective_function_test = lambda config: problem.objective_function(
                config, rng=self.rng)
        elif problem_name == 'svm':
            from hpobench.benchmarks.surrogates.svm_benchmark import SurrogateSVMBenchmark
            problem = SurrogateSVMBenchmark(rng=seed)
            self.min_budget = kwargs.get("min_budget", 1 / 128)
            self.max_budget = kwargs.get("max_budget", 1)
            # Parameter space to be used by DE
            cs = problem.get_configuration_space(seed=seed)

            def f(config, budget=None, **kwargs):
                if budget is not None:
                    fidelity = {"dataset_fraction": budget}
                    res = problem.objective_function(config, fidelity=fidelity)
                else:
                    res = problem.objective_function(config)
                res[Key.COST] = res['info']['fidelity']["dataset_fraction"]

                fitness, cost = res[Key.FUNC_VALUE], res[Key.COST]
                return res

            self._objective_function = f
            self._objective_function_test = problem.objective_function_test
        elif problem_name.startswith('nas_101_cifar10'):
            from hpobench.benchmarks.nas.nasbench_101 import NASCifar10ABenchmark, NASCifar10BBenchmark, NASCifar10CBenchmark
            bench_map = {
                "nas_101_cifar10A": NASCifar10ABenchmark,
                "nas_101_cifar10B": NASCifar10BBenchmark,
                "nas_101_cifar10C": NASCifar10CBenchmark
            }
            self.min_budget = 4
            self.max_budget = 108
            problem = bench_map[problem_name](data_dir='./data',
                                              multi_fidelity=True)

            def f(config, budget=None, **kwargs):
                if budget is None:
                    res = problem.objective_function(config)
                else:
                    res = problem.objective_function(
                        config, fidelity={"dataset_fraction": int(budget)})
                    res[Key.COST] = res['info']['fidelity']["dataset_fraction"]
                return res

            self._objective_function = f
            self._objective_function_test = problem.objective_function_test
            cs = problem.get_configuration_space(seed=seed)
            # y_star_valid = b.y_star_valid
            # y_star_test = b.y_star_test
            # inc_config = None
        elif problem_name.startswith('nas_201'):
            from hpobench.benchmarks.nas.nasbench_201 import NasBench201BaseBenchmark
            dataset = kwargs.get("dataset", "cifar10-valid")
            problem = NasBench201BaseBenchmark(dataset=dataset, rng=seed)
            self.min_budget = 1
            self.max_budget = 200

            def f(config, budget=None, **kwargs):
                if budget is None:
                    res = problem.objective_function(config)
                else:
                    res = problem.objective_function(
                        config, fidelity={"epoch": int(budget)})
                    res[Key.COST] = res['info']['fidelity']["epoch"]
                return res

            self._objective_function = f
            self._objective_function_test = problem.objective_function_test
            cs = problem.get_configuration_space(seed=seed)
            # y_star_valid = b.y_star_valid
            # y_star_test = b.y_star_test
            # inc_config = None
        else:
            raise NotImplementedError

        return problem, cs

    def _call_obj(self, trial, **kwargs):
        if isinstance(trial, Trial):
            config = trial.configuration
        elif isinstance(trial, Configuration):
            config = trial
        else:
            raise NotImplementedError
        budget = kwargs.get(Key.BUDGET)
        r = {}
        if budget is None:
            kwargs[Key.BUDGET] = self.max_budget
        res = self._objective_function(config, **kwargs)
        r.update(kwargs)
        r.update(res)
        res_test = self._objective_function_test(config, **kwargs)
        r[Key.REGRET_TEST] = res_test[Key.FUNC_VALUE]
        r[Key.REGRET_VAL] = res[Key.FUNC_VALUE]
        if Key.COST not in r:
            r[Key.COST] = r.get(Key.BUDGET, kwargs[Key.BUDGET])
        return r

    # def _call_obj_test(self, trial: Trial, **kwargs):
    #     budget = kwargs.get(Key.BUDGET)
    #     r = {}
    #     if budget is None:
    #         kwargs[Key.BUDGET] = self.max_budget
    #     res = self._objective_function_test(trial.configuration, **kwargs)
    #     r.update(kwargs)
    #     r.update(res)
    #     return r

    # def _observe(self, trial_list):
    #     for trial in (trial_list):
    #         info = trial.info.copy()
    #         res = self._call_obj(trial, **info)  # TODO 2
    #         # if self.problem_name in ["countingones"]:
    #         # if hasattr(self, "_call_obj_test"):
    #         #     res_test = self._call_obj_test(trial, **info)  # TODO 2
    #         #     info["regret_test"] = res_test["function_value"]
    #         # info["function_value"] = res["function_value"]
    #         info.update(res)
    #         trial.add_observe_value(observe_value=info['function_value'],
    #                                 obs_info=info)
    #     self.optimizer_instance.observe(trial_list)  # TODO 3

    # def _suggest(self):
    #     return self.optimizer_instance.suggest(self.n_suggestions)  # TODO 1

    def run_one_exp(self):
        self.optimizer_instance.optimize()
        # while not self.optimizer_instance.check_stop():
        #     trial_list = self._suggest()
        #     self._observe(trial_list)
        self.trials = self.optimizer_instance.trials

    def save_to_file(self, run_id):
        trials: Trials = self.trials
        if Key.COST in trials.infos[0]:
            cost_key = Key.COST
        else:
            cost_key = Key.BUDGET
        dumpOBJ(self.out_dir, 'trials_{}.pkl'.format(run_id), trials)
        res = {}
        tmp = np.minimum.accumulate(trials._his_observe_value)
        res[Key.REGRET_VAL] = tmp.tolist()
        if Key.REGRET_TEST in trials.infos[0]:
            res[Key.REGRET_TEST] = np.array(
                [_dict[Key.REGRET_TEST] for _dict in trials.infos])
            res[Key.REGRET_TEST][1:][np.diff(tmp) == 0] = np.nan
            res[Key.REGRET_TEST] = pd.Series(
                res[Key.REGRET_TEST]).fillna(method='ffill').to_list()
        #  = ([_dict['regret_test'] for _dict in trials.infos]).tolist()
        res[Key.COST] = np.cumsum([_dict[cost_key]
                                   for _dict in trials.infos]).tolist()

        dumpJson(self.out_dir, 'res_{}.json'.format(run_id), res)
