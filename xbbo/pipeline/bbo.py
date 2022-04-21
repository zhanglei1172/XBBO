import numpy as np
from time import time
import tqdm

from xbbo.search_space import problem_register
from xbbo.search_algorithm import alg_register
from xbbo.configspace import build_space
from xbbo.utils.constants import MAXINT
from xbbo.utils.record import Record


class BBO:

    def __init__(self, cfg, seed):
        # setup TestProblem
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.function_instance = problem_register[cfg.TEST_PROBLEM.name](seed=self.rng.randint(MAXINT), **cfg.TEST_PROBLEM.kwargs)

        self.api_config = self.function_instance.get_api_config()  # 优化的hp
        self.config_spaces = build_space(self.api_config,seed=self.rng.randint(MAXINT))

        # Setup optimizer
        opt_class = alg_register[cfg.OPTM.name]
        self.optimizer_instance = opt_class(self.config_spaces,suggest_limit=cfg.OPTM.max_call,seed=self.rng.randint(MAXINT), **dict(cfg.OPTM.kwargs))


        self.n_suggestions = cfg.OPTM.n_suggestions
        self.n_obj = cfg.OPTM.n_obj

        assert self.n_suggestions >= 1, "batch size must be at least 1"
        assert self.n_obj >= 1, "Must be at least one objective"


        # self.suggest_time = np.zeros(n_calls)
        # self.observe_time = np.zeros(n_calls)
        # self.eval_time = np.zeros((n_calls, n_suggestions))
        # self.function_evals = np.zeros((n_calls, n_suggestions, self.n_obj))
        # self.suggest_log = [None] * n_calls
        self.n_calls = cfg.OPTM.max_call
        self.record = Record(self.cfg.GENERAL.exp_dir)


    def evaluate(self, param):
        return self.function_instance.evaluate(param)

    def run(self):
        pbar = tqdm.tqdm(range(self.n_calls))
        pbar.set_description(f"Optimizer {self.cfg.OPTM.name} is running:")
        
        for ii in pbar:

            tt = time()
            trial_list = self.optimizer_instance.suggest(self.n_suggestions)  # TODO 1

            # try:
            #     next_points, features = self.optimizer_instance.suggest(self.n_suggestions)  # TODO 1
            # except Exception as e:
            #     # logger.warning("Failure in optimizer suggest. Falling back to random search.")
            #     # logger.exception(e, exc_info=True)
            #     print(json.dumps({"optimizer_suggest_exception": {'iter': ii}}))
            #     # api_config = self.function_instance.get_api_config()
            #     # TODO 直接随机采样
            #     x_guess_configs = self.optimizer_instance.space.sample_configuration(size=self.n_suggestions) # a list
            #     next_points = [x_guess_config.get_dict_unwarped() for x_guess_config in x_guess_configs]
            #     features = [x_guess_config.get_array(sparse=False) for x_guess_config in x_guess_configs]

            suggest_time = time() - tt

            assert len(trial_list) == self.n_suggestions, "invalid number of suggestions provided by the optimizer"
            # eval_time = [None for _ in range(self.n_suggestions)]
            function_evals = []
            # losses = []
            for trial in (trial_list):
                # try:
                tt = time()

                f_current_eval = self.evaluate(trial.config_dict) # TODO 2
                eval_time = time() - tt

                # except Exception as e:
                #     f_current_eval = np.full((len(self.cfg.TEST_PROBLEM.func_evals),), np.inf, dtype=float)
                #     loss = np.full((len(self.cfg.TEST_PROBLEM.losses),), np.inf, dtype=float)


                trial.add_observe_value(observe_value=f_current_eval, obs_info={Key.EVAL_TIME:eval_time})
                function_evals.append(f_current_eval)
            # if self.cfg.OPTM.n_obj == 1:
            #     eval_list = np.asarray(function_evals)[:, :self.cfg.OPTM.n_obj].ravel().tolist() # TODO
            # else:
            #     raise NotImplementedError()
                # eval_list = np.array(losses)[:, :self.cfg.OPTM.n_obj].tolist() # TODO
            # assert self.cfg.OPTM.n_obj == 1
            tt = time()
            self.optimizer_instance.observe(trial_list)  # TODO 3
            # try:
            #     self.optimizer_instance.observe(features, eval_list)  # TODO 3
            # except Exception as e:
                # logger.warning("Failure in optimizer observe. Ignoring these observations.")
                # logger.exception(e, exc_info=True)
                # print(json.dumps({"optimizer_observe_exception": {'iter': ii}}))
            observe_time = time() - tt
            timing = {
                         'suggest_time_per_suggest': suggest_time,
                         'observe_time_per_suggest': observe_time,
                         'eval_time_per_suggest': sum(trial.time for trial in trial_list)
            }
            self.record.append([trial.dense_array if trial.sparse_array is None else trial.sparse_array for trial in trial_list], function_evals, timing=timing, suggest_point=[trial.config_dict for trial in trial_list])
            # print(self.optimizer_instance.trials.best_observe_value)
            print(function_evals)

        print(self.optimizer_instance.trials.best_observe_value)


class BBO_REBAR(BBO):
    def __init__(self):
        BBO.__init__(self)
    
    def evaluate(self,params):
        return BBO.evaluate(self, params)
    
    def run(self):
        pass