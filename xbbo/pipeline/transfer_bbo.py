import numpy as np
import json
from time import time
import tqdm

from xbbo.search_space import problem_register
from xbbo.search_algorithm import alg_register
from xbbo.configspace import build_space
from xbbo.utils.record import Record


class Transfer_BBO:

    def __init__(self, cfg):
        # setup TestProblem
        self.cfg = cfg
        self.function_instance = problem_register[cfg.TEST_PROBLEM.name](cfg)

        self.api_config = self.function_instance.get_api_config()  # 优化的hp
        self.config_spaces = build_space(self.api_config)


        # Setup optimizer
        opt_class = alg_register[cfg.OPTM.name]
        self.optimizer_instance = opt_class(self.config_spaces, **dict(cfg.OPTM.kwargs))

        old_D_x_params, old_D_y, new_D_x_param = self.function_instance.array_to_config(ret_param=True)
        # old_D_x_params, old_D_y, new_D_x_param = self.function_instance.array_to_config()
        # self.function_instance.cache(new_D_x_param)

        self.optimizer_instance.prepare(old_D_x_params, old_D_y, new_D_x_param,
                                        np.argsort(list(self.api_config.keys())), params=True)
        # self.optimizer_instance.prepare(old_D_x_params, old_D_y, new_D_x_param,
        #                                 np.argsort(list(self.api_config.keys())))
        # old_D_x_params, old_D_y, new_D_x_param = self.function_instance.array_to_config()
        # # self.function_instance.cache(new_D_x_param)
        #
        # self.optimizer_instance.prepare(old_D_x_params, old_D_y, new_D_x_param, np.argsort(list(self.api_config.keys())))

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
        self.record = Record(self.cfg)

        # # 预加载观测
        # if history:
        #     self._load_history_and_obs(history)



    # def _load_history_and_obs(self, filename):
    #     obs_x, obs_y, isFeature = load_history(filename)
    #     for i in range(len(obs_x)):
    #         if not isFeature:
    #             obs_x[i] = Configurations.dictUnwarped_to_array(
    #                     self.optimizer_instance.space,
    #                     obs_x[i]
    #             )
    #     obs_x = self.optimizer_instance.transform_sparseArray_to_optSpace(obs_x)
    #     self.optimizer_instance.observe(obs_x, obs_y)
    #     print('成功加载先前观测！')


    def evaluate(self, param):
        return self.function_instance.evaluate(param)

    def run(self):
        pbar = tqdm.tqdm(range(self.n_calls))
        pbar.set_description(f"Optimizer {self.cfg.OPTM.name} is running:")
        
        for ii in pbar:

            tt = time()
            next_points, features = self.optimizer_instance.suggest(self.n_suggestions)  # TODO 1

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

            assert len(next_points) == self.n_suggestions, "invalid number of suggestions provided by the optimizer"
            # eval_time = [None for _ in range(self.n_suggestions)]
            function_evals = []
            losses = []
            tt = time()
            for next_point in (next_points):
                # try:
                f_current_eval, loss =  self.evaluate(next_point) # TODO 2
                # except Exception as e:
                #     f_current_eval = np.full((len(self.cfg.TEST_PROBLEM.func_evals),), np.inf, dtype=float)
                #     loss = np.full((len(self.cfg.TEST_PROBLEM.losses),), np.inf, dtype=float)



                function_evals.append(f_current_eval)
                losses.append(loss)
            eval_time = time() - tt
            eval_list = np.asarray(losses)[:, :self.cfg.OPTM.n_obj].ravel().tolist() # TODO
            assert self.cfg.OPTM.n_obj == 1
            tt = time()
            self.optimizer_instance.observe(features, eval_list)  # TODO 3
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
                         'eval_time_per_suggest': eval_time
            }
            self.record.append(features, losses, function_evals, timing=timing, suggest_point=next_points)


