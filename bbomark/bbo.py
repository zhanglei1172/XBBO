import numpy as np
import json
from time import time


from constants import ITER
from bbomark.model import build_test_problem
from bbomark.configspace import build_space
from bbomark.data.record import Record

class BBO:

    def __init__(
            self,
            opt_class,
            feature_spaces,
            opt_kwargs,
            model_name,
            dataset,
            scorer,
            n_calls,
            n_suggestions,
            data_root=None,
            callback=None):
        self.function_instance = build_test_problem(model_name, dataset, scorer, data_root)
        self.feature_spaces = feature_spaces
        # Setup optimizer
        self.api_config = self.function_instance.get_api_config()  # 优化的hp
        self.config_spaces = build_space(self.api_config)
        self.optimizer_instance = opt_class(self.config_spaces, self.feature_spaces, **opt_kwargs)

        # assert function_instance.objective_names == OBJECTIVE_NAMES
        # assert OBJECTIVE_NAMES[0] == cc.VISIBLE_TO_OPT
        self.n_obj = len(self.function_instance.objective_names)
        self.n_suggestions = n_suggestions
        self.callback = callback

        assert self.n_suggestions >= 1, "batch size must be at least 1"
        assert self.n_obj >= 1, "Must be at least one objective"

        if self.callback is not None:
            # First do initial log at inf score, in case we don't even get to first eval before crash/job timeout
            self.callback(np.full((self.n_obj,), np.inf, dtype=float), 0)

        # self.suggest_time = np.zeros(n_calls)
        # self.observe_time = np.zeros(n_calls)
        # self.eval_time = np.zeros((n_calls, n_suggestions))
        # self.function_evals = np.zeros((n_calls, n_suggestions, self.n_obj))
        # self.suggest_log = [None] * n_calls
        self.n_calls = n_calls
        self.record = Record(n_suggestions=self.n_suggestions)



    def _check(self):
        pass


    def evaluate(self, param):
        return self.function_instance.evaluate(param)

    def run(self):

        for ii in range(self.n_calls):
            # next_points, features = self.optimizer_instance.suggest(self.n_suggestions)  # TODO 1

            tt = time()
            try:
                next_points, features = self.optimizer_instance.suggest(self.n_suggestions)  # TODO 1
            except Exception as e:
                # logger.warning("Failure in optimizer suggest. Falling back to random search.")
                # logger.exception(e, exc_info=True)
                print(json.dumps({"optimizer_suggest_exception": {ITER: ii}}))
                api_config = self.function_instance.get_api_config()
                # TODO 直接随机采样
                x_guess_configs = self.optimizer_instance.space.sample_configuration(size=self.n_suggestions) # a list
                next_points = [x_guess_config.get_dict_unwarped() for x_guess_config in x_guess_configs]
                features = [x_guess_config.get_array() for x_guess_config in x_guess_configs]

            suggest_time = time() - tt

            assert len(next_points) == self.n_suggestions, "invalid number of suggestions provided by the optimizer"
            eval_time = [None for _ in range(self.n_suggestions)]
            function_evals = [None for _ in range(self.n_suggestions)]
            for jj, next_point in enumerate(next_points):
                tt = time()
                try:
                    f_current_eval =  self.evaluate(next_point) # TODO 2
                except Exception as e:

                    f_current_eval = np.full((self.n_obj,), np.inf, dtype=float)
                eval_time[jj] = time() - tt
                assert np.shape(f_current_eval) == (self.n_obj,)

                function_evals[jj] = f_current_eval

            eval_list = np.asarray(function_evals)[:, 0].tolist()

            if self.callback is not None:
                raise NotImplementedError()
                # idx_ii, idx_jj = argmin_2d(function_evals[: ii + 1, :, 0])
                # callback(function_evals[idx_ii, idx_jj, :], ii + 1)

            tt = time()
            try:
                self.optimizer_instance.observe(features, eval_list)  # TODO 3
            except Exception as e:
                # logger.warning("Failure in optimizer observe. Ignoring these observations.")
                # logger.exception(e, exc_info=True)
                print(json.dumps({"optimizer_observe_exception": {ITER: ii}}))
            observe_time = time() - tt
            timing = {
                         'suggest_time': suggest_time,
                         'observe_time': eval_time,
                         'eval_time': observe_time
            }
            self.record.append(features, function_evals, timing=timing, )


