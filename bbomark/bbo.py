import numpy as np
import json
from time import time


from constants import ITER
from bbomark.model import build_test_problem
from bbomark.space import build_space


class BBO:

    def __init__(
            self,
            opt_class,
            opt_kwargs,
            model_name,
            dataset,
            scorer,
            n_calls,
            n_suggestions,
            data_root=None,
            callback=None):
        self.function_instance = build_test_problem(model_name, dataset, scorer, data_root)

        # Setup optimizer
        self.api_config = self.function_instance.get_api_config()  # 优化的hp
        self.config_spaces = build_space(self.api_config)
        self.optimizer_instance = opt_class(self.config_spaces, **opt_kwargs)

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

        self.suggest_time = np.zeros(n_calls)
        self.observe_time = np.zeros(n_calls)
        self.eval_time = np.zeros((n_calls, n_suggestions))
        self.function_evals = np.zeros((n_calls, n_suggestions, self.n_obj))
        self.suggest_log = [None] * n_calls
        self.n_calls = n_calls


    def _check(self):
        pass


    def evaluate(self, param):
        return self.function_instance.evaluate(param)

    def run(self):

        for ii in range(self.n_calls):
            tt = time()
            try:
                next_points = self.optimizer_instance.suggest(self.n_suggestions)  # TODO 1
            except Exception as e:
                # logger.warning("Failure in optimizer suggest. Falling back to random search.")
                # logger.exception(e, exc_info=True)
                print(json.dumps({"optimizer_suggest_exception": {ITER: ii}}))
                api_config = self.function_instance.get_api_config()
                # TODO 直接随机采样
                self.optimizer_instance.space.sample_configuration_and_unwarp(size=self.n_suggestions)

            self.suggest_time[ii] = time() - tt

            assert len(next_points) == self.n_suggestions, "invalid number of suggestions provided by the optimizer"

            for jj, next_point in enumerate(next_points):
                tt = time()
                try:
                    f_current_eval =  self.evaluate(next_point) # TODO 2
                except Exception as e:

                    f_current_eval = np.full((self.n_obj,), np.inf, dtype=float)
                self.eval_time[ii, jj] = time() - tt
                assert np.shape(f_current_eval) == (self.n_obj,)

                self.suggest_log[ii] = next_points
                self.function_evals[ii, jj, :] = f_current_eval

            eval_list = self.function_evals[ii, :, 0].tolist()

            if self.callback is not None:
                raise NotImplementedError()
                # idx_ii, idx_jj = argmin_2d(function_evals[: ii + 1, :, 0])
                # callback(function_evals[idx_ii, idx_jj, :], ii + 1)

            tt = time()
            try:
                self.optimizer_instance.observe(next_points, eval_list)  # TODO 3
            except Exception as e:
                # logger.warning("Failure in optimizer observe. Ignoring these observations.")
                # logger.exception(e, exc_info=True)
                print(json.dumps({"optimizer_observe_exception": {ITER: ii}}))
            self.observe_time[ii] = time() - tt

        self.timing = (self.suggest_time, self.eval_time, self.observe_time)

        return self.function_evals, self.timing, self.suggest_log

