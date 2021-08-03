from functools import reduce

import numpy as np
import json
from time import time

import matplotlib.pyplot as plt

from bbomark.configspace.space import Configurations
from bbomark.utils.loading import load_history, save_history
from constants import ITER, ALPHA, EVAL_Q
from bbomark.model import build_test_problem
from bbomark.configspace import build_space
from bbomark.data.record import Record
import bbomark.utils.quantiles as qt

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
            history=None,
            # history_dict=None,
            custom_model_dir=None,
            data_root=None,
            callback=None):
        self.function_instance = build_test_problem(model_name, dataset, scorer, data_root, custom_model_dir)
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

        # 预加载观测
        if history:
            self._load_history_and_obs(history)



    def _check(self):
        pass

    def _load_history_and_obs(self, filename):
        obs_x, obs_y, isFeature = load_history(filename)
        for i in range(len(obs_x)):
            if not isFeature:
                obs_x[i] = Configurations.dictUnwarped_to_array(
                        self.optimizer_instance.space,
                        obs_x[i]
                )
        obs_x = self.optimizer_instance.transform_sparseArray_to_optSpace(obs_x)
        self.optimizer_instance.observe(obs_x, obs_y)
        print('成功加载先前观测！')

    def save_as_history(self, filename='../out/history.pkl'):
        feats = np.asarray(self.record.features)
        params = np.asarray(self.record.suggest_log)
        opt_vis_y = np.asarray(self.record.func_evals)[..., 0]
        save_history(
            filename,
            {
                'features': feats.reshape(-1, feats.shape[-1]).tolist(),
                'params': params.reshape(-1).tolist(),
                'y': opt_vis_y.reshape(-1).tolist()
            }
        )
        print('成功保存所有观测！')

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
                # api_config = self.function_instance.get_api_config()
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
            self.record.append(features, function_evals, timing=timing, suggest_log=next_points)

    def summary(self):
        self.coord_y_multi = (np.asarray(self.record.func_evals))
        self.coord_y = self.coord_y_multi[..., 0].mean(axis=-1)

        self.minimum_history = np.minimum.accumulate(self.coord_y_multi, axis=0)
        self.coord_x = np.arange(start=1, stop=len(self.coord_y)+1)
        # self.stat_res = qt.quantile_and_CI(self.coord_y, EVAL_Q, alpha=ALPHA)
        self.cal_history_best()

    def cal_history_best(self):
        idx, feat, res, param = self.record.get_best()
        self.best_feature = feat
        self.best_target = res
        self.best_param = param
        self.best_idx = idx

    def visualize(self, ax=None):

        self.summary()
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        else:
            fig = None
        opt_name = self.optimizer_instance.opt_name


        # plt.fill_between(
        #     self.coord_x,
        #     self.stat_res[1],
        #     self.stat_res[2],
        #     # color=opt_name,
        #     alpha=0.5,
        # )
        line = ax.plot(
            self.coord_x,
            self.coord_y,
            # color=opt_name,
            label=opt_name,
            marker=".",
            alpha=0.6
        )

        ax.scatter(
            x=self.best_idx+1,
            y=self.best_target,
            s=20,
            marker='*',
            c=line[0].get_color()
        )
        ax.set_xlabel("evaluation", fontsize=10)
        # plt.ylabel("normalized median score", fontsize=10)
        ax.set_ylabel("loss", fontsize=10)
        ax.grid()
        if fig:
            ax.set_title(opt_name)
            ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
            fig.show()


        return ax
        # ax.show()
