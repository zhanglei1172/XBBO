import collections
from glob import glob
import os, yaml
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import rcParams

from xbbo.core.trials import Trial, Trials

from xbbo.search_algorithm import alg_register
from xbbo.utils.constants import MAXINT, Key
from xbbo.utils.util import dumpJson, dumpOBJ, loadJson


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


class Analyze():
    def __init__(self,
                 exp_dir_root='./exp',
                 benchmark='countingones',
                 methods=None,
                 limit=1e7,
                 **kwargs) -> None:
        self.exp_dir_root = exp_dir_root
        self.methods = methods
        self.benchmark = benchmark
        self._set_plot()
        plt.clf()
        hashset = set()
        min_cost = np.inf
        max_cost = 0
        self.min_regret = 1
        self.max_regret = 0
        self.regret_key = Key.REGRET_TEST
        # new first
        cfg_paths = sorted(glob(exp_dir_root + '/*/*.yaml'), reverse=True)
        mean_df = {}
        std_df = {}
        index = -1
        for cfg_path in cfg_paths:  # for every method
            with open(cfg_path, 'r') as f:
                cfg = yaml.load(f)
            # hash_name = cfg_path.split('/')[-2].split('_')[-1]
            mark_label = cfg["mark_label"]
            method_name = cfg["OPTM"]["name"]  # tmp[-1][:-5]
            if cfg["TEST_PROBLEM"]["name"] != self.benchmark or (
                    self.methods is None or method_name
                    not in self.methods) or mark_label in hashset:
                continue
            hashset.add(mark_label)
            index += 1

            regret = []
            costs = []
            for jfile in glob('{}/{}/*.json'.format(cfg["GENERAL"]["exp_dir"],
                                                    method_name)):
                j = loadJson(jfile)
                curr_regret = np.array(j[self.regret_key])
                curr_cost = np.array(j[Key.COST])
                if self.benchmark == "countingones":
                    d = cfg["TEST_PROBLEM"].get("n_continuous",
                                                4) + cfg["TEST_PROBLEM"].get(
                                                    "n_categorical", 4)
                    curr_regret = (curr_regret + d) / d  # 0-1
                _, idx = np.unique(curr_regret, return_index=True)
                idx.sort()

                regret.append(curr_regret[idx])
                costs.append(curr_cost[idx])
            # regret = np.array(regret)
            # costs = np.array(costs)

            # finds the latest time where the first measurement was made across runs
            t = np.max([costs[i][0] for i in range(len(costs))])
            # t = costs[:, 0].max()
            min_cost = min(min_cost, t)
            te, cost = self._fill_trajectory(regret, costs, replace_nan=1)

            idx = cost.tolist().index(t)
            te = te[idx:, :]
            cost = cost[idx:]

            # Clips off all measurements after 10^7s
            idx = np.where(cost <= limit)[0]
            # if hash_to_label_map is not None:
            #     label_name = hash_to_label_map.get(hash_name, method_name+'-'+hash_name)
            # else:
            #     label_name = method_name+'-'+hash_name
            print("{}. Plotting for {}".format(index, mark_label))
            print(len(regret), len(costs))
            print("\nMean: {}; Std: {}\n".format(
                np.mean(te, axis=1)[idx][-1],
                stats.sem(te[idx], axis=1)[-1]))
            # The mean plot
            plt.plot(cost[idx],
                     np.mean(te, axis=1)[idx],
                     color='C{}'.format(index),
                     linewidth=4,
                     label=mark_label,
                     linestyle=self.linestyles[index % len(self.linestyles)],
                     marker=self.marker[index % len(self.marker)],
                     markevery=(0.1, 0.1),
                     markersize=15)
            # The error band
            plt.fill_between(
                cost[idx],
                np.mean(te[idx], axis=1)[idx] + 2 * stats.sem(te[idx], axis=1),
                np.mean(te[idx], axis=1)[idx] - 2 * stats.sem(te[idx], axis=1),
                color="C%d" % index,
                alpha=0.2)

            # Stats to dynamically impose limits on the axes of the plots
            max_cost = max(max_cost, cost[idx][-1])
            self.min_regret = min(self.min_regret,
                                  np.mean(te, axis=1)[idx][-1])
            self.max_regret = max(self.max_regret, np.mean(te, axis=1)[idx][0])

            # For final score table
            mean_df[method_name] = pd.Series(data=np.mean(te, axis=1)[idx],
                                             index=cost[idx])
            std_df[method_name] = pd.Series(data=np.std(te, axis=1)[idx],
                                            index=cost[idx])
        mean_df = pd.DataFrame(mean_df)
        all_mean_df = mean_df.copy()
        all_mean_df.ffill().to_pickle(
            os.path.join(self.exp_dir_root, 'all_mean_df.pkl'))
        std_df = pd.DataFrame(std_df)
        # minimum of the maximum time limit recorded for each algorithm
        cutoff_idx = min(
            list(
                map(lambda x: np.where(~mean_df.isna()[x] == True)[0][-1],
                    mean_df.columns)))
        mean_df = mean_df.iloc[:cutoff_idx + 1].ffill()
        std_df = std_df.iloc[:cutoff_idx + 1].ffill()
        if len(hashset) > 1:
            rank_df = mean_df.apply(stats.rankdata,
                                    axis=1,
                                    result_type='broadcast')
            rank_df.to_pickle(os.path.join(self.exp_dir_root, 'rank_df.pkl'))
        mean_df.iloc[-1].to_pickle(
            os.path.join(self.exp_dir_root, 'mean_df.pkl'))
        std_df.iloc[-1].to_pickle(os.path.join(self.exp_dir_root,
                                               'std_df.pkl'))

        # self.plt = plt
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.min_regret = self.min_regret
        self.max_regret = self.max_regret
        self._regret_plot(**kwargs)

    def _set_plot(self, fix_colors=False):
        rcParams["font.size"] = "25"
        rcParams['text.usetex'] = False
        rcParams['font.family'] = 'serif'
        rcParams['figure.figsize'] = (16.0, 9.0)
        rcParams['figure.frameon'] = True
        rcParams['figure.edgecolor'] = 'k'
        rcParams['grid.color'] = 'k'
        rcParams['grid.linestyle'] = ':'
        rcParams['grid.linewidth'] = 0.5
        rcParams['axes.linewidth'] = 1
        rcParams['axes.edgecolor'] = 'k'
        rcParams['axes.grid.which'] = 'both'
        rcParams['legend.frameon'] = 'True'
        rcParams['legend.framealpha'] = 1

        rcParams['ytick.major.size'] = 12
        rcParams['ytick.major.width'] = 1.5
        rcParams['ytick.minor.size'] = 6
        rcParams['ytick.minor.width'] = 1
        rcParams['xtick.major.size'] = 12
        rcParams['xtick.major.width'] = 1.5
        rcParams['xtick.minor.size'] = 6
        rcParams['xtick.minor.width'] = 1
        self.marker = ['x', '^', 'D', 'o', 's', 'h', '*', 'v', '<', ">"]
        self.linestyles = ['-', '--', '-.', ':']
        # plot setup
        # self.colors = ["C%d" % i for i in range(len(self.methods))]
        # if fix_colors and len(self.methods) <= 8:
        #     _colors = dict()
        #     _colors["RS"] = "C0"
        #     _colors["HB"] = "C7"
        #     _colors["BOHB"] = "C1"
        #     _colors["TPE"] = "C3"
        #     _colors["SMAC"] = "C4"
        #     _colors["RE"] = "C5"
        #     _colors["DE"] = "C6"
        #     _colors["DEHB"] = "C2"
        #     self.colors = []
        #     for (_, l) in self.methods:
        #         self.colors.append(_colors[l])

    def _fill_trajectory(self,
                         performance_list,
                         cost_list,
                         replace_nan=np.NaN):
        frame_dict = collections.OrderedDict()
        counter = np.arange(0, len(performance_list))
        for p, t, c in zip(performance_list, cost_list, counter):
            if len(p) != len(t):
                raise ValueError("(%d) Array length mismatch: %d != %d" %
                                 (c, len(p), len(t)))
            frame_dict[str(c)] = pd.Series(data=p, index=t)

        # creates a dataframe where the rows are indexed based on time
        # fills with NA for missing values for the respective timesteps
        merged = pd.DataFrame(frame_dict)
        # ffill() acts like a fillna() wherein a forward fill happens
        # only remaining NAs for in the beginning until a value is recorded
        merged = merged.ffill()

        performance = merged.to_numpy()  # converts to a 2D numpy array
        cost_ = merged.index.values  # retrieves the timestamps

        performance[np.isnan(performance)] = replace_nan

        if not np.isfinite(performance).all():
            raise ValueError(
                "\nCould not merge lists, because \n"
                "\t(a) one list is empty?\n"
                "\t(b) the lists do not start with the same times and"
                " replace_nan is not set?\n"
                "\t(c) any other reason.")

        return performance, cost_

    def _regret_plot(self, **kwargs):
        plot_name = kwargs.get("plot_name", "comparison")
        output_type = kwargs.get("output_type", "pdf")
        plot_type = kwargs.get("plot_type", "wallclock")
        # bench_type = kwargs.get("bench_type", 'protein')
        title = kwargs.get("title", "benchmark")
        if self.benchmark != 'cc18':
            plt.xscale("log")
        if self.benchmark != 'svm' and self.benchmark != 'bnn':
            plt.yscale("log")
        plt.tick_params(which='both', direction="in")
        if self.benchmark == 'svm' or self.benchmark == 'bnn' or self.benchmark == "cc18" or self.benchmark == "paramnet":
            plt.legend(loc='upper right',
                       framealpha=1,
                       prop={
                           'size': 40,
                           'weight': 'normal'
                       })
        elif self.benchmark == "rl":
            plt.legend(loc='lower left',
                       framealpha=1,
                       prop={
                           'size': 40,
                           'weight': 'normal'
                       },
                       ncol=1)
        else:  #elif self.benchmark == "countingones":
            plt.legend(loc='lower left',
                       framealpha=1,
                       prop={
                           'size': 40,
                           'weight': 'normal'
                       })
        plt.title(title, size=40)

        if self.benchmark == 'rl':
            plt.xlabel("time $[s]$", fontsize=45)
        elif self.benchmark == 'bnn':
            plt.xlabel("MCMC steps", fontsize=45)
        elif self.benchmark == 'countingones':
            plt.xlabel("cummulative budget / $b_{max}$", fontsize=45)
        elif self.benchmark == 'speed':
            plt.xlabel("Runtime sans function evalution")
        elif plot_type == "wallclock":
            plt.xlabel("estimated wallclock time $[s]$", fontsize=45)
        elif plot_type == "fevals":
            plt.xlabel("number of function evaluations", fontsize=45)

        if self.benchmark == 'svm':
            plt.ylabel("{} error".format(self.regret_key), fontsize=45)
        elif self.benchmark == 'rl':
            plt.ylabel("epochs until convergence", fontsize=45)
        elif self.benchmark == 'bnn':
            plt.ylabel("negative log-likelihood", fontsize=45)
        elif self.benchmark == 'countingones':
            plt.ylabel("normalized {} regret".format(self.regret_key),
                       fontsize=40)
        elif self.benchmark == 'countingones':
            plt.ylabel("number of function evaluations", fontsize=45)
        else:
            plt.ylabel("{} regret".format(self.regret_key), fontsize=45)

        if self.benchmark == 'rl':
            # plt.xlim(1e2, 1e5)
            plt.xlim(1e2, self.max_cost)
            # plt.xlim(self.min_cost, self.max_cost)
        elif self.benchmark == 'bnn':
            # plt.xlim(min_limit, self.max_cost)
            plt.xlim(50000, self.max_cost)  # min(self.max_cost*10, limit))
        elif self.benchmark == 'countingones':
            plt.xlim(self.min_cost, self.max_cost)
            # plt.xlim(self.min_cost, 1e4)
        elif self.benchmark == 'cc18':
            plt.xlim(0.1, self.max_cost)
        elif self.benchmark == "paramnet":
            print("Max time: {}".format(self.max_cost))
            plt.xlim(self.min_cost, self.max_cost)
        elif self.benchmark == "101":
            plt.xlim(1e2, self.max_cost)
        else:
            plt.xlim(self.min_cost, self.max_cost)
            # plt.xlim(max(self.min_cost/10, 1e0), min(self.max_cost*10, 1e7))

        if self.benchmark == 'bnn':
            plt.ylim(3, 10)  # 75)
        elif self.benchmark == 'rl':
            plt.ylim(1e2, 1e4)
        elif self.benchmark == 'cc18':
            plt.ylim(0, self.max_regret)
        elif self.benchmark == 'svm':
            plt.ylim(self.min_regret, 0.5)
        else:
            plt.ylim(self.min_regret, self.max_regret)

        plt.grid(which='both', alpha=0.2, linewidth=0.5)
        print(
            os.path.join(self.exp_dir_root,
                         '{}.{}'.format(plot_name, output_type)))
        plt.savefig(os.path.join(self.exp_dir_root,
                                 '{}.{}'.format(plot_name, output_type)),
                    bbox_inches='tight')
