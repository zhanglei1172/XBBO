import collections
from glob import glob
import os, yaml
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd

from matplotlib import rcParams

from xbbo.utils.util import loadJson, loadOBJ
from xbbo.utils.constants import Key


class Analyse():
    def __init__(self,
                 exp_dir_root='./exp',
                 benchmark='countingones',
                 methods=None,
                 marks=None,
                 limit=1e7,
                 **kwargs) -> None:
        self.exp_dir_root = exp_dir_root
        self.out_dir = os.path.join(exp_dir_root, benchmark)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.methods = methods
        self.marks = marks
        self.benchmark = benchmark
        self._set_plot()
        plt.clf()
        hashset = set()
        min_cost = np.inf
        max_cost = 0
        self.min_regret = np.inf
        self.max_regret = -np.inf
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
                (self.methods is not None) and
                (method_name not in self.methods)) or (
                    (self.marks is not None) and
                    (mark_label not in self.marks)) or mark_label in hashset:
                continue
            jfiles = glob('{}/{}/*.json'.format(cfg["GENERAL"]["exp_dir"],
                                                method_name))
            if len(jfiles) == 0:
                continue
            hashset.add(mark_label)
            index += 1

            regret = []
            costs = []
            for jfile in jfiles:
                j = loadJson(jfile)
                curr_regret = np.array(j[self.regret_key])
                if method_name == 'dehb_':
                    curr_cost = np.array(j['runtime'])
                else:
                    curr_cost = np.array(j[Key.COST])
                if self.benchmark == "countingones":
                    d = cfg["TEST_PROBLEM"].get("n_continuous",
                                                4) + cfg["TEST_PROBLEM"].get(
                                                    "n_categorical", 4)
                    if method_name == 'dehb_':
                        curr_regret = curr_regret * d - d
                    curr_regret = (curr_regret + d) / d  # 0-1
                    max_budget = 93312 / d
                    curr_cost /= max_budget
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
            mean_df[mark_label] = pd.Series(data=np.mean(te, axis=1)[idx],
                                            index=cost[idx])
            std_df[mark_label] = pd.Series(data=np.std(te, axis=1)[idx],
                                           index=cost[idx])
        mean_df = pd.DataFrame(mean_df)
        all_mean_df = mean_df.copy()
        all_mean_df.ffill().to_pickle(
            os.path.join(self.out_dir, 'all_mean_df.pkl'))
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
            rank_df.to_pickle(os.path.join(self.out_dir, 'rank_df.pkl'))
        mean_df.iloc[-1].to_pickle(os.path.join(self.out_dir, 'mean_df.pkl'))
        std_df.iloc[-1].to_pickle(os.path.join(self.out_dir, 'std_df.pkl'))

        # self.plt = plt
        self.min_cost = min_cost
        self.max_cost = max_cost
        # self.min_regret = self.min_regret
        # self.max_regret = self.max_regret
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
        legend_size = kwargs.get("legend_size", 40)
        if self.benchmark != 'cc18':
            plt.xscale("log")
        if self.benchmark != 'svm' and self.benchmark != 'bnn':
            plt.yscale("log")
        plt.tick_params(which='both', direction="in")
        if self.benchmark == 'svm' or self.benchmark == 'bnn' or self.benchmark == "cc18" or self.benchmark == "paramnet":
            plt.legend(loc='upper right',
                       framealpha=1,
                       prop={
                           'size': legend_size,
                           'weight': 'normal'
                       })
        elif self.benchmark == "rl":
            plt.legend(loc='lower left',
                       framealpha=1,
                       prop={
                           'size': legend_size,
                           'weight': 'normal'
                       },
                       ncol=1)
        else:  #elif self.benchmark == "countingones":
            plt.legend(loc='lower left',
                       framealpha=1,
                       prop={
                           'size': legend_size,
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
        elif self.benchmark == "nas_101_cifar10":
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
            os.path.join(self.out_dir, '{}.{}'.format(plot_name, output_type)))
        plt.savefig(os.path.join(self.out_dir,
                                 '{}.{}'.format(plot_name, output_type)),
                    bbox_inches='tight')


class Analyse_multi_benchmark():
    def __init__(self, exp_dir_root='./exp', **kwargs) -> None:
        self.exp_dir_root = exp_dir_root
        list_of_mean_files = glob(
            os.path.join(self.exp_dir_root, '*/mean_df.pkl'))

        list_of_std_files = glob(
            os.path.join(self.exp_dir_root, '*/std_df.pkl'))
        self.mean_dfs = {}
        for filename in list_of_mean_files:
            benchname = filename.split('/')[-2]
            self.mean_dfs[benchname] = loadOBJ(filename)
        self.mean_dfs = pd.DataFrame(self.mean_dfs).transpose()
        self.mean_dfs.to_pickle(
            os.path.join(self.exp_dir_root, "all_mean_dfs.pkl"))

        std_dfs = {}
        for filename in list_of_std_files:
            benchname = filename.split('/')[-2]
            std_dfs[benchname] = loadOBJ(filename)
        std_dfs = pd.DataFrame(std_dfs).transpose()
        std_dfs.to_pickle(os.path.join(self.exp_dir_root, "all_std_dfs.pkl"))

        # Load run statistics to create a relative ranking plot over time

        rank_list_candidates = glob(
            os.path.join(self.exp_dir_root, '*/rank_df.pkl'))
        list_of_rank_files = []
        for name in rank_list_candidates:
            # ignore benchmarks where the runtime is not wallclock time in seconds
            # if "countingones" in name or "bnn" in name or "svm" in name:
            #     continue
            list_of_rank_files.append(name)

        # load rankings per benchmark
        rank_dfs = []
        for filename in list_of_rank_files:
            rank_dfs.append(loadOBJ(filename))
        # reorganize data to have algorithms as the top hierarchy, followed by every benchmark for the algo
        avg_rank = {}
        for i in range(len(rank_dfs)):
            for name in rank_dfs[i].columns:
                if name not in avg_rank.keys():
                    avg_rank[name] = {}
                if i not in avg_rank[name].keys():
                    avg_rank[name][i] = None
                avg_rank[name][i] = pd.Series(data=rank_dfs[i][name],
                                              index=rank_dfs[i].index)

        # assigning mean rank to all algorithms at start
        starting_rank = np.mean(np.arange(1, 1 + len(avg_rank.keys())))
        for name, v in avg_rank.items():
            avg_rank[name] = pd.DataFrame(v)
            avg_rank[name].iloc[0] = [starting_rank] * avg_rank[name].shape[1]

        # compute mean relative rank of each algorithm across all benchmarks
        self.rank_lists = {}
        for name, v in avg_rank.items():
            self.rank_lists[name] = pd.Series(data=np.mean(
                avg_rank[name].ffill(), axis=1),
                                              index=avg_rank[name].index)
        self.rank_lists = pd.DataFrame(self.rank_lists)

        self.linestyles = [
            (0, (1, 5)),  # loosely dotted
            (0, (5, 5)),  # loosely dashed
            'dotted',
            (0, (3, 2, 1, 2, 1, 2)),  # dash dot dotted
            'dashed',
            'dashdot',
            (0, (3, 1, 1, 1, 1, 1)),
            'solid'
        ]

        self.colors = ["C%d" % i for i in range(len(self.rank_lists.columns))]
        # if len(rank_lists.columns) <= 8:
        #     _colors = dict()
        #     _colors["RS"] = "C0"
        #     _colors["HB"] = "C7"
        #     _colors["BOHB"] = "C1"
        #     _colors["TPE"] = "C3"
        #     _colors["SMAC"] = "C4"
        #     _colors["RE"] = "C5"
        #     _colors["DE"] = "C6"
        #     _colors["DEHB"] = "C2"
        #     colors = []
        #     for l in rank_lists.columns:
        #         colors.append(_colors[l])

        self.landmarks = np.arange(start=0,
                                   stop=self.rank_lists.shape[0],
                                   step=5)  # for smoothing
        self._set_plot()
        self.rank_plot()

    def _set_plot(self, ):
        rcParams['font.family'] = 'serif'

    def rank_plot(self, ):
        plt.clf()
        xlims = [np.inf, -np.inf]
        for i, name in enumerate(self.rank_lists.columns):
            if name == 'DEHB':
                lw, a = (1.75, 1)
            else:
                lw, a = (1.5, 0.7)
            plt.plot(self.rank_lists[name].index.to_numpy()[self.landmarks],
                     self.rank_lists[name].to_numpy()[self.landmarks],
                     label=name,
                     alpha=a,
                     linestyle=self.linestyles[i],
                     linewidth=1.5,
                     color=self.colors[i])
            xlims[0] = min(xlims[0], self.rank_lists[name].index.to_numpy()[0])
            xlims[1] = max(xlims[1],
                           self.rank_lists[name].index.to_numpy()[-1])

        plt.xscale('log')
        plt.legend(loc='upper left', framealpha=1, prop={'size': 12}, ncol=4)
        # plt.fill_between(
        #     rank_lists['DEHB'].index.to_numpy()[landmarks],
        #     0, rank_lists['DEHB'].to_numpy()[landmarks],
        #     alpha=0.5, color=_colors["DEHB"]
        # )
        # plt.fill_between(
        #     rank_lists['DEHB'].index.to_numpy()[landmarks],
        #     0, starting_rank,
        #     alpha=0.3, color='gray'
        # )
        # plt.hlines(starting_rank, 0, 1e7)
        plt.xlim(xlims[0], xlims[1])
        plt.ylim(1, self.rank_lists.shape[1])
        plt.xlabel('estimated wallclock time $[s]$', fontsize=15)
        plt.ylabel('average relative rank', fontsize=15)
        plt.savefig(os.path.join(self.exp_dir_root, 'rank_plot.pdf'),
                    bbox_inches='tight')

        rank_stats = {}
        rank_stats['minimum'] = np.min(self.rank_lists, axis=0)
        rank_stats['maximum'] = np.max(self.rank_lists, axis=0)
        rank_stats['variance'] = np.var(self.rank_lists, axis=0)
        rank_stats = pd.DataFrame(rank_stats)
        # Ranks based on final numbers
        rank_df = {}
        for idx in self.mean_dfs.index:
            rank_df[idx] = pd.Series(data=stats.rankdata(
                self.mean_dfs.loc[idx]),
                                     index=self.mean_dfs.loc[idx].index)
        rank_df = pd.DataFrame(rank_df)
        print(rank_df.mean(axis=1))