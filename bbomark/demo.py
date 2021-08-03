import uuid
from copy import deepcopy
from itertools import product
import time
import numpy as np
from matplotlib import pyplot as plt

from bbomark.utils.cmd_parse import CmdArgs
from bbomark.utils import cmd_parse as cmd

# import random_search as rs
from constants import ITER, ALPHA, EVAL_Q
import bbomark.utils.quantiles as qt
from bbomark.utils.loading import load_optimizer_kwargs, load_history
from bbomark.optimizers import get_opt_class
from bbomark.bbo import BBO
from bbomark.model.data import (
    METRICS_LOOKUP,
    DATA_LOADERS
)





def experiment_main(args=None, axs=None):  # pragma: main
    """This is in effect the `main` routine for this experiment. However, it is called from the optimizer wrapper file
    so the class can be passed in. The optimizers are assumed to be outside the package, so the optimizer class can't
    be named from inside the main function without using hacky stuff like `eval`.
    """
    if args is None:
        description = "Run a study with one benchmark function and an optimizer"
        args = cmd.parse_args(cmd.experiment_parser(description))
    # args[CmdArgs.opt_rev] = opt_class.get_version()
    # load meta info
    # run_uuid = uuid.UUID(args[CmdArgs.uuid])

    opt_class, feature_space_class = get_opt_class(args[CmdArgs.optimizer])
    if feature_space_class is None:
        feature_space = None
    else:
        feature_space = feature_space_class()
    opt_kwargs = load_optimizer_kwargs(args[CmdArgs.optimizer], args[CmdArgs.optimizer_root])
    features_record = []
    func_eval_record = []
    timeing_record = []
    suggest_log_record = []
    for exp_num in range(args[CmdArgs.n_repeat]):
        bbo = BBO(opt_class,feature_space,
                    opt_kwargs,
                    args[CmdArgs.classifier],
                    args[CmdArgs.data],
                    args[CmdArgs.metric],
                    args[CmdArgs.n_calls],
                    args[CmdArgs.n_suggest],
                    history=args[CmdArgs.history],
                    custom_model_dir=args[CmdArgs.model_dir],
                    # history_dict=args[CmdArgs.history_dict],
                    data_root=args[CmdArgs.data_root],
                    callback=None)

        bbo.run()
        features_record.append(bbo.record.features)
        func_eval_record.append(bbo.record.func_evals)
        timeing_record.append(bbo.record.timing)
        suggest_log_record.append(bbo.record.suggest_log)
        # print(bbo.record)
    features_record = np.asarray(features_record)
    # (repeat_num, eval_num, n_suggest, res_num)
    func_eval_record = np.asarray(func_eval_record)[...,0].mean(axis=-1).transpose()
    estimate, LB, UB= qt.quantile_and_CI(func_eval_record, EVAL_Q, alpha=ALPHA)
    if axs is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        fig2, ax2 = plt.subplots(figsize=(5, 5), dpi=300)
    else:
        ax, ax2 = axs
        fig = None
        fig2 = None
    opt_name = bbo.optimizer_instance.opt_name

    # plt.fill_between(
    #     self.coord_x,
    #     self.stat_res[1],
    #     self.stat_res[2],
    #     # color=opt_name,
    #     alpha=0.5,
    # )
    coord_x =list(range(1, len(estimate)+1))
    line = ax.plot(
        coord_x,
        np.log(estimate),
        # color=opt_name,
        label=opt_name,
        marker=".",
        # alpha=0.6
    )
    ax.fill_between(
        coord_x,
        np.log(LB),
        np.log(UB),
        color=line[0].get_color(),
        alpha=0.5,
    )
    line = ax2.plot(
        coord_x,
        np.log(np.minimum.accumulate(estimate)),
        # color=opt_name,
        label=opt_name,
        marker=".",
        # alpha=0.6
    )

    
    ax.set_xlabel("evaluation", fontsize=10)
    # plt.ylabel("normalized median score", fontsize=10)
    ax.set_ylabel("loss", fontsize=10)
    ax.grid()
    if fig:
        ax.set_title(opt_name)
        ax.legend(fontsize=8,
                  bbox_to_anchor=(1.05, 1),
                  # loc='best',
                  loc="upper right",
                  borderaxespad=0.0)
        fig.show()
        ax2.set_title(opt_name)
        ax2.legend(fontsize=8,
                  bbox_to_anchor=(1.05, 1),
                  # loc='best',
                  loc="upper right",
                  borderaxespad=0.0)
        fig2.show()

    print('-'*100)
    return [ax, ax2]
    # eval_ds = build_eval_ds(function_evals, OBJECTIVE_NAMES)
    # time_ds = build_timing_ds(*timing)
    # suggest_ds = build_suggest_ds(suggest_log)

def main():
    description = "Run a study with one benchmark function and an optimizer"
    args = cmd.parse_args(cmd.experiment_parser(description))
    experiment_main(args=args)

def main_all():
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    description = "Run all benchmark functions"
    args = cmd.parse_args(cmd.launcher_parser(description))
    # G = product(range_str(args[CmdArgs.n_repeat]), c_list, d_list, o_list)
    G = product(args[CmdArgs.classifier], args[CmdArgs.data], args[CmdArgs.optimizer], args[CmdArgs.metric])
    for classifier, data, optimizer, metric in G:
        problemType = DATA_LOADERS[data][1]
        if metric not in METRICS_LOOKUP[problemType]:
            continue
        args_sub = deepcopy(args)
        args_sub[CmdArgs.classifier], args_sub[CmdArgs.data], args_sub[CmdArgs.optimizer], \
        args_sub[CmdArgs.metric] = classifier, data, optimizer, metric
        ax = experiment_main(args=args_sub, ax=ax)
    # ax = experiment_main(args=args)
    ax.legend(fontsize=8,
              # loc="upper left",
              loc="upper right",
              borderaxespad=0.0)
    fig.show()
    fig.savefig('../out/demo_res.png')

def benchmark_opt():
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    fig2, ax2 = plt.subplots(figsize=(5, 5), dpi=300)
    axs = [ax, ax2]
    description = "Run a benchmark function with all optimizers"
    args = cmd.parse_args(cmd.benchmark_opt_parser(description))
    # G = product(range_str(args[CmdArgs.n_repeat]), c_list, d_list, o_list)
    # G = product(args[CmdArgs.classifier], args[CmdArgs.data], args[CmdArgs.optimizer], args[CmdArgs.metric])
    for optimizer in args[CmdArgs.optimizer]:
        args_sub = deepcopy(args)
        args_sub[CmdArgs.optimizer] = optimizer
        axs = experiment_main(args=args_sub, axs=axs)
    # ax = experiment_main(args=args)
    axs[0].legend(fontsize=8, loc="upper right", borderaxespad=0.0)
    axs[1].legend(fontsize=8, loc="upper right", borderaxespad=0.0)
    fig.show()
    fig.savefig('../out/demo_res{}.png'.format(time.time()))
    fig2.show()
    fig2.savefig('../out/demo_res{}.png'.format(time.time()))

if __name__ == '__main__':
    # main()
    benchmark_opt()
