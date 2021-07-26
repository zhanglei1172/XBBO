from copy import deepcopy
from itertools import product

from matplotlib import pyplot as plt

from bbomark.utils.cmd_parse import CmdArgs
from bbomark.utils import cmd_parse as cmd

# import random_search as rs
from bbomark.utils.config_load import load_optimizer_kwargs
from bbomark.optimizers import get_opt_class
from bbomark.bbo import BBO
from bbomark.data.data import (
    METRICS_LOOKUP,
    DATA_LOADERS
)





def experiment_main(args=None, ax=None):  # pragma: main
    """This is in effect the `main` routine for this experiment. However, it is called from the optimizer wrapper file
    so the class can be passed in. The optimizers are assumed to be outside the package, so the optimizer class can't
    be named from inside the main function without using hacky stuff like `eval`.
    """
    if args is None:
        description = "Run a study with one benchmark function and an optimizer"
        args = cmd.parse_args(cmd.experiment_parser(description))
    # args[CmdArgs.opt_rev] = opt_class.get_version()
    # load meta info
    opt_class, feature_space_class = get_opt_class(args[CmdArgs.optimizer])
    if feature_space_class is None:
        feature_space = None
    else:
        feature_space = feature_space_class()
    opt_kwargs = load_optimizer_kwargs(args[CmdArgs.optimizer], args[CmdArgs.optimizer_root])
    bbo = BBO(opt_class,feature_space,
                opt_kwargs,
                args[CmdArgs.classifier],
                args[CmdArgs.data],
                args[CmdArgs.metric],
                args[CmdArgs.n_calls],
                args[CmdArgs.n_suggest],
                data_root=args[CmdArgs.data_root],
                callback=None)

    bbo.run()
    print(bbo.record)
    ax = bbo.visualize(ax)
    print('-'*100)
    return ax
    # eval_ds = build_eval_ds(function_evals, OBJECTIVE_NAMES)
    # time_ds = build_timing_ds(*timing)
    # suggest_ds = build_suggest_ds(suggest_log)

def main():
    description = "Run a study with one benchmark function and an optimizer"
    args = cmd.parse_args(cmd.experiment_parser(description))
    experiment_main(args=args)

def main_all():
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    description = "Run a study with one benchmark function and an optimizer"
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
    ax.legend(fontsize=8, loc="upper left", borderaxespad=0.0)
    fig.show()
    fig.savefig('../out/demo_res.png')

if __name__ == '__main__':
    # main()
    main_all()
