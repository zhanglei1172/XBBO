from copy import deepcopy
from itertools import product

from matplotlib import pyplot as plt

from bbomark.utils.cmd_parse import CmdArgs
from bbomark.utils import cmd_parse as cmd

# import random_search as rs
from bbomark.utils.loading import load_optimizer_kwargs, load_history, save_history
from bbomark.optimizers import get_opt_class
from bbomark.bbo import BBO
from bbomark.model.data import (
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
                history=args[CmdArgs.history],
                # history_dict=args[CmdArgs.history_dict],
                data_root=args[CmdArgs.data_root],
                callback=None)

    next_points, features = bbo.optimizer_instance.suggest(bbo.n_suggestions)  # TODO 1
    print('-'*100)
    return (next_points)
    # eval_ds = build_eval_ds(function_evals, OBJECTIVE_NAMES)
    # time_ds = build_timing_ds(*timing)
    # suggest_ds = build_suggest_ds(suggest_log)

def parse_config(param):
    new_param = {}
    for name in param:
        if param[name] is None:
            continue
        if name == 'e':
            for i in range(len(param[name])):  # 21层
                new_param[name + '_{}'.format(i)] = {
                    'type': 'real',
                    'range': (4, 6)
                }
            continue
        if name == 'ks':
            val = [3, 5, 7]
        elif name == 'd':
            val = [2, 3, 4]
        elif '_bits_' in name:
            val = [4, 6, 8]
        else:
            assert False
        for i in range(len(param[name])): # 21层
            new_param[name + '_%d' % i] = {
                'type': 'cat',
                'values': (val)
            }
    # print(new_param)

def prepare(params, acc):
    new_params = []
    for param in params:
        new_param = {}
        for name in param:
            if param[name] is None:
                continue
            for i, layer_p in enumerate(param[name]):
                new_param[name + '_{}'.format(i)] = layer_p
        new_params.append(new_param)
    # print(new_params)
    save_history('../out/history.pkl', {
        'params': new_params,
        'y': list(map(lambda x: -x, acc))
    })

def main():
    description = "Run a study with one benchmark function and an optimizer"
    args = cmd.parse_args(cmd.experiment_parser(description))
    params = [{'wid': None, 'ks': [5, 3, 7, 3, 7, 3, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 3, 3, 3], 'e': [5.0, 5.0, 6.0, 5.0, 5.666666666666667, 5.8, 5.0, 5.2, 5.6, 5.1, 5.3, 5.8, 5.6, 5.833333333333333, 5.0, 4.916666666666667, 4.666666666666667, 5.875, 5.875, 4.916666666666667, 6.0], 'd': [2, 4, 4, 3, 4, 3], 'pw_w_bits_setting': [6, 6, 6, 4, 6, 4, 4, 4, 4, 8, 6, 6, 6, 6, 6, 8, 4, 8, 4, 4, 4], 'pw_a_bits_setting': [4, 8, 6, 8, 6, 4, 6, 8, 6, 8, 8, 8, 8, 8, 8, 4, 6, 8, 8, 4, 6], 'dw_w_bits_setting': [4, 6, 6, 4, 8, 4, 4, 4, 6, 4, 8, 4, 8, 8, 8, 4, 4, 8, 4, 6, 6], 'dw_a_bits_setting': [6, 4, 6, 6, 6, 4, 8, 8, 6, 8, 6, 4, 6, 6, 6, 6, 6, 4, 6, 8, 4]}] * 1
    prepare(params, acc=[0.9])
    parse_config(params[0])
    args[CmdArgs.history] = '../out/history.pkl'
    res = experiment_main(args=args)
    print('suggest:',  res)

if __name__ == '__main__':
    main()

