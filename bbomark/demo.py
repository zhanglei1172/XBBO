from bbomark.utils.cmd_parse import CmdArgs
from bbomark.utils import cmd_parse as cmd

# import random_search as rs
from bbomark.utils.config_load import load_optimizer_kwargs
from bbomark.optimizers import get_opt_class
from bbomark.bbo import BBO





def experiment_main(args=None):  # pragma: main
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
    # eval_ds = build_eval_ds(function_evals, OBJECTIVE_NAMES)
    # time_ds = build_timing_ds(*timing)
    # suggest_ds = build_suggest_ds(suggest_log)

def main():
    description = "Run a study with one benchmark function and an optimizer"
    args = cmd.parse_args(cmd.experiment_parser(description))
    experiment_main(args=args)

if __name__ == '__main__':
    main()
