from bbomark.optimizers.config import CONFIG


def get_opt_class(opt_name):
    """Load the relevant wrapper class based on this optimizer name.

    There is inherently a bit ugly, but is only called at the main() level before the inner workings get going. There
    are a few ways to do this with some pro and con:
    1) The way done here: based on the filename, load that module via conditional imports and if-else. cons:
        - uses conditional imports
        - must manually repeat yourself in the if-else, but these are checked in unit testing
    2) Import everything and then pick the right optimizer based on a dict of name_str -> class. cons:
        - loads every dependency no matter which is used so could be slow
        - also a stupid dependency might change global state in a way that corrupts experiments
    3) Use the wrapper file as the entry point and add that to setup.py. cons:
        - Will clutter the CLI namespace with one command for each wrapper
    4) Use importlib to import the specified file. cons:
        - Makes assumptions about relative path structure. For pip-installed packages, probably safer to let python
        find the file via import.
    This option (1) seems least objectionable. However, this function could easily be switched to use importlib without
    any changes elsewhere.
    """
    wrapper_file, _ = CONFIG[opt_name]

    if wrapper_file == "hyperopt_optimizer.py":
        from bbomark.optimizers import hyperopt_optimizer as opt
    elif wrapper_file == "nevergrad_optimizer.py":
        from bbomark.optimizers import nevergrad_optimizer as opt
    elif wrapper_file == "opentuner_optimizer.py":
        from bbomark.optimizers import opentuner_optimizer as opt
    elif wrapper_file == "pysot_optimizer.py":
        from bbomark.optimizers import pysot_optimizer as opt
    elif wrapper_file == "random_optimizer.py":
        from bbomark.optimizers import random_optimizer as opt
    elif wrapper_file == "scikit_optimizer.py":
        from bbomark.optimizers import scikit_optimizer as opt
    elif wrapper_file == "bore_optimizer.py":
        from bbomark.optimizers import bore_optimizer as opt
    else:
        assert False, "CONFIG for built in optimizers has added a new optimizer, but not updated this function."

    opt_class = opt.opt_wrapper
    return opt_class, opt.feature_space