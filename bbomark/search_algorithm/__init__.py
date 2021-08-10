
def get_opt_class(opt_name):

    if opt_name == "hyperopt_optimizer":
        from bbomark.search_algorithm import hyperopt_optimizer as opt
    elif opt_name == "nevergrad_optimizer":
        from bbomark.search_algorithm import nevergrad_optimizer as opt
    elif opt_name == "opentuner_optimizer":
        from bbomark.search_algorithm import opentuner_optimizer as opt
    elif opt_name == "pysot_optimizer":
        from bbomark.search_algorithm import pysot_optimizer as opt
    elif opt_name == "random_optimizer":
        from bbomark.search_algorithm import random_optimizer as opt
    elif opt_name == "scikit_optimizer":
        from bbomark.search_algorithm import scikit_optimizer as opt
    elif opt_name == "bore_optimizer":
        from bbomark.search_algorithm import bore_optimizer as opt
    elif opt_name == "sng_optimizer":
        from bbomark.search_algorithm import sng_optimizer as opt
    else:
        assert False, f"{opt_name} is not in bbomark/search_algorithm or not change bbomark/search_algorithm/__init__.py"

    return opt.opt_class