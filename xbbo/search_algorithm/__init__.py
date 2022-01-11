class Register():
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            print("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

alg_register = Register("all avaliable search algorithms")

def get_opt_class(opt_name):

    if opt_name == "hyperopt_optimizer":
        from xbbo.search_algorithm import hyperopt_optimizer as opt
    elif opt_name == "nevergrad_optimizer":
        from xbbo.search_algorithm import nevergrad_optimizer as opt
    elif opt_name == "opentuner_optimizer":
        from xbbo.search_algorithm import opentuner_optimizer as opt
    elif opt_name == "pysot_optimizer":
        from xbbo.search_algorithm import pysot_optimizer as opt
    elif opt_name == "random_optimizer":
        from xbbo.search_algorithm import random_optimizer as opt
    elif opt_name == "scikit_optimizer":
        from xbbo.search_algorithm import scikit_optimizer as opt
    elif opt_name == "bore_optimizer":
        from xbbo.search_algorithm import bore_optimizer as opt
    elif opt_name == "sng_optimizer":
        from xbbo.search_algorithm import sng_optimizer as opt
    elif opt_name == "toy_tpe_optimizer":
        from xbbo.search_algorithm import toy_tpe_optimizer as opt
    elif opt_name == "cem_optimizer":
        from xbbo.search_algorithm import cem_optimizer as opt
    elif opt_name == "anneal_optimizer":
        from xbbo.search_algorithm import anneal_optimizer as opt
    elif opt_name == "anneal_tree_optimizer":
        from xbbo.search_algorithm import anneal_tree_optimizer as opt
    elif opt_name == "transfer_tst_optimizer":
        from xbbo.search_algorithm import transfer_tst_optimizer as opt
    elif opt_name == "transfer_taf_optimizer":
        from xbbo.search_algorithm import transfer_taf_optimizer as opt
    elif opt_name == "transfer_baseline_optimizer":
        from xbbo.search_algorithm import transfer_baseline_optimizer as opt
    elif opt_name == "transfer_rgpe_optimizer":
        from xbbo.search_algorithm import transfer_rgpe_optimizer as opt
    elif opt_name == "transfer_rs_optimizer":
        from xbbo.search_algorithm import transfer_rs_optimizer as opt
    elif opt_name == "transfer_baseline_optimizer_":
        from xbbo.search_algorithm import transfer_baseline_optimizer_ as opt
    elif opt_name == "transfer_tst_optimizer_":
        from xbbo.search_algorithm import transfer_tst_optimizer_ as opt
    elif opt_name == "transfer_taf_optimizer_":
        from xbbo.search_algorithm import transfer_taf_optimizer_ as opt
    elif opt_name == "transfer_rgpe_mean_optimizer_":
        from xbbo.search_algorithm import transfer_rgpe_mean_optimizer_ as opt
    elif opt_name == "transfer_taf_rgpe_optimizer_":
        from xbbo.search_algorithm import transfer_taf_rgpe_optimizer_ as opt
    elif opt_name == "transfer_RMoGP_optimizer_":
        from xbbo.search_algorithm import transfer_RMoGP_optimizer_ as opt
    elif opt_name == "de_optimizer":
        from xbbo.search_algorithm import de_optimizer as opt
    elif opt_name == "cma_optimizer":
        from xbbo.search_algorithm import cma_optimizer as opt
    elif opt_name == "nsga_optimizer":
        from xbbo.search_algorithm import nsga_optimizer as opt
    elif opt_name == "regularizedEA_optimizer":
        from xbbo.search_algorithm import regularizedEA_optimizer as opt
    elif opt_name == "pbt_optimizer":
        from xbbo.search_algorithm import pbt_optimizer as opt
    elif opt_name == "turbo_optimizer":
        from xbbo.search_algorithm import turbo_optimizer as opt
    elif opt_name == "bo_optimizer":
        from xbbo.search_algorithm import bo_optimizer as opt
    else:
        assert False, f"{opt_name} is not in xbbo/search_algorithm or not change xbbo/search_algorithm/__init__.py"

    return opt.opt_class