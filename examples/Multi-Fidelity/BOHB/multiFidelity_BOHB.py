import numpy as np
from ConfigSpace import ConfigurationSpace
# import matplotlib.pyplot as plt
from xbbo.problem.fast_example_problem import CountingOnes
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from xbbo.search_algorithm.multi_fidelity.hyperband import HB
from xbbo.search_algorithm.multi_fidelity.DEHB import DEHB
from xbbo.core.constants import MAXINT, Key


if __name__ == "__main__":
    rng = np.random.RandomState(42)

    # define black box function
    mf_blackbox_func = CountingOnes(n_categorical=4,n_continuous=4, rng=rng)
    # define search space
    cs = mf_blackbox_func.get_configuration_space()
    # define black box optimizer
    mf_hpopt = DEHB(space=cs,
                  budget_bound=[9, 729],
                  eta=3,
                  seed=rng.randint(MAXINT),
                  round_limit=1)
    # ---- Begin BO-loop ----
    cnt = 0
    while not mf_hpopt.check_stop():
        # suggest
        trial_list = mf_hpopt.suggest()
        # evaluate
        obs = mf_blackbox_func(trial_list[0].config_dict, **trial_list[0].info)
        # observe
        trial_list[0].add_observe_value(obs)
        mf_hpopt.observe(trial_list=trial_list)
        
        print(obs)
        cnt += 1
    print(cnt)

    print('find best (value, config):{}'.format(mf_hpopt.trials.get_best()))
