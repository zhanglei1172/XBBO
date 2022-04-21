import numpy as np
# import matplotlib.pyplot as plt
from xbbo.search_space.fast_example_problem import mf_stochastic_count_one, build_mf_SCO_space

from xbbo.search_algorithm.multi_fidelity.hyperband import HB
from xbbo.utils.constants import MAXINT

if __name__ == "__main__":
    rng = np.random.RandomState(42)

    # define black box function
    mf_blackbox_func = mf_stochastic_count_one
    # define search space
    cs = build_mf_SCO_space(rng, dim=8)
    # define black box optimizer
    mf_hpopt = HB(space=cs,
                  budget_bound=[9, 729],
                  eta=3,
                  seed=rng.randint(MAXINT),
                  round_limit=10)
    # Example call of the black-box function
    def_res = mf_blackbox_func(cs.get_default_configuration())
    print("Default res: {}".format(def_res))
    # ---- Begin BO-loop ----
    cnt = 0
    while not mf_hpopt.check_stop():
        # suggest
        trial_list = mf_hpopt.suggest()
        # evaluate
        res = mf_blackbox_func(trial_list[0].config_dict, trial_list[0].info)
        # observe
        trial_list[0].add_observe_value(observe_value=res['fitness'],
                                        obs_info=res)
        mf_hpopt.observe(trial_list=trial_list)
        
        print(res['fitness'])
        cnt += 1
    print(cnt)

    # plt.plot(hpopt.trials.get_history()[0])
    # plt.savefig('./out/rosenbrock_bo_gp.png')
    # plt.show()
    print('find best value:{}'.format(mf_hpopt.trials.get_best()[0]))
