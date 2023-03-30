import numpy as np
# import matplotlib.pyplot as plt
from ConfigSpace import ConfigurationSpace
from xbbo.configspace.space import DenseConfiguration
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.conditions import LessThanCondition

# from xbbo.search_algorithm.transfer_tst_optimizer import TransferBO
# from xbbo.search_algorithm.transfer_taf_optimizer import TransferBO
# from xbbo.search_algorithm.transfer_rgpe_mean_optimizer import TransferBO
# from xbbo.search_algorithm.transfer_taf_rgpe_optimizer import TransferBO
# from xbbo.search_algorithm.transfer_RMoGP_optimizer import TransferBO
from xbbo.search_algorithm.transfer_bo_optimizer import TransferBO

from xbbo.problem.transfer_problem import BenchName, TransferBenchmark
from xbbo.core.constants import MAXINT

from utils import get_logger, init_logger

init_logger('./logs/TAF.log')
_logger = get_logger()

if __name__ == "__main__":
    MAX_CALL = 30
    rng = np.random.RandomState(42)

    transfer_bench = TransferBenchmark(bench_name=BenchName.TST,
                                       target_task_name="A9A",
                                       data_path_root='./data',
                                       data_base_name='svm',
                                       rng=rng.randint(MAXINT))
    # transfer_bench = TransferBenchmark(bench_name=BenchName.Table_deepar,
    #                                    target_task_name="wiki-rolling",
    #                                    data_path_root='./data/offline_evaluations',
    #                                    data_base_name='DeepAR.csv.zip',
    #                                    rng=rng.randint(MAXINT))
    cs = transfer_bench.get_configuration_space()
    old_D_X, old_D_y = transfer_bench.get_old_data()

    # use transfer
    # hpopt = TransferBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='tst', acq_func='ei', weight_srategy='kernel', acq_opt='rs') # TST-R
    hpopt = TransferBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_func='taf', weight_srategy='kernel', acq_opt='rs') # TAF
    # hpopt = TransferBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='tst', acq_func='ei', weight_srategy='rw', acq_opt='rs') # RGPE(mean)
    # hpopt = TransferBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_func='taf', weight_srategy='rw', acq_opt='rs') # TAF(rw)
    # hpopt = TransferBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_func='mogp', weight_srategy='rw', acq_opt='rs') # RMoGP
    # not use transfer
    # hpopt = TransferBO(space=cs,
    #                    seed=rng.randint(MAXINT),
    #                    suggest_limit=MAX_CALL,
    #                    initial_design='sobol',
    #                    surrogate='gp',
    #                    acq_func='ei',
    #                    weight_srategy='kernel',
    #                    acq_opt='rs')  # vanila bo
    hpopt.get_transfer_knowledge(old_D_X, old_D_y)
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate
        obs = transfer_bench(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(obs)
        hpopt.observe(trial_list=trial_list)

        # print(obs)
        _logger.info('iter:{}, obs:{}'.format(i, obs))

    # print('find best (value, config):{}'.format(hpopt.trials.get_best()))
    _logger.info('find best (value, config):{}'.format(hpopt.trials.get_best()))
