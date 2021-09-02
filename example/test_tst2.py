import glob
import os
import time

import numpy as np
import random

from bbomark.bbo import BBO
from bbomark.search_algorithm import get_opt_class
from bbomark.transfer_bbo import Transfer_BBO
from bbomark.utils.config import cfg, load_cfg_fom_args
from bbomark.nas import NAS

# cfg.freeze()


def experiment_main(cfg_clone):  # pragma: main
    # seed = cfg.GENARAL.random_seed

    if cfg_clone.GENERAL.pipeline == 'BBO':

        # opt_kwargs = load_optimizer_kwargs(args[CmdArgs.optimizer], args[CmdArgs.optimizer_root])
        for r in range(cfg_clone.repeat_num):
            SEED = cfg_clone.GENERAL.random_seed + r
            np.random.seed(SEED)
            random.seed(SEED)
            bbo = BBO(cfg_clone)

            bbo.run()
            bbo.record.save_to_file(r)
            print(bbo.record)
    elif cfg_clone.GENERAL.pipeline == 'NAS':
        for r in range(cfg_clone.repeat_num):
            SEED = cfg_clone.GENERAL.random_seed + r
            np.random.seed(SEED)
            random.seed(SEED)
            nas = NAS(cfg_clone)

            nas.run()
            nas.record.save_to_file(r)
            print(nas.record)
    elif cfg_clone.GENERAL.pipeline == 'transfer':

        for r in range(cfg_clone.repeat_num):
            SEED = cfg_clone.GENERAL.random_seed + r
            np.random.seed(SEED)
            random.seed(SEED)
            bbo = Transfer_BBO(cfg_clone)

            bbo.run()
            bbo.record.save_to_file(r)
            print(bbo.record)

    else:
        raise NotImplementedError


def main(cfg_clone):
    # load_cfg_fom_args()

    experiment_main(cfg_clone)


if __name__ == '__main__':

    for filename in os.listdir("/home/zhang/PycharmProjects/MAC/TST/data/svm/"):
        cfg_clone = cfg.clone()
        load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/transfer_baseline_svm.yaml', '-r', '1',
                                           "TEST_PROBLEM.kwargs.test_data", filename])
        SEED = cfg_clone.GENERAL.random_seed
        main(cfg_clone)

    time.sleep(1)
    for filename in os.listdir("/home/zhang/PycharmProjects/MAC/TST/data/svm/"):
        cfg_clone = cfg.clone()
        load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/transfer_taf_svm.yaml', '-r', '1',
                                           "TEST_PROBLEM.kwargs.test_data", filename])
        SEED = cfg_clone.GENERAL.random_seed
        main(cfg_clone)

    time.sleep(1)
    for filename in os.listdir("/home/zhang/PycharmProjects/MAC/TST/data/svm/"):
        cfg_clone = cfg.clone()
        load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/transfer_svm.yaml', '-r', '1',
                                           "TEST_PROBLEM.kwargs.test_data", filename])
        SEED = cfg_clone.GENERAL.random_seed
        main(cfg_clone)


    # opt_class = get_opt_class(cfg.OPTM.name)
    # optimizer_instance = opt_class(None)

    # main(cfg_clone)

    # benchmark_opt()
