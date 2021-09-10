import glob

import numpy as np
import random

import torch

from bbomark.bbo import BBO
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
            torch.manual_seed(SEED)
            # torch.seed()
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


    cfg_clone = cfg.clone()
    load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/gp.yaml', '-r', '15'])

    main(cfg_clone)

    cfg_clone = cfg.clone()
    load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/bo_tst.yaml', '-r', '15'])

    main(cfg_clone)

    cfg_clone = cfg.clone()
    load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/bo_taf.yaml', '-r', '15'])
    main(cfg_clone)


    cfg_clone = cfg.clone()
    load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/bo_rgpe_mean.yaml', '-r', '15'])
    main(cfg_clone)

    cfg_clone = cfg.clone()
    load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/bo_taf_rgpe.yaml', '-r', '15'])
    main(cfg_clone)

    cfg_clone = cfg.clone()
    load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/bo_RMoGP.yaml', '-r', '15'])
    main(cfg_clone)
    # benchmark_opt()

    # cfg_clone = cfg.clone()
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/transfer_taf_svm.yaml', '-r', '1'])
    # main(cfg_clone)
