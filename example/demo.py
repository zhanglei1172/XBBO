import glob

import numpy as np
import random

from bbomark.bbo import BBO
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
    else:
        raise NotImplementedError





def main(cfg_clone):
    # load_cfg_fom_args()

    experiment_main(cfg_clone)



if __name__ == '__main__':

    # for file in glob.iglob('../cfgs/*.yaml'):
    #     cfg_clone = cfg.clone()
    #     cfg.freeze()
    #     load_cfg_fom_args(cfg_clone, argv=['-c', file, '-r', '3'])
    #     main(cfg_clone)
    #     cfg.defrost()

    cfg_clone = cfg.clone()
    cfg.freeze()
    # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_scikit.yaml', '-r', '3'])
    load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_sng.yaml', '-r', '1'])
    main(cfg_clone)
    cfg.defrost()

    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_rs.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_bore.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_hyperopt.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_nevergrad.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_opentuner.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_pysot.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_scikit.yaml', '-r', 1])
    # main(cfg_clone)
    # benchmark_opt()
