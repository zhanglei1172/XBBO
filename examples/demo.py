import glob

import numpy as np
import random

from xbbo.pipeline.bbo import BBO
from xbbo.pipeline.pbt import PBT
from xbbo.pipeline.transfer_bbo import Transfer_BBO
from xbbo.utils.config import cfg, load_cfg_fom_args
from xbbo.core.constants import MAXINT

# cfg.freeze()


def experiment_main(cfg_clone):  # pragma: main
    # seed = cfg.GENARAL.random_seed
    SEED = cfg_clone.GENERAL.random_seed
    rng = np.random.RandomState(SEED)
    if cfg_clone.GENERAL.pipeline == 'BBO':

        # opt_kwargs = load_optimizer_kwargs(args[CmdArgs.optimizer], args[CmdArgs.optimizer_root])
        for r in range(cfg_clone.repeat_num):
            seed = rng.randint(MAXINT)
            # np.random.seed(SEED)
            # random.seed(SEED)
            bbo = BBO(cfg_clone, seed)

            bbo.run()
            bbo.record.save_to_file(r)
            print(bbo.record)
    elif cfg_clone.GENERAL.pipeline == 'NAS':
        for r in range(cfg_clone.repeat_num):

            # SEED = cfg_clone.GENERAL.random_seed + r
            # np.random.seed(SEED)
            # random.seed(SEED)
            nas = NAS(cfg_clone)

            nas.run()
            nas.record.save_to_file(r)
            print(nas.record)
    elif cfg_clone.GENERAL.pipeline == 'transfer_bbo':

        for r in range(cfg_clone.repeat_num):

            # SEED = cfg_clone.GENERAL.random_seed + r
            # np.random.seed(SEED)
            # random.seed(SEED)
            bbo = Transfer_BBO(cfg_clone)

            bbo.run()
            bbo.record.save_to_file(r)
            print(bbo.record)
    elif cfg_clone.GENERAL.pipeline == 'PBT':

        # opt_kwargs = load_optimizer_kwargs(args[CmdArgs.optimizer], args[CmdArgs.optimizer_root])
        for r in range(cfg_clone.repeat_num):

            # SEED = cfg_clone.GENERAL.random_seed + r
            # np.random.seed(SEED)
            # random.seed(SEED)
            pbt = PBT(cfg_clone)

            scores = pbt.run()
            pbt.show_res(scores)
            # pbt.show_toy_res(scores)
            print(pbt)
    else:
        raise NotImplementedError


def main(cfg_clone):
    # load_cfg_fom_args()

    experiment_main(cfg_clone)


if __name__ == '__main__':
    # toy_bbo_cfg_files = [
    #     # "toy_turbo-1.yaml"
    #     # "toy_turbo-5.yaml"
    #     # "toy_gp.yaml",
    #     # "toy_anneal.yaml",
    #     # "toy_bore.yaml",
    #     # "toy_cem.yaml",
    #     # "toy_cma.yaml",
    #     # "toy_de.yaml",
    #     # "toy_rea.yaml",
    #     # "toy_rs.yaml",
    #     # "toy_tpe.yaml"

    #     "bo_gp2.yaml"
    # ]

    # for file in toy_bbo_cfg_files:
    #     cfg_clone = cfg.clone()
    #     cfg.freeze()
    #     load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/'+file, '-r', '3']) # repeat 3 times with diffent seeds
    #     main(cfg_clone)
    #     cfg.defrost()
    cfg_clone = cfg.clone()
    cfg.freeze()
    load_cfg_fom_args(cfg_clone)  # repeat 3 times with diffent seeds
    main(cfg_clone)
    cfg.defrost()
    # cfg_clone = cfg.clone()
    # cfg.freeze()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_scikit.yaml', '-r', '3'])
    # # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_rng.yaml', '-r', '1'])
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_hyperopt.yaml', '-r', '1'])
    # main(cfg_clone)
    # cfg.defrost()

    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_rs.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_bore.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_hyperopt.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_nevergrad.yaml', '-r', '1'])
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_de.yaml', '-r', '1'])
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_cma.yaml', '-r', '1'])
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_nsga.yaml', '-r', '1'])
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/pbt_mnist.yaml', '-r', '1'])
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/pbt_toy.yaml', '-r', '1'])

    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_rea.yaml', '-r', '1'])
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_tpe.yaml', '-r', '1'])
    # load_cfg_fom_args(cfg_clone, argv=['-c', './cfgs/toy_cem.yaml', '-r', '1'])

    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_opentuner.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_pysot.yaml', '-r', 1])
    # cfg_clone = cfg.clone()
    # load_cfg_fom_args(cfg_clone, argv=['-c', '../cfgs/toy_scikit.yaml', '-r', 1])
    # main(cfg_clone)
    # benchmark_opt()
