import numpy as np
import gc

from xbbo.pipeline.bbo_benchmark import BBObenchmark
from xbbo.utils.analysis import Analyse, Analyse_multi_benchmark
from xbbo.utils.config import cfg, load_cfg_fom_args
from xbbo.utils.constants import MAXINT


def do_experiment(cfg_clone):  # pragma: main
    # seed = cfg.GENARAL.random_seed
    SEED = cfg_clone.GENERAL.random_seed
    rng = np.random.RandomState(SEED)
    if cfg_clone.GENERAL.pipeline == 'BBO':
        bbo = BBObenchmark(cfg_clone, SEED)

        for r in range(cfg_clone.repeat_num):
            print('==EXP {}==:'.format(r))
            # np.random.seed(SEED)
            # random.seed(SEED)

            bbo.run_one_exp()
            bbo.save_to_file(r)
            print('='*20)
            seed = rng.randint(MAXINT)
            bbo.reset(seed)
            gc.collect()


    else:
        raise NotImplementedError


if __name__ == '__main__':
    # cfg_clone = cfg.clone()
    # cfg.freeze()
    # load_cfg_fom_args(cfg_clone, argv=[                "-c",
    #             "./cfgs/dehb.yaml",
    #             "-r",
    #             "10",
    #             "--mark",
    #             "DEHB"])  # repeat 3 times with diffent seeds
    # do_experiment(cfg_clone)
    # cfg.defrost()

    # cfg_clone = cfg.clone()
    # cfg.freeze()
    # load_cfg_fom_args(cfg_clone, argv=[                "-c",
    #             "./cfgs/rs.yaml",
    #             "-r",
    #             "10",
    #             "--mark",
    #             "RS"])  # repeat 3 times with diffent seeds
    # do_experiment(cfg_clone)
    # cfg.defrost()

    cfg_clone = cfg.clone()
    cfg.freeze()
    load_cfg_fom_args(cfg_clone, argv=[                "-c",
                "./cfgs/smac3.yaml",
                "-r",
                "10",
                "--mark",
                "smac3"])  # repeat 3 times with diffent seeds
    do_experiment(cfg_clone)
    cfg.defrost()
    Analyse(cfg_clone.GENERAL.exp_dir_root, benchmark='nas_201', methods=['dehb', 'rs', "smac3"])
    Analyse_multi_benchmark()
