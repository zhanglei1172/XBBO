import numpy as np

from xbbo.pipeline.bbo_benchmark import BBObenchmark
from xbbo.utils.analysis import Analyse, Analyse_multi_benchmark
from xbbo.utils.config import cfg, load_cfg_fom_args
from xbbo.utils.constants import MAXINT


def do_experiment(cfg_clone):  # pragma: main
    # seed = cfg.GENARAL.random_seed
    SEED = cfg_clone.GENERAL.random_seed
    rng = np.random.RandomState(SEED)
    if cfg_clone.GENERAL.pipeline == 'BBO':

        for r in range(cfg_clone.repeat_num):
            print('==EXP {}==:'.format(r))
            seed = rng.randint(MAXINT)
            # np.random.seed(SEED)
            # random.seed(SEED)
            bbo = BBObenchmark(cfg_clone, seed)

            bbo.run_one_exp()
            bbo.save_to_file(r)
            print('='*20)


    else:
        raise NotImplementedError


if __name__ == '__main__':
    cfg_clone = cfg.clone()
    cfg.freeze()
    load_cfg_fom_args(cfg_clone, argv=[                "-c",
                "./cfgs/dehb.yaml",
                "-r",
                "10",
                "--mark",
                "DEHB"])  # repeat 3 times with diffent seeds
    do_experiment(cfg_clone)
    cfg.defrost()

    cfg_clone = cfg.clone()
    cfg.freeze()
    load_cfg_fom_args(cfg_clone, argv=[                "-c",
                "./cfgs/rs.yaml",
                "-r",
                "10",
                "--mark",
                "RS"])  # repeat 3 times with diffent seeds
    do_experiment(cfg_clone)
    cfg.defrost()
    Analyse(cfg_clone.GENERAL.exp_dir_root, benchmark='countingones', methods=['dehb', 'rs'])
    Analyse_multi_benchmark()
