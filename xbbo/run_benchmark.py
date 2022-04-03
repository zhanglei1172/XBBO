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
        # bbo = BBObenchmark(cfg_clone, SEED)

        # for r in range(cfg_clone.repeat_num):
        #     print('==EXP {}==:'.format(r))
        #     # np.random.seed(SEED)
        #     # random.seed(SEED)

        #     bbo.run_one_exp()
        #     bbo.save_to_file(r)
        #     print('=' * 20)
        #     seed = rng.randint(MAXINT)
        #     bbo.reset(seed)
        #     gc.collect()

        for r in range(cfg_clone.repeat_num):
            print('==EXP{}-{}-dir:"{}"==:'.format(r,cfg_clone.OPTM.name,cfg_clone.GENERAL.exp_dir))
            # np.random.seed(SEED)
            # random.seed(SEED)
            bbo = BBObenchmark(cfg_clone, r)

            bbo.run_one_exp()
            bbo.save_to_file(r)
            print('='*20)
            seed = rng.randint(MAXINT)
            bbo.reset(seed)
            gc.collect()

    else:
        raise NotImplementedError


if __name__ == '__main__':
    confs = {
        "./cfgs/ext_openbox_hb.yaml": ["--mark", "openbox_hb"],
        # "./cfgs/rfdehb.yaml": ["--mark", "rfdehb"],
        # "./cfgs/ext_dehb.yaml": ["--mark", "ext_dehb"],
        # "./cfgs/rfhb.yaml": ["--mark", "RFHB"],
        # "./cfgs/dehb.yaml": ["--mark", "DEHB"],
        # "./cfgs/rs.yaml": ["--mark", "RS"],
        # "./cfgs/bohb.yaml": ["--mark", "bohb"],
        # "./cfgs/hb.yaml": ["--mark", "hb"],
        # "./cfgs/ext_hb.yaml": ["--mark", "ext_hb"],
        # "./cfgs/ext_bohb.yaml": ["--mark", "ext_bohb"],
    }
    general_argv = ["-r", "50"]
    general_opts = ["TEST_PROBLEM.name", "countingones"]
    for conf in confs:
        cfg_clone = cfg.clone()
        cfg.freeze()

        argv = ["-c", conf]
        argv.extend(general_argv)
        argv.extend(confs[conf])
        argv.extend(general_opts)
        load_cfg_fom_args(cfg_clone,
                          argv=argv)  # repeat 3 times with diffent seeds
        cfg_clone.OPTM.kwargs.bracket_limit=200
        cfg_clone.OPTM.kwargs.round_limit=40
        do_experiment(cfg_clone)
        cfg.defrost()

    # cfg_clone = cfg.clone()
    # cfg.freeze()
    # load_cfg_fom_args(cfg_clone,
    #                   argv=[
    #                       "-c", "./cfgs/rfhb.yaml", "-r", "50", "--mark",
    #                       "RFHB", "TEST_PROBLEM.name", "nas_201"
    #                   ])  # repeat 3 times with diffent seeds
    # do_experiment(cfg_clone)
    # cfg.defrost()

    # cfg_clone = cfg.clone()
    # cfg.freeze()
    # load_cfg_fom_args(
    #     cfg_clone,
    #     argv=["-c", "./cfgs/dehb.yaml", "-r", "50", "--mark",
    #           "DEHB"])  # repeat 3 times with diffent seeds
    # do_experiment(cfg_clone)
    # cfg.defrost()

    # cfg_clone = cfg.clone()
    # cfg.freeze()
    # load_cfg_fom_args(
    #     cfg_clone, argv=["-c", "./cfgs/rs.yaml", "-r", "50", "--mark",
    #                      "RS"])  # repeat 3 times with diffent seeds
    # do_experiment(cfg_clone)
    # cfg.defrost()

    # # # cfg_clone = cfg.clone()
    # # # cfg.freeze()
    # # # load_cfg_fom_args(cfg_clone, argv=[                "-c",
    # # #             "./cfgs/smac3.yaml",
    # # #             "-r",
    # # #             "50",
    # # #             "--mark",
    # # #             "smac3"])  # repeat 3 times with diffent seeds

    # cfg_clone = cfg.clone()
    # cfg.freeze()
    # load_cfg_fom_args(cfg_clone,
    #                   argv=[
    #                       "-c", "./cfgs/bohb.yaml", "-r", "50", "--mark",
    #                       "bohb_array_inf"
    #                   ])  # repeat 3 times with diffent seeds
    # do_experiment(cfg_clone)
    # cfg.defrost()

    # cfg_clone = cfg.clone()
    # cfg.freeze()
    # load_cfg_fom_args(
    #     cfg_clone, argv=["-c", "./cfgs/hb.yaml", "-r", "50", "--mark",
    #                      "hb"])  # repeat 3 times with diffent seeds
    # do_experiment(cfg_clone)
    # cfg.defrost()
    marks = [ "RFHB", "hb","DEHB","bohb", "RFHB_OH", "rfdehb", "ext_hb", "ext_bohb","ext_dehb", "openbox_hb"]
    # marks = ["DEHB_DEHB", "RFHB", "hb","DEHB","bohb_array_inf", "RFHB_OH"]
    Analyse('./exp', benchmark='countingones', marks=marks, legend_size=16)
    # Analyse_multi_benchmark()
