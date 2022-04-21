from yacs.config import CfgNode
import os, argparse, sys, time
import hashlib
import shutil
from contextlib import redirect_stdout

_C = CfgNode()

cfg = _C

_C.GENERAL = CfgNode()
_C.GENERAL.gpu = ''
_C.GENERAL.random_seed = 42
_C.GENERAL.exp_dir_root = './exp'
_C.GENERAL.pipeline = 'BBO'

_C.BBO = CfgNode()
_C.BBO.DATASETS = CfgNode()
_C.BBO.DATASETS.name = ''
_C.BBO.DATASETS.path = ''
_C.BBO.DATASETS.batch_size = 128

_C.OPTM = CfgNode()
_C.OPTM.name = 'rs' 
_C.OPTM.n_suggestions = 1
_C.OPTM.n_obj = 1 
_C.OPTM.max_call = 30
_C.OPTM.pop_size = 0 # for PBT
_C.OPTM.epoch = 30 # for PBT
_C.OPTM.interval = 0.5 # for PBT
# _C.OPTM.fraction = 0.2 # for PBT

_C.OPTM.kwargs = CfgNode(new_allowed=True)

_C.NAS = CfgNode() # TODO

_C.TEST_PROBLEM = CfgNode()
_C.TEST_PROBLEM.name = 'toy-problems' # filename
_C.TEST_PROBLEM.kwargs = CfgNode(new_allowed=True)
# _C.TEST_PROBLEM.func_evals = ('raw', 'noise') # 放非优化器优化目标的结果、metrics
# _C.TEST_PROBLEM.losses = ('val', 'test') # 必须前n_obj个为opt的优化目标
# _C.TEST_PROBLEM.metrics = ()
# _C.TEST_PROBLEM.kwargs.func_name = 'rosenbrock'

# _C.TEST_PROBLEM.kwargs = CfgNode()
_C.TEST_PROBLEM.kwargs.func_name = 'rosenbrock' # if empty, TestProblem must provide `_load_api_config` method
# _C.TEST_PROBLEM.kwargs.hp = CfgNode(new_allowed=True)
_C.TEST_PROBLEM.kwargs.dim = 30

# _C.SPACE = CfgNode()
# _C.SPACE.NAME = 'darts'
# # Loss function
# _C.SPACE.LOSS_FUN = "cross_entropy"
# # num of classes
# _C.SPACE.NUM_CLASSES = 10
# # Init channel
# _C.SPACE.CHANNEL = 16
# # number of layers
# _C.SPACE.LAYERS = 8
# # number of nodes in a cell
# _C.SPACE.NODES = 4
# # number of  PRIMITIVE
# _C.SPACE.PRIMITIVES = []
# # number of nodes in a cell
# _C.SPACE.BASIC_OP = []

def load_cfg_fom_args(cfg_, description = "Config file options.", argv=None):
    """Load config from command line arguments and set any specified options.
       How to use: python xx.py --cfg path_to_your_config.cfg test1 0 test2 True
       opts will return a list with ['test1', '0', 'test2', 'True'], yacs will compile to corresponding values
    """

    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument('-c, '"--cfg", dest="cfg_file",
                        help=help_s, required=True, type=str)
    parser.add_argument('-r, '"--repeat", dest="repeat_num",
                        help=help_s, required=True, type=int)
    parser.add_argument("--mark", dest="mark_label", default='',
                        help=help_s, required=False, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None,
                        nargs=argparse.REMAINDER)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    args = parser.parse_args(argv)
    cfg_.merge_from_file(args.cfg_file)
    cfg_.merge_from_list(args.opts)
    cfg_.repeat_num = args.repeat_num
    cfg_.mark_label = args.mark_label
    if not os.path.exists(cfg_.GENERAL.exp_dir_root):
        os.mkdir(cfg_.GENERAL.exp_dir_root)
    m = hashlib.md5(cfg_.__repr__().encode('utf-8'))
    exp_dir = time.strftime('/%Y-%m-%d__%H_%M_%S__',time.localtime(time.time()))+m.hexdigest()
    # _C.runtime = time.time()
    if cfg_.mark_label == '':
        cfg_.mark_label = '{}-{}'.format(cfg_.OPTM.name, m.hexdigest())
    cfg_.GENERAL.exp_dir = cfg_.GENERAL.exp_dir_root + exp_dir # TODO
    if os.path.exists(cfg_.GENERAL.exp_dir):
        assert False
    os.mkdir(cfg_.GENERAL.exp_dir)
    os.mkdir(cfg_.GENERAL.exp_dir + '/res')
    os.mkdir(cfg_.GENERAL.exp_dir + '/log')
    # os.mkdir(_C.GENARAL.exp_dir+'/script')
    project_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../..'
        )
    )
    shutil.copytree(project_dir +'/xbbo/', cfg_.GENERAL.exp_dir + '/scripts/')


    # _C.freeze()
    # cfg_.freeze()



    with open(os.path.join(cfg_.GENERAL.exp_dir, os.path.basename(args.cfg_file)), 'w') as f:
        with redirect_stdout(f): print(cfg_.dump())


if __name__ == '__main__':
    # print(cfg.BBO)
    print(1 if _C.NAS else 0)
    print(1 if _C.OPTM.kwargs else 0)