import numpy as np
# import matplotlib.pyplot as plt
from ConfigSpace import ConfigurationSpace
from xbbo.configspace.space import DenseConfiguration
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.conditions import LessThanCondition

# from xbbo.search_algorithm.transfer_tst_optimizer import SMBO
# from xbbo.search_algorithm.transfer_taf_optimizer import SMBO
# from xbbo.search_algorithm.transfer_rgpe_mean_optimizer import SMBO
# from xbbo.search_algorithm.transfer_taf_rgpe_optimizer import SMBO
# from xbbo.search_algorithm.transfer_RMoGP_optimizer import SMBO
from xbbo.search_algorithm.transfer_bo_optimizer import SMBO

from xbbo.problem.offline_hp import Model
from xbbo.core.constants import MAXINT
from xbbo.surrogate.transfer.base_surrogate import BaseModel

def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """

    x1 = x["x0"]
    # x2 = x["x1"]
    x2 = x.get('x1', x1)

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val

def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y

def build_space(rng):
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])
    con = LessThanCondition(x1, x0, 1.)
    cs.add_condition(con)
    return cs

def build_branin_space(rng):
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
    x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
    cs.add_hyperparameters([x1, x2])
    return cs

if __name__ == "__main__":
    MAX_CALL = 30
    rng = np.random.RandomState(42)

    test_model = Model(rng.randint(MAXINT), test_task='a6a', )

    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    confs = test_model.get_api_config()
    for conf in confs:
        cs.add_hyperparameter(UniformFloatHyperparameter(conf, confs[conf]['range'][0], confs[conf]['range'][1]))
    blackbox_func = test_model.evaluate
    base_models = []
    for i in range(len(test_model.old_D_x)):
        base_models.append(BaseModel(cs, rng=rng,do_optimize=False))
        base_models[-1].train(test_model.old_D_x[i], test_model.old_D_y[i])

    # use transfer
    # hpopt = SMBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_func='ei', weight_srategy='kernel', acq_opt='rs', base_models=base_models) # vanila bo
    # hpopt = SMBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='tst', acq_func='ei', weight_srategy='kernel', acq_opt='rs', base_models=base_models) # TST-R
    # hpopt = SMBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_func='taf', weight_srategy='kernel', acq_opt='rs', base_models=base_models) # TAF
    # hpopt = SMBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='tst', acq_func='ei', weight_srategy='rw', acq_opt='rs', base_models=base_models) # RGPE(mean)
    # hpopt = SMBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_func='taf', weight_srategy='rw', acq_opt='rs', base_models=base_models) # TAF(rw)
    hpopt = SMBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_func='mogp', weight_srategy='rw', acq_opt='rs', base_models=base_models) # RMoGP
    # not use transfer
    # hpopt = SMBO(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_opt='rs_ls', base_models=[]]) 
    # Example call of the black-box function
    def_value = blackbox_func(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate 
        value = blackbox_func(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(observe_value=value)
        hpopt.observe(trial_list=trial_list)
        
        print(value)
    
    # plt.plot(hpopt.trials.get_history()[0])
    # plt.savefig('./out/rosenbrock_bo_gp.png')
    # plt.show()
    print('find best value:{}'.format(hpopt.trials.get_best()[0]))

