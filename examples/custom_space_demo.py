from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from xbbo.search_algorithm.bo_optimizer import BO

def custom_black_box_func(config):
    '''
    define black box function:
    y = x^2
    '''
    return config['x'] ** 2

def custom_search_space():
    '''
    define search space
    '''
    configuration_space = ConfigurationSpace()
    configuration_space.add_hyperparameter(UniformFloatHyperparameter('x', -10, 10, default_value=-3))
    return configuration_space

if __name__ == "__main__":
    MAX_CALL = 30

    cs = custom_search_space()

    # specify black box optimizer
    hpopt = BO(space=cs, suggest_limit=MAX_CALL)
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest() # defalut suggest one trial
        # evaluate 
        obs = custom_black_box_func(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(obs)
        hpopt.observe(trial_list=trial_list)
        
        print(obs)
    
    print('find best (value, config):{}'.format(hpopt.trials.get_best()))