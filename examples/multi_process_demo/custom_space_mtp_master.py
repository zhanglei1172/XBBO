import time
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from xbbo.search_algorithm.bo_optimizer import BO
from xbbo.utils.message_queue.master_messager import MasterMessager


def custom_search_space():
    '''
    define search space
    '''
    configuration_space = ConfigurationSpace()
    configuration_space.add_hyperparameter(UniformFloatHyperparameter('x', -10, 10, default_value=-3))
    return configuration_space

if __name__ == "__main__":
    MAX_CALL = 30
    WAITING_WORKER_TIME = None
    master_messager = MasterMessager('127.0.0.1', 5678, b'abc', 1, 1)
    cs = custom_search_space()

    # specify black box optimizer
    hpopt = BO(space=cs, suggest_limit=MAX_CALL)
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest() # defalut suggest one trial
        msg = trial_list[0].config_dict
        master_messager.send_message((msg, WAITING_WORKER_TIME))
        # # evaluate 
        # obs = custom_black_box_func()
        # observe
        observation = master_messager.receive_message()
        trial_list[0].add_observe_value(observation)
        hpopt.observe(trial_list=trial_list)
        
        print(observation)
    
    print('find best (value, config):{}'.format(hpopt.trials.get_best()))