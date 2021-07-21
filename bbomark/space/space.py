
import ConfigSpace as CS
# import ConfigSpace.hyperparameters as CSH
import numpy as np
from copy import deepcopy

from bbomark.space.warp import WARP_DICT, UNWARP_DICT

class Space(CS.ConfigurationSpace):
    def __init__(self, seed=0, meta_param=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().seed(seed)
        self.all_warp = {}
        self.meta_param = meta_param

    @staticmethod
    def warp_space(param_range, warp='linear'):
        warp_func = WARP_DICT[warp]
        return warp_func(np.asarray(param_range)).astype(np.float_, copy=False)

    @staticmethod
    def unwarp_space(param_range_warped, warp='linear'):
        unwarp_func = UNWARP_DICT[warp]
        return unwarp_func(np.asarray(param_range_warped)).astype(np.float_, copy=False)


    def warp(self, xx_list): # TODO
        '''
        xx_list: [dict]
        return: [dict]
        '''
        xx_warped_list = []
        for xx in xx_list:
            xx_warped = deepcopy(xx)
            for param_key in self.all_warp:
                warp_func = WARP_DICT[self.all_warp[param_key]]
                xx_warped[param_key] = warp_func(xx_warped[param_key])
            xx_warped_list.append(xx_warped)
        return xx_warped_list

    def unwrap(self, xx_warped_list): # TODO
        xx_list = []
        for xx_w in xx_warped_list:
            xx = deepcopy(xx_w)
            for param_key in self.all_warp:
                unwarp_func = UNWARP_DICT[self.all_warp[param_key]]
                xx[param_key] = unwarp_func(xx[param_key])
            xx_list.append(xx)
        return xx_list

    # def get_hyperparameters_dict(self, *args, **kwargs): # meta hp?

    def sample_configuration_and_unwarp(self, size=1, *args, **kwargs):
        '''
        return [:dict: {k, v}]
        '''
        config_sample_dicts = []

        config_samples = super().sample_configuration(size=size, *args, **kwargs)
        if size == 1:
            config_samples = [config_samples]
        for config_sample in config_samples: # for every suggest
            config_sample_dict = deepcopy(config_sample.get_dictionary())
            for param_key in self.meta_param:
                unwarp_func = UNWARP_DICT[self.all_warp[param_key]]
                config_sample_dict[param_key] = unwarp_func(config_sample_dict[param_key])
            config_sample_dicts.append(config_sample_dict)

        return config_sample_dicts

    @staticmethod
    def convert_configs_to_array(config_list):
        return np.array([config.get_array() for config in config_list], dtype=np.float64)

