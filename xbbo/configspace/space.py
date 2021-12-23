'''
array_sparse:cat hp use idx
array_dense:cat hp use onr-hot
ref:https://github.com/ltiao/bore/blob/c137ed5f7a2d859b0053b9c20e7ed1b744186ea8/bore/plugins/hpbandster/types.py
'''
from typing import List
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from scipy.optimize import Bounds
from ConfigSpace.util import deactivate_inactive_hyperparameters as _deactivate_inactive_hyperparameters
# from xbbo.configspace.warp import WARP_DICT, UNWARP_DICT



class DenseConfigurationSpace(CS.ConfigurationSpace):
    # def __init__(self, other, warp, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.add_hyperparameters(other.get_hyperparameters())
    #     # super().seed(seed)
    #     # self.all_warp = all_warp
    #     # self.meta_param = meta_param
    #     self.warp = warp
    def add_hyperparameters(self, hyperparameters: List[CSH.Hyperparameter]) -> List[CSH.Hyperparameter]:
        for hp in hyperparameters:
            self.add_hyperparameter(hp)
        return hyperparameters

    def add_hyperparameter(
            self, hyperparameter: CSH.Hyperparameter) -> CSH.Hyperparameter:
        hyperparameter = super().add_hyperparameter(hyperparameter)
        self._creat_mappings()
        return hyperparameter

    def get_dimensions(self, sparse=False):
        '''
        size_dense = size_sparse - cat_var_num + catOnehot_encoding_length
        so: size_dense >= size_sparse
        '''
        return self.size_sparse if sparse else self.size_dense

    def sample_configuration(self, size=1):

        config_sparse = super(DenseConfigurationSpace, self) \
            .sample_configuration(size=size)

        configs_sparse_list = config_sparse if size > 1 else [config_sparse]

        configs = []
        for config in configs_sparse_list:
            configs.append(
                DenseConfiguration(self,
                                   values=config.get_dictionary()))

        return configs

    def get_bounds(self):
        lowers = np.zeros(self.size_dense)
        uppers = np.ones(self.size_dense)

        # return list(zip(lowers, uppers))
        return Bounds(lowers, uppers)

    def _creat_mappings(self):

        nums = []
        cats = []

        src_ind = trg_ind = 0
        for src_ind, hp in enumerate(self.get_hyperparameters()):
            if isinstance(hp, CS.CategoricalHyperparameter):
                cat_size = hp.num_choices
                cats.append((src_ind, trg_ind, cat_size))
                trg_ind += cat_size
            elif isinstance(hp, (CS.UniformIntegerHyperparameter,
                                 CS.UniformFloatHyperparameter)):
                nums.append((src_ind, trg_ind))
                trg_ind += 1
            else:
                raise NotImplementedError(
                    "Only hyperparameters of types "
                    "`CategoricalHyperparameter`, "
                    "`UniformIntegerHyperparameter`, "
                    "`UniformFloatHyperparameter` are supported!")

        size_sparse = src_ind + 1
        size_dense = trg_ind
        if nums:
            self.num_src, self.num_trg = map(np.uintp, zip(*nums))

        if cats:
            self.cat_src, self.cat_trg, self.cat_sizes = \
                map(np.uintp, zip(*cats))

        self.nums = nums
        self.cats = cats
        self.size_sparse = size_sparse
        self.size_dense = size_dense


class DenseConfiguration(CS.Configuration):
    '''
    dense array will be used by optimizer, bound in [0, 1], cat: one-hot
    '''
    @staticmethod
    def sparse_array_to_dict(cs, array_sparse):
        # cs = space.configuration_space
        config = DenseConfiguration(
            cs, vector=array_sparse)  # dense_array构造config类
        dict = config.get_dictionary()
        return dict

    @staticmethod
    def dict_to_sparse_array(cs, dict):
        config = DenseConfiguration(cs, values=dict)  # dict构造config类
        array_sparse = config.get_array()  # ->4
        return array_sparse  #[cs.rerangeIDX]

    @classmethod
    def from_sparse_array(cls, configuration_space, array_sparse):
        return cls(configuration_space=configuration_space,
                   vector=array_sparse)

    def get_sparse_array(self,):
        return self.get_array()

    def __init__(self, configuration_space: DenseConfigurationSpace, *args,
                 **kwargs):
        '''
        values:dict
        vector:inner representation
        '''
        assert isinstance(configuration_space, DenseConfigurationSpace)
        super(DenseConfiguration, self).__init__(configuration_space, *args,
                                                 **kwargs)

    @classmethod
    def from_dict(cls, configuration_space, dictionary):
        return cls(configuration_space=configuration_space, values=dictionary)

    @classmethod
    def from_dense_array(cls,
                         configuration_space,
                         array_dense,
                         dtype="float64"):
        '''
        扩展的opt给出一个suggest vector（dense array）
        需要转换成sparse array(*) => dict => unwarp dict
        '''
        cs = configuration_space
        # initialize output array
        array_sparse = np.empty(cs.size_sparse, dtype=dtype)

        # process numerical hyperparameters
        if cs.nums:
            array_sparse[cs.num_src] = array_dense[cs.num_trg]

        # process categorical hyperparameters
        for src_ind, trg_ind, size in cs.cats:
            ind_max = np.argmax(array_dense[trg_ind:trg_ind + size])
            array_sparse[src_ind] = ind_max

        return cls(configuration_space=configuration_space,
                   vector=array_sparse)

    @staticmethod
    def dense_array_to_dict(cs, array_dense):
        # cs = space.configuration_space
        config = DenseConfiguration.from_dense_array(
            cs, array_dense=array_dense)  # dense_array构造config类
        dict = config.get_dictionary()
        return dict

    @staticmethod
    def dict_to_dense_array(cs, dict):
        config = DenseConfiguration(cs, values=dict)  # dict构造config类
        array_dense = config.get_dense_array()  # ->4
        return array_dense  #[cs.rerangeIDX]

    def get_dense_array(self, dtype="float64"):

        cs = self.configuration_space
        array_sparse = super(DenseConfiguration, self).get_array()

        # initialize output array
        # TODO(LT): specify `dtype` flexibly
        array_dense = np.zeros(cs.size_dense, dtype=dtype)

        # process numerical hyperparameters
        if cs.nums:
            array_dense[cs.num_trg] = array_sparse[cs.num_src]

        # process categorical hyperparameters
        if cs.cats:
            cat_trg_offset = np.uintp(array_sparse[cs.cat_src])
            array_dense[cs.cat_trg + cat_trg_offset] = 1

        return array_dense

def convert_denseConfigurations_to_array(configs: List[DenseConfiguration]) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_sparse_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].configuration_space
    return impute_default_values(configuration_space, configs_array)


def impute_default_values(
        configuration_space: DenseConfigurationSpace,
        configs_array: np.ndarray
) -> np.ndarray:
    """Impute inactive hyperparameters in configuration array with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configuration_space : ConfigurationSpace
    
    configs_array : np.ndarray
        Array of configurations.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    for hp in configuration_space.get_hyperparameters():
        default = hp.normalized_default_value
        idx = configuration_space.get_idx_by_hyperparameter_name(hp.name)
        nonfinite_mask = ~np.isfinite(configs_array[:, idx])
        configs_array[nonfinite_mask, idx] = default

    return configs_array

def deactivate_inactive_hyperparameters(*arg,**kwargs):
    configuration = _deactivate_inactive_hyperparameters(*arg,**kwargs)
    return DenseConfiguration(configuration_space=configuration.configuration_space,values=configuration.get_dictionary())

def convert_denseConfiguration(conf:CS.Configuration):
    return DenseConfiguration(configuration_space=conf.configuration_space,values=conf.get_dictionary())
