'''
array:cat hp use idx [0,1,2...]
array_OH:cat hp use one-hot
array_bin:cat hp use idx and then normalize[0, 1], all var in [0,1]
ref:https://github.com/ltiao/bore/blob/c137ed5f7a2d859b0053b9c20e7ed1b744186ea8/bore/plugins/hpbandster/types.py
'''
from typing import List
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from scipy.optimize import Bounds
from ConfigSpace.conditions import ConditionComponent
from ConfigSpace.util import deactivate_inactive_hyperparameters as _deactivate_inactive_hyperparameters
# from xbbo.configspace.warp import WARP_DICT, UNWARP_DICT
from ConfigSpace.util import get_one_exchange_neighbourhood as _get_one_exchange_neighbourhood

class Bin():
    '''
    normalize to [0, 1]
    '''
    def __init__(self, src, trg, sizes) -> None:
        self.src = src
        self.trg = trg
        self.sizes = sizes
        self.bins_width = 1 / self.sizes
        self.bins = np.array([np.arange(start=0, stop=1, step=b) for b in self.bins_width])

    def convert(self, array_dense, array_sparse):
        for i in range(len(self.src)):
            array_sparse[self.src[i]] = np.searchsorted(self.bins[i], array_dense[self.trg[i]],side="right") - 1
        # array_sparse[self.src] = np.round(array_dense[self.trg]*(self.sizes-1))
        return array_sparse
    def invconvert(self, array_dense, array_sparse):
        array_dense[self.trg] = array_sparse[self.src] / self.sizes #+ self.bins_width/2
        return array_dense
    def get_bounds(self,):
        return np.zeros(len(self.trg)), np.ones(len(self.trg))

class Round():
    def __init__(self, src, trg, sizes) -> None:
        self.src = src
        self.trg = trg
        self.sizes = sizes
    def convert(self, array_dense, array_sparse):
        array_sparse[self.src] = np.round(array_dense[self.trg])
        return array_sparse
    def invconvert(self, array_dense, array_sparse):
        array_dense[self.trg] = array_sparse[self.src]
        return array_dense
    def get_bounds(self,):
        return np.zeros(len(self.trg)), np.array((self.sizes))-1

class OneHot():
    def __init__(self, src, trg, sizes) -> None:
        self.src = src
        self.trg = trg
        self.sizes = sizes
        self.cats = list(zip(self.src.tolist(),self.trg.tolist(),self.sizes.tolist()))
    def convert(self, array_dense, array_sparse):
        for src_ind, trg_ind, size in self.cats:
            tmp = array_dense[trg_ind:trg_ind + size]
            # ind_max = np.argmax(tmp) if tmp.any() else np.nan
            ind_max = np.argmax(tmp)
            array_sparse[src_ind] = ind_max
        return array_sparse
    def invconvert(self,array_dense, array_sparse):
        choice = array_sparse[self.src] # conditional=>nan
        cat_trg_offset = np.uintp(choice)
        idx = np.where(~np.isnan(choice))[0]
        array_dense[(self.trg + cat_trg_offset)[idx]] = 1
        return array_dense
    # def get_bounds(self,):
    #     return np.zeros(self.trg), np.ones(self.trg)

class Num():
    def __init__(self, src, trg) -> None:
        self.src = src
        self.trg = trg
    def convert(self, array_dense, array_sparse):
        array_sparse[self.src] = array_dense[self.trg]
        return array_sparse
    def invconvert(self, array_dense, array_sparse):
        array_dense[self.trg] = array_sparse[self.src]
        return array_dense
    def get_bounds(self,):
        return np.zeros(len(self.trg)), np.ones(len(self.trg))

class DenseConfigurationSpace(CS.ConfigurationSpace):
    def __init__(self,other: CS.ConfigurationSpace, encoding_cat, encoding_ord, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # deep-copy only the hyperparameters. conditions, clauses, seed,
        # and other metadata ignored
        self.random = other.random
        hps = other.get_hyperparameters()
        self.add_hyperparameters(hps)
        # self._dim = len(hps)
        self.encoding_cat = encoding_cat
        self.encoding_ord = encoding_ord
        self._creat_mappings()
        self._create_type()

    def __len__(self):
        return self.size_sparse

    def get_dimensions(self):
        return self.size_dense

    def sample_configuration(self, size=1):
        if size == 0:
            return []

        config = super().sample_configuration(size=size)

        configs = config if size > 1 else [config]


        return [DenseConfiguration(self, values=config.get_dictionary()) for config in configs]

    def get_bounds(self):
        dim = self.get_dimensions()
        lower = np.zeros(dim)
        upper = np.ones(dim)
        for v in self.map.values():
            if isinstance(v, OneHot):
                continue
            l, u = v.get_bounds()
            lower[v.trg] = l
            upper[v.trg] = u
        return Bounds(lower, upper)

    def _create_type(self):
        types = [0] * len(self.get_hyperparameters())
        bounds = [(np.nan, np.nan)] * len(types)

        for i, param in enumerate(self.get_hyperparameters()):
            parents = self.get_parents_of(param.name)
            if len(parents) == 0:
                can_be_inactive = False
            else:
                can_be_inactive = True

            if isinstance(param, (CSH.CategoricalHyperparameter)):
                n_cats = len(param.choices)
                if can_be_inactive:
                    n_cats = len(param.choices) + 1
                types[i] = n_cats
                bounds[i] = (int(n_cats), np.nan)

            elif isinstance(param, (CSH.OrdinalHyperparameter)):
                n_cats = len(param.sequence)
                types[i] = 0
                if can_be_inactive:
                    bounds[i] = (0, int(n_cats))
                else:
                    bounds[i] = (0, int(n_cats) - 1)

            elif isinstance(param, CSH.Constant):
                # for constants we simply set types to 0 which makes it a numerical
                # parameter
                if can_be_inactive:
                    bounds[i] = (2, np.nan)
                    types[i] = 2
                else:
                    bounds[i] = (0, np.nan)
                    types[i] = 0
                # and we leave the bounds to be 0 for now
            elif isinstance(param, CSH.UniformFloatHyperparameter):
                # Are sampled on the unit hypercube thus the bounds
                # are always 0.0, 1.0
                if can_be_inactive:
                    bounds[i] = (-1.0, 1.0)
                else:
                    bounds[i] = (0, 1.0)
            elif isinstance(param, CSH.UniformIntegerHyperparameter):
                if can_be_inactive:
                    bounds[i] = (-1.0, 1.0)
                else:
                    bounds[i] = (0, 1.0)
            elif not isinstance(param, (CSH.UniformFloatHyperparameter,
                                        CSH.UniformIntegerHyperparameter,
                                        CSH.OrdinalHyperparameter,
                                        CSH.CategoricalHyperparameter)):
                raise TypeError("Unknown hyperparameter type %s" % type(param))


        types = np.array(types, dtype=np.uint)
        bounds = np.array(bounds, dtype=object)
        self._types = types
        self._bounds = bounds

    def _creat_mappings(self):

        nums = []
        bin_cats = []
        oh_cats = []
        round_cats = []

        src_ind = trg_ind = 0
        for src_ind, hp in enumerate(self.get_hyperparameters()):
            if isinstance(hp, CS.CategoricalHyperparameter):
                cat_size = hp.num_choices
                if self.encoding_cat == 'one-hot':
                    oh_cats.append((src_ind, trg_ind, cat_size))
                    trg_ind += cat_size
                elif self.encoding_cat == 'bin':
                    bin_cats.append((src_ind, trg_ind, cat_size))
                    trg_ind += 1
                elif self.encoding_cat == 'round':
                    round_cats.append((src_ind, trg_ind, cat_size))
                    trg_ind += 1
                else:
                    raise ValueError()


            elif isinstance(hp, (CSH.OrdinalHyperparameter)):
                cat_size = len(hp.sequence)
                if self.encoding_cat == 'one-hot':
                    oh_cats.append((src_ind, trg_ind, cat_size))
                    trg_ind += cat_size
                elif self.encoding_cat == 'bin':
                    bin_cats.append((src_ind, trg_ind, cat_size))
                    trg_ind += 1
                elif self.encoding_cat == 'round':
                    round_cats.append((src_ind, trg_ind, cat_size))
                    trg_ind += 1
                else:
                    raise ValueError()
            elif isinstance(hp, (CS.UniformIntegerHyperparameter,
                                 CS.UniformFloatHyperparameter)):
                nums.append((src_ind, trg_ind))
                trg_ind += 1
            else:
                raise NotImplementedError(
                    "Only hyperparameters of types "
                    "`CategoricalHyperparameter`, "
                    "`UniformIntegerHyperparameter`, "
                    "`UniformFloatHyperparameter`,"
                    "`OrdinalHyperparameter` are supported!")

        size_sparse = src_ind + 1
        size_dense = trg_ind
        self.map = {}
        if nums:
        # num_src, num_trg = list(map(np.uintp, zip(
        #     *nums))) if nums else ([], [])
            self.map['num'] = Num(*list(map(np.uintp, zip(
            *nums))) if nums else ([], []))

        # if cats:
        # cat_src, cat_trg, cat_sizes = \
                

        # self.nums = nums
        # self.cats = cats
        if oh_cats:
            self.map['one-hot'] = OneHot(*list(map(np.uintp, zip(*oh_cats))) if oh_cats else ([], [],[]))
        # else:
        #     self.onehot = None
        if bin_cats:
            self.map['bin'] = Bin(*list(map(np.uintp, zip(*bin_cats))) if bin_cats else ([], [],[]))
        # else:
        #     self.bin = None
        if round_cats:
            # raise NotImplementedError
            self.map['round'] = Round(*list(map(np.uintp, zip(*round_cats))) if round_cats else ([], [],[]))
        # else:
        #     self.round = None
        self.size_sparse = size_sparse
        self.size_dense = size_dense


class DenseConfiguration(CS.Configuration):

    def __init__(self, configuration_space: DenseConfigurationSpace, *args,
                 **kwargs):
        '''
        values:dict
        vector:inner representation
        '''
        assert isinstance(configuration_space, DenseConfigurationSpace)
        # if 'vector' in kwargs and configuration_space.cats:
        #     kwargs['vector'][configuration_space.cat_src] = np.round(
        #         kwargs['vector'][configuration_space.cat_src])
        super(DenseConfiguration, self).__init__(configuration_space, *args,
                                                 **kwargs)

    @classmethod
    def from_dict(cls, configuration_space, dictionary):
        return cls(configuration_space=configuration_space, values=dictionary)

    @classmethod
    def from_array(cls,
                         configuration_space,
                         array_dense,
                         dtype="float64",**kwargs):
        '''
        扩展的opt给出一个suggest vector（dense array）
        需要转换成sparse array(*) => dict => unwarp dict
        '''
        cs = configuration_space
        # initialize output array
        array_sparse = np.zeros(cs.size_sparse, dtype=dtype)
        for v in cs.map.values():
            array_sparse = v.convert(array_dense, array_sparse)
        # process numerical hyperparameters
        # if cs.num:
        #     array_sparse = cs.num.convert(array_dense, array_sparse)
        # # process categorical hyperparameters
        # if cs.round:
        #     array_sparse = cs.round.convert(array_dense, array_sparse)
        # if cs.bin:
        #     array_sparse = cs.bin.convert(array_dense, array_sparse)
        # if cs.onehot:
        #     array_sparse = cs.onehot.convert(array_dense, array_sparse)
        


        return cls(configuration_space=configuration_space,
                   vector=array_sparse)

    @staticmethod
    def array_to_dict(cs, array_dense):
        # cs = space.configuration_space
        config = DenseConfiguration.from_array(
            cs, array_dense=array_dense)  # dense_array构造config类
        dict = config.get_dictionary()
        return dict

    @staticmethod
    def dict_to_array(cs, dict):
        config = DenseConfiguration(cs, values=dict)  # dict构造config类
        array_dense = config.get_array(sparse=False)  # ->4
        return array_dense  #[cs.rerangeIDX]

    def get_array(self, sparse=True, dtype="float64"):

        cs = self.configuration_space
        array_sparse = super(DenseConfiguration, self).get_array()
        if sparse:
            return array_sparse
        # initialize output array
        # TODO(LT): specify `dtype` flexibly
        array_dense = np.zeros(cs.size_dense, dtype=dtype)

        # process numerical hyperparameters
        # if cs.nums:
        #     array_dense[cs.num_trg] = array_sparse[cs.num_src]

        # # process categorical hyperparameters
        # if cs.cats:
        #     choice = array_sparse[cs.cat_src] # conditional=>nan
        #     cat_trg_offset = np.uintp(choice)
        #     array_dense[cs.cat_trg + cat_trg_offset] = 1 if choice else choice
        # # process numerical hyperparameters
        # if cs.num:
        #     array_dense = cs.num.invconvert(array_dense, array_sparse)
        # # process categorical hyperparameters
        # if cs.round:
        #     array_dense = cs.round.invconvert(array_dense, array_sparse)
        # if cs.bin:
        #     array_dense = cs.bin.invconvert(array_dense, array_sparse)
        # if cs.onehot:
        #     array_dense = cs.onehot.invconvert(array_dense, array_sparse)
        for v in cs.map.values():
            array_dense = v.invconvert(array_dense, array_sparse)
        return array_dense
 


def convert_denseConfigurations_to_array(
        configs: List[DenseConfiguration]) -> np.ndarray:
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
    configs_array = np.array([config.get_array(sparse=False) for config in configs],
                             dtype=np.float64)
    # configuration_space = configs[0].configuration_space
    return configs_array
    # return impute_default_values(configuration_space, configs_array)


def impute_default_values(configuration_space: DenseConfigurationSpace,
                          configs_array: np.ndarray) -> np.ndarray:
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


def deactivate_inactive_hyperparameters(*arg, **kwargs):
    configuration = _deactivate_inactive_hyperparameters(*arg, **kwargs)
    # return DenseConfiguration(configuration_space=configuration.configuration_space,values=configuration.get_dictionary())
    return convert_denseConfiguration(configuration)


def convert_denseConfiguration(conf: CS.Configuration):
    # return DenseConfiguration(configuration_space=conf.configuration_space,values=conf.get_dictionary())
    conf.__class__ = DenseConfiguration
    return conf


def get_one_exchange_neighbourhood(*args, **kwargs):
    configs = map(
        convert_denseConfiguration,
        _get_one_exchange_neighbourhood(*args,
                                        stdev=0.05,
                                        num_neighbors=8,
                                        **kwargs))
    return configs