'''
config的几种表示：
1. dict_unwarped, 即model传入hyper-param的dict            *
2. dict_warped，即ConfigSpace使用的dict         config.get_dictionary()
3. spase_array，即ConfigSpace使用的vector表示（cat是int）   config.get_array()
4. dense_array, optimizer使用的vector表示（cat是one-hot） * config.to_array()
1->2->3->4
4->3->2->1
# 1<->2（在Warp类中实现）
# # 2<->3（在Space类中实现）
# 3<->4（在Configurations类中实现）
'''
import ConfigSpace as CS
# import ConfigSpace.hyperparameters as CSH
import numpy as np
from scipy.optimize import Bounds

from bbomark.configspace.warp import WARP_DICT, UNWARP_DICT
# def array_from_dict(config_space, dct):
#     '''
#     hp_dict
#     '''
#     # TODO warp
#     config = Configurations(config_space, warped_values=dct)
#     return config.get_dense_array()
#
#
# def dict_from_array(config_space, array):
#     config = Configurations.from_dense_array(config_space, array_dense=array)
#     # TODO upwarp
#     return config.get_dictionary()

class Space(CS.ConfigurationSpace):
    def __init__(self, other, warp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super().seed(seed)
        # self.all_warp = all_warp
        # self.meta_param = meta_param
        self.warp = warp
        self.add_hyperparameters(other.get_hyperparameters())

        nums, cats, size_sparse, size_dense = self._get_mappings()

        if nums:
            self.num_src, self.num_trg = map(np.uintp, zip(*nums))

        if cats:
            self.cat_src, self.cat_trg, self.cat_sizes = \
                map(np.uintp, zip(*cats))

        self.nums = nums
        self.cats = cats
        self.size_sparse = size_sparse
        self.size_dense = size_dense

    def get_dimensions(self, sparse=False):
        return self.size_sparse if sparse else self.size_dense

    def sample_configuration(self, size=1):

        config_sparse = super(Space, self) \
            .sample_configuration(size=size)

        configs_sparse_list = config_sparse if size > 1 else [config_sparse]

        configs = []
        for config in configs_sparse_list:
            configs.append(Configurations(self, warped_values=config.get_dictionary()))

        return configs

    def get_bounds(self):
        lowers = np.zeros(self.size_dense)
        uppers = np.ones(self.size_dense)

        # return list(zip(lowers, uppers))
        return Bounds(lowers, uppers)

    def _get_mappings(self):

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

        return nums, cats, size_sparse, size_dense




    # def get_hyperparameters_dict(self, *args, **kwargs): # meta hp?

    # def sample_configuration_and_unwarp(self, size=1, *args, **kwargs):
    #     '''
    #     return [:dict: {k, v}]
    #     '''
    #     config_sample_dicts = []
    #
    #     config_samples = super().sample_configuration(size=size, *args, **kwargs)
    #     if size == 1:
    #         config_samples = [config_samples]
    #     for config_sample in config_samples: # for every suggest
    #         config_sample_dict = deepcopy(config_sample.get_dictionary())
    #         for param_key in self.meta_param:
    #             unwarp_func = UNWARP_DICT[self.all_warp[param_key]]
    #             config_sample_dict[param_key] = unwarp_func(config_sample_dict[param_key])
    #         config_sample_dicts.append(config_sample_dict)
    #
    #     return config_sample_dicts

    # @staticmethod
    # def convert_configs_to_array(config_list):
    #     return np.array([config.get_array() for config in config_list], dtype=np.float64)

class Configurations(CS.Configuration):
    '''
    dense array will be used by optimizer, bound in [0, 1], cat: one-hot
    dense array = transform(feature_for_opt), cannot invert transform
    '''

    def __init__(self, configuration_space, *args, **kwargs):
        if kwargs.get('values'):
            raise NotImplementedError('make sure must input warped dict')
        if kwargs.get("warped_values"):
            kwargs['values'] = kwargs.pop('warped_values')
        assert isinstance(configuration_space, Space)
        super(Configurations, self).__init__(configuration_space,
                                                 *args, **kwargs)

    @classmethod
    def from_dense_array(cls, configuration_space, array_dense, dtype="float64"):
        '''
        扩展的opt给出一个suggest vector（dense array）
        需要转换成sparse array(*) => dict => unwarp dict
        '''
        assert isinstance(configuration_space, Space)
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

        return cls(configuration_space=configuration_space, vector=array_sparse)

    def get_dense_array(self, dtype="float64"):
        '''
        将config value的内部表示(get_array,连续值缩放到0~1，但cat非one-hot编码)
        中的cat变量展开为one-hot变量
        i.e. 稀疏表示 => dense表示，便于扩展的opt来使用
        '''

        cs = self.configuration_space
        # array_sparse = super(Configurations, self).get_array() # 0~1
        array_sparse = self.get_array() # 0~1

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

    def get_dictionary(self, *args, **kwargs):
        raise NotImplementedError('Please use .get_dict_unwarped()')

    def _get_dictionary(self, *args, **kwargs):
        return super().get_dictionary(*args, **kwargs)

    def get_dict_unwarped(self):
        dict_warped = self._get_dictionary()
        dict_unwarped = self.configuration_space.warp.unwrap(dict_warped)
        return dict_unwarped

    def denseArray_to_dictUnwarped(self, dense_array):
        cs = self.configuration_space
        config = Configurations.from_dense_array(cs, array_dense=dense_array) # dense_array构造config类
        dict_warped = config._get_dictionary()
        dict_unwarped = cs.warp.unwrap(dict_warped)
        return dict_unwarped

    def dictUnwarped_to_denseArray(self, dict_unwarped):
        cs = self.configuration_space
        dict_warped = cs.warp.warp(dict_unwarped) # 1->2
        config = Configurations(cs, warped_values=dict_warped) # dict构造config类
        array_dense = config.get_dense_array() # ->4
        return array_dense


