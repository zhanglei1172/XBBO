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

from xbbo.configspace.warp import WARP_DICT, UNWARP_DICT

# RERANGE_TYPES_SEQ = ['float', 'int', 'ord', 'cat']
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

class Array_idx_map():
    src_ids = None # 对应于configspace 的array索引
    trg_ids = None # 对应于array_dense的索引（trg_ids >= src_ids）
    cat_sizes = None
    cats = None

    def __len__(self):
        return len(self.src_ids) if self.src_ids else 0

class Space(CS.ConfigurationSpace):
    def __init__(self, other, warp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super().seed(seed)
        # self.all_warp = all_warp
        # self.meta_param = meta_param
        self.warp = warp
        self.add_hyperparameters(other.get_hyperparameters())
        self.dtypes_idx_map = {
            'cat': Array_idx_map(),
            'int': Array_idx_map(),
            'float': Array_idx_map(),
            'ord': Array_idx_map(),
        }

        nums, ords, cats, size_sparse, size_dense = self._get_mappings()
        for i, dtype in enumerate(('int', 'float')):
            num = nums[i]
            if num:


                self.dtypes_idx_map[dtype].src_ids, self.dtypes_idx_map[dtype].trg_ids = map(np.uintp, zip(*num))
        if ords:
            self.dtypes_idx_map['ord'].src_ids, self.dtypes_idx_map['ord'].trg_ids, self.dtypes_idx_map[
                'ord'].cat_sizes = map(np.uintp, zip(*ords))
            self.dtypes_idx_map['ord'].cats = ords

        if cats:
            self.dtypes_idx_map['cat'].src_ids, self.dtypes_idx_map['cat'].trg_ids, self.dtypes_idx_map['cat'].cat_sizes = \
                map(np.uintp, zip(*cats))
            self.dtypes_idx_map['cat'].cats = cats
        # re range ordinal
        # self.rerangeIDX = []
        # for dtype in RERANGE_TYPES_SEQ:
        #     src_ids = self.dtypes_idx_map[dtype].src_ids
        #     if src_ids is None:
        #         continue
        #     self.rerangeIDX.append(src_ids)
        # self.rerangeIDX = np.asarray(self.rerangeIDX).ravel()
        # self.rererangeIDX = self.rerangeIDX.argsort()
        
        # self.nums = nums
        # self.cats = cats
        self.size_sparse = size_sparse
        self.size_dense = size_dense

    def get_dimensions(self, sparse=False):
        '''
        size_dense = size_sparse - cat_var_num + catOnehot_encoding_length
        so: size_dense >= size_sparse
        '''
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

        nums_int = []
        nums_float = []
        # nums_ord = []
        cats = []
        ords = []

        src_ind = trg_ind = 0
        for src_ind, hp in enumerate(self.get_hyperparameters()):
            if isinstance(hp, CS.CategoricalHyperparameter):
                cat_size = hp.num_choices
                cats.append((src_ind, trg_ind, cat_size))
                trg_ind += cat_size
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                nums_int.append((src_ind, trg_ind))
                trg_ind += 1
            elif isinstance(hp, CS.UniformFloatHyperparameter):

                nums_float.append((src_ind, trg_ind))
                trg_ind += 1
            elif isinstance(hp, CS.OrdinalHyperparameter):
                cat_size = hp.num_elements
                ords.append((src_ind, trg_ind, cat_size))
                trg_ind += 1
            # elif isinstance(hp, (CS.UniformIntegerHyperparameter,
            #                      CS.UniformFloatHyperparameter,
            #                      CS.OrdinalHyperparameter)):
            #     nums.append((src_ind, trg_ind))

            else:
                raise NotImplementedError(
                    "Only hyperparameters of types "
                    "`CategoricalHyperparameter`, "
                    "`UniformIntegerHyperparameter`, "
                    "`OrdinalHyperparameter`, "
                    "`UniformFloatHyperparameter` are supported!")

        size_sparse = src_ind + 1
        size_dense = trg_ind

        return (nums_int, nums_float), ords, cats, size_sparse, size_dense

    # def add_hyperparameters(self):
    #     raise NotImplementedError("cannot modify space")


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
        super().is_valid_configuration()

    @classmethod
    def from_array(cls, configuration_space, array_sparse, dtype="float64"):
        '''
        扩展的opt给出一个suggest vector（dense array）
        需要转换成sparse array(*) => dict => unwarp dict
        '''
        assert isinstance(configuration_space, Space)
        # array_sparse = array_sparse[configuration_space.rererangeIDX]

        return cls(configuration_space=configuration_space, vector=array_sparse)

    def get_dictionary(self, *args, **kwargs):
        raise NotImplementedError('Please use .get_dict_unwarped()')

    def _get_dictionary(self, *args, **kwargs):
        return super().get_dictionary(*args, **kwargs)

    def get_dict_unwarped(self):
        dict_warped = self._get_dictionary()
        dict_unwarped = self.configuration_space.warp.unwrap(dict_warped)
        return dict_unwarped

    @staticmethod
    def array_to_dictUnwarped(cs, array_sparse):
        # cs = space.configuration_space
        config = Configurations.from_array(cs, array_sparse=array_sparse) # dense_array构造config类
        dict_warped = config._get_dictionary()
        dict_unwarped = cs.warp.unwrap(dict_warped)
        return dict_unwarped

    @staticmethod
    def dictUnwarped_to_array(cs, dict_unwarped):
        # cs = space.configuration_space
        dict_warped = cs.warp.warp(dict_unwarped) # 1->2
        config = Configurations(cs, warped_values=dict_warped) # dict构造config类
        array_sparse = config.get_array() # ->4
        return array_sparse#[cs.rerangeIDX]


