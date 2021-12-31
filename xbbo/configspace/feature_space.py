import warnings
import ConfigSpace as CS
import numpy as np
from scipy import stats

from xbbo.configspace.abstract_feature_space import (
    AbstractFeatureSpace,
    Identity,
    Ord2Uniform,
    Cat2Onehot,
    U2gaussian,
    Category,
    Ordinal,
    U2Onehot
)

class FeatureSpace_discrete_all_oneHot(AbstractFeatureSpace):
    '''
    sparse array <=> feature space
    all variable i.i.d ~ U(0, 1)
    '''

    def __init__(self, discrete_degree):
        super().__init__()
        # self.dtypes_idx_map = dtypes_idx_map
        # self.discrete_degree = discrete_degree
        self.discrete_degree = discrete_degree
        self.unif = U2Onehot()
        self.cat = Cat2Onehot()
        self.ord = Cat2Onehot()
        self.features_ = []
        self.src_ids = []
        self.trg_ids = []
        self.dtypes_idx_map = {
            'cat': Array_idx_map(),
            'int': Array_idx_map(),
            'float': Array_idx_map(),
            'ord': Array_idx_map(),
        }

        nums, ords, cats, size_sparse, size_dense, categories = self._get_mappings()
        if nums:
            self.dtypes_idx_map['float'].src_ids, self.dtypes_idx_map['float'].trg_ids, self.dtypes_idx_map[
                'float'].cat_sizes = \
                map(np.uintp, zip(*nums))
            self.dtypes_idx_map['float'].cats = nums
        if ords:
            self.dtypes_idx_map['ord'].src_ids, self.dtypes_idx_map['ord'].trg_ids, self.dtypes_idx_map[
                'ord'].cat_sizes = \
                map(np.uintp, zip(*ords))
            self.dtypes_idx_map['ord'].cats = ords

        if cats:
            self.dtypes_idx_map['cat'].src_ids, self.dtypes_idx_map['cat'].trg_ids, self.dtypes_idx_map[
                'cat'].cat_sizes = \
                map(np.uintp, zip(*cats))
            self.dtypes_idx_map['cat'].cats = cats

        # self.nums = nums
        # self.cats = cats
        self.categories = np.asarray(categories)
        self.sparse_dimension = size_sparse
        self.dense_dimension = size_dense

    def _get_mappings(self):

        nums = []
        # nums_float = []
        # nums_ord = []
        cats = []
        ords = []
        categories = []
        src_ind = trg_ind = 0
        for src_ind, hp in enumerate(self.space.get_hyperparameters()):
            if isinstance(hp, CS.CategoricalHyperparameter):
                cat_size = hp.num_choices
                cats.append((src_ind, trg_ind, cat_size))
                trg_ind += cat_size
                categories.append(len(hp.choices))
            elif isinstance(hp, (CS.UniformIntegerHyperparameter,CS.UniformFloatHyperparameter)):
                nums.append((src_ind, trg_ind, self.discrete_degree))
                categories.append(self.discrete_degree)
                trg_ind += self.discrete_degree
            elif isinstance(hp, CS.OrdinalHyperparameter):
                categories.append(len(hp.sequence))
                cat_size = hp.num_elements
                ords.append((src_ind, trg_ind, cat_size))
                trg_ind += cat_size
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


        return nums, ords, cats, size_sparse, size_dense, categories


    def feature_to_array(self, x_feature, sparse_dim):
        '''
        return sparse array for construct Configurations
        '''
        assert not (self.dtypes_idx_map is None)
        x_array = np.empty(shape=(sparse_dim))

        for dtype, array_idx_map in self.dtypes_idx_map.items():
            if array_idx_map.src_ids is None:
                continue
            if dtype == 'cat':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    x_array[src_ind] = \
                        self.cat.feature_to_sparse_array(
                        x_feature[trg_ind:trg_ind + size], size
                    )
            elif dtype == 'ord':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    x_array[src_ind] = \
                        self.ord.feature_to_sparse_array(
                        x_feature[trg_ind:trg_ind + size], size
                    )
                # x_array[array_idx_map.src_ids] = \
                #     self.ord.feature_to_sparse_array(
                #         x_feature[array_idx_map.trg_ids],
                #     )
            elif dtype in ('float', 'int'):
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    x_array[src_ind] = \
                        self.unif.feature_to_sparse_array(
                        x_feature[trg_ind:trg_ind + size], size
                    )
            else:
                pass
        return x_array

    def array_to_feature(self, array, dense_dim): # TODO
        warnings.warn("This method may no reason be called?")
        assert not (self.dtypes_idx_map is None)
        feature = np.zeros(shape=(dense_dim))
        for dtype, array_idx_map in self.dtypes_idx_map.items():
            # if array_idx_map.src_ids:
            if array_idx_map.src_ids is None:
                continue
            if dtype == 'cat':
                # for src_ind, trg_ind, size in (array_idx_map.cats):
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    feature[trg_ind:trg_ind+size] = \
                        self.cat.sparse_array_to_feature(
                            array[src_ind], size
                        )
            elif dtype == 'ord':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    feature[trg_ind:trg_ind+size] = \
                        self.ord.sparse_array_to_feature(
                            array[src_ind], size
                        )
            elif dtype in ('float', 'int'):
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    feature[trg_ind:trg_ind+size] = \
                        self.unif.sparse_array_to_feature(
                            array[src_ind], size
                        )
            else:
                pass
        return feature

    def record_feature(self, feature):
        self.features_.append(feature)

class FeatureSpace_uniform(AbstractFeatureSpace):
    '''
    sparse array <=> feature space
    all variable i.i.d ~ U(0, 1)
    '''

    def __init__(self, dtypes_idx_map):
        super().__init__()
        self.dtypes_idx_map = dtypes_idx_map
        self.unif = Identity()
        self.cat = Cat2Onehot()
        self.ord = Ord2Uniform()
        self.features_ = []


    def feature_to_array(self, x_feature, sparse_dim):
        '''
        return sparse array for construct Configurations
        '''
        assert not (self.dtypes_idx_map is None)
        x_array = np.empty(shape=(sparse_dim))
        for dtype, array_idx_map in self.dtypes_idx_map.items():
            if array_idx_map.src_ids is None:
                continue
            if dtype == 'cat':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    x_array[src_ind] = \
                        self.cat.feature_to_sparse_array(
                        x_feature[trg_ind:trg_ind + size], size
                    )
            elif dtype == 'ord':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    x_array[src_ind] = \
                        self.ord.feature_to_sparse_array(
                        x_feature[trg_ind], size
                    )
                # x_array[array_idx_map.src_ids] = \
                #     self.ord.feature_to_sparse_array(
                #         x_feature[array_idx_map.trg_ids],
                #     )
            elif dtype in ('float', 'int'):
                x_array[array_idx_map.src_ids] = \
                    self.unif.feature_to_sparse_array(
                        x_feature[array_idx_map.trg_ids]
                    )
            else:
                pass
        return x_array

    def array_to_feature(self, array, dense_dim): # TODO
        warnings.warn("This method may no reason be called?")
        assert not (self.dtypes_idx_map is None)
        feature = np.zeros(shape=(dense_dim))
        for dtype, array_idx_map in self.dtypes_idx_map.items():
            # if array_idx_map.src_ids:
            if array_idx_map.src_ids is None:
                continue
            if dtype == 'cat':
                # for src_ind, trg_ind, size in (array_idx_map.cats):
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    feature[trg_ind:trg_ind+size] = \
                        self.cat.sparse_array_to_feature(
                            array[src_ind], size
                        )
            elif dtype == 'ord':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    feature[trg_ind] = \
                        self.ord.sparse_array_to_feature(
                            array[src_ind], size
                        )
            elif dtype in ('float', 'int'):
                feature[array_idx_map.trg_ids] = \
                    self.unif.sparse_array_to_feature(
                        array[array_idx_map.src_ids]
                    )
            else:
                pass
        return feature

    def record_feature(self, feature):
        self.features_.append(feature)

class FeatureSpace_gaussian(AbstractFeatureSpace):

    '''
    sparse array <=> feature space
    all variable i.i.d ~ Normal distribution N(0, 1)
    '''

    def __init__(self, dtypes_idx_map):
        super().__init__()
        self.dtypes_idx_map = dtypes_idx_map
        self.unif = U2gaussian()
        self.cat = Category()
        self.ord = Ordinal()
        self.features_ = []


    def feature_to_array(self, x_feature, sparse_dim):
        '''
        return sparse array for construct Configurations
        '''
        assert not (self.dtypes_idx_map is None)
        x_array = np.empty(shape=(sparse_dim))
        for dtype, array_idx_map in self.dtypes_idx_map.items():
            if array_idx_map.src_ids is None:
                continue
            if dtype == 'cat':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    x_array[src_ind] = \
                        self.cat.feature_to_sparse_array(
                        x_feature[trg_ind:trg_ind + size], size
                    )
            elif dtype == 'ord':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    x_array[src_ind] = \
                        self.ord.feature_to_sparse_array(
                        x_feature[trg_ind], size
                    )
                # x_array[array_idx_map.src_ids] = \
                #     self.ord.feature_to_sparse_array(
                #         x_feature[array_idx_map.trg_ids],
                #     )
            elif dtype in ('float', 'int'):
                x_array[array_idx_map.src_ids] = \
                    self.unif.feature_to_sparse_array(
                        x_feature[array_idx_map.trg_ids]
                    )
            else:
                pass
        return x_array

    def array_to_feature(self, array, dense_dim): # TODO
        warnings.warn("This method may no reason be called?")
        assert not (self.dtypes_idx_map is None)
        feature = np.zeros(shape=(dense_dim))
        for dtype, array_idx_map in self.dtypes_idx_map.items():
            # if array_idx_map.src_ids:
            if array_idx_map.src_ids is None:
                continue
            if dtype == 'cat':
                # for src_ind, trg_ind, size in (array_idx_map.cats):
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    feature[trg_ind:trg_ind+size] = \
                        self.cat.sparse_array_to_feature(
                            array[src_ind], size
                        )
            elif dtype == 'ord':
                for src_ind, trg_ind, size in (array_idx_map.cats):
                    feature[trg_ind] = \
                        self.ord.sparse_array_to_feature(
                            array[src_ind], size
                        )
            elif dtype in ('float', 'int'):
                feature[array_idx_map.trg_ids] = \
                    self.unif.sparse_array_to_feature(
                        array[array_idx_map.src_ids]
                    )
            else:
                pass
        return feature

    def record_feature(self, feature):
        self.features_.append(feature)

class Uniform2Gaussian(AbstractFeatureSpace):
    def __init__(self):
        super().__init__()
    
    def array_to_feature(self, array):
        return stats.norm.ppf(array)

    def feature_to_array(self, feature):
        return stats.norm.cdf(feature)