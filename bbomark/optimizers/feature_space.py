import numpy as np

from bbomark.core.abstract_feature_space import (
    AbstractFeatureSpace,
    Gaussian,
    U2gaussian,
    Ordinal,
    Category
)



class FeatureSpace_nevergrad(AbstractFeatureSpace):
    '''
    dense array i.i.d ~ U(0, 1)
    all variable i.i.d ~ Normal distribution N(0, 1)
    '''

    def __init__(self):
        super().__init__()
        self.dtypes_idx_map = None
        self.u2g = U2gaussian()
        self.cat = Category()
        self.ord = Ordinal()


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
                    self.u2g.feature_to_sparse_array(
                        x_feature[array_idx_map.trg_ids]
                    )
            else:
                pass
        return x_array

    def array_to_feature(self, array, dense_dim): # TODO
        assert not (self.dtypes_idx_map is None)
        x_array = np.zeros(shape=(dense_dim))
        for dtype, array_idx_map in self.dtypes_idx_map:
            if array_idx_map.src_ids:
                if dtype == 'cat':
                    # for src_ind, trg_ind, size in (array_idx_map.cats):
                    x_array[array_idx_map.trg_ids +\
                            self.cat.sparse_array_to_feature(
                        array[trg_ind:trg_ind + size],
                    )] = 1
                elif dtype == 'ord':
                    x_array[array_idx_map.trg_ids] = \
                        self.ord.sparse_array_to_feature(
                            array[array_idx_map.src_ids]
                        )
                elif dtype in ('float', 'int'):
                    x_array[array_idx_map.trg_ids] = \
                        self.u2g.sparse_array_to_feature(
                            array[array_idx_map.src_ids]
                        )
                else:
                    pass
        return x_array

    def record_feature(self, feature):
        self.features.append(feature)