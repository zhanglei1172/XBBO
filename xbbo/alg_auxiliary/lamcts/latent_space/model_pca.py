'''
ref: https://github.com/yangkevin2/neurips2021-lap3/
'''

import numpy as np
from sklearn.decomposition import PCA

# from lamcts_planning.util import num_params


class LatentConverterPCA:
    def __init__(self,  bounds, dim, latent_dim, rng=np.random.RandomState(), **kwargs):
        self.latent_dim = latent_dim
        self.dim = dim
        self.bounds = bounds
        self.rng = rng
        self.reset()

    def reset(self): # unclear if we need this
        self.model = PCA(n_components=self.latent_dim)

    def fit(self, inputs, **kwargs):
        """
        Given vectors in the latent space, fit the model
        inputs: batch x horizon x action, presumably small enough to just run through GPU as a single batch. 
        """
        if type(inputs)==list:
            inputs = np.stack(inputs, axis=0)
        assert inputs.shape[0] >= self.latent_dim,  "ERROR: Warning: latent dim too large for number of inputs at this tree step"
            # print('Warning: latent dim too large for number of inputs at this tree step')
            # assert False,
            # self.model = PCA(n_components=inputs.shape[0])
        self.model.fit(inputs)
    
    def encode(self, inputs):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        if shape_len == 1:
            inputs = inputs.reshape(1, -1)
        encoded = self.model.transform(inputs)
        if shape_len == 1:
            encoded = encoded.ravel()
        output = encoded
        if is_list:
            output = [o for o in output]
        return output

    def decode(self, inputs):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        if shape_len == 1:
            inputs = inputs.reshape(1, -1)
        decoded = self.model.inverse_transform(inputs)
        if shape_len == 1:
            decoded = decoded.ravel()
        output = decoded
        if is_list:
            output = [o for o in output]
        return output