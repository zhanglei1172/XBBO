'''
ref: https://github.com/yangkevin2/neurips2021-lap3/
'''
import numpy as np
# import torch
# import torch.nn as nn

# from lamcts_planning.util import num_params

# class IdentityModel(nn.Module):
#     def __init__(self):
#         super(IdentityModel, self).__init__()
#         self.reset()
    
#     def reset(self):
#         return
    
#     def forward(self, x):
#         return x

class LatentConverterIdentity:
    def __init__(self, bounds, dim, latent_dim, device='cpu', rng=np.random.RandomState(), **kwargs):
        self.device = device
        self.reset()
        self.rng = rng
        # self.latent_dim = latent_dim
        self.dim = dim
        self.latent_dim = dim
        self.bounds = bounds
        # self.latent_dim = env_info['action_dims'] * args.horizon

    def reset(self): # unclear if we need this
        pass
        # self.model = IdentityModel()

    def fit(self, inputs,**kwargs):
        """
        Given vectors in the latent space, fit the model
        inputs: batch x horizon x action, presumably small enough to just run through GPU as a single batch. 
        """
        return
    
    # @torch.no_grad()
    def encode(self, inputs):
        return inputs

    # @torch.no_grad()
    def decode(self, inputs):
        return inputs