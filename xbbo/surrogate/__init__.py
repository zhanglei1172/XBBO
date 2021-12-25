# import gpytorch
# import torch
# from gpytorch.constraints import GreaterThan
# from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.mlls import ExactMarginalLogLikelihood, DeepApproximateMLL, GammaRobustVariationalELBO
# from botorch.models import FixedNoiseGP, SingleTaskGP
# from botorch.fit import fit_gpytorch_model
# import numpy as np


# class StandardTransform:

#     def __init__(self, y: np.array):
#         assert y.ndim == 2
#         self.dim = y.shape[1]
#         self.mean = y.mean(axis=0, keepdims=True)
#         self.std = y.std(axis=0, keepdims=True)

#     def transform(self, y: np.array):
#         z = (y - self.mean) / np.clip(self.std, a_min=0.001, a_max=None)
#         return z


# def get_fitted_model(train_X, train_Y, train_Yvar, state_dict=None):
#     """
#     Get a single task GP. The model will be fit unless a state_dict with model
#         hyperparameters is provided.
#     """
#     train_Yvar = torch.full_like(train_Y, 1)
#     Y_mean = torch.Tensor([[0]])
#     # Y_mean = train_Y.mean(dim=-2, keepdim=True)
#     Y_std = torch.Tensor([[1]])
#     # Y_std = train_Y.std(dim=-2, keepdim=True)
#     model = FixedNoiseGP(train_X, (train_Y - Y_mean)/torch.clip(Y_std, min=0.001), train_Yvar)
#     # model = SingleTaskGP(train_X, (train_Y - Y_mean) / torch.clip(Y_std, min=0.001),)
#                          # likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-1)))
#     # model = SingleTaskGP(train_X, train_Y,
#     #                      likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-3)))

#     model.Y_mean = Y_mean
#     model.Y_std = Y_std
#     if state_dict is None:

#         mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)

#         # mll = GammaRobustVariationalELBO(model.likelihood, model, num_data=1000).to(train_X)
#         # with gpytorch.settings.cholesky_jitter(1e-1):
#             # try:
#         fit_gpytorch_model(mll)
#             # except:
#             # print(1)
#     else:
#         model.load_state_dict(state_dict)
#     return model
