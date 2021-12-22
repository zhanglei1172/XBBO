import glob
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from matplotlib import pyplot as plt
from botorch.optim import optimize_acqf
import tqdm, random

from xbbo.acquisition_function import mogp
from xbbo.acquisition_function.ei import EI
from xbbo.acquisition_function.mogp import MoGP_
from xbbo.acquisition_function.taf import TAF_
from xbbo.configspace.feature_space import FeatureSpace_uniform
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import Configurations

from xbbo.core.trials import Trials
from xbbo.surrogate import get_fitted_model
from xbbo.surrogate.gaussian_process import GaussianProcessRegressor, GaussianProcessRegressorARD_gpy, \
    GaussianProcessRegressorARD_torch
from xbbo.surrogate.rgpe import RGPE_mean_surrogate_
from xbbo.surrogate.tst import TST_surrogate_



class SMBO(AbstractOptimizer, FeatureSpace_uniform):

    def __init__(self,
                 config_spaces,
                 min_sample=4,
                 noise_std=0.01,
                 rho=0.75,
                 bandwidth=0.9,
                 mc_samples=256,
                 raw_samples=100,
                 purn=True,
                 alpha=0
                 # avg_best_idx=2.0,
                 # meta_data_path=None,
                 ):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_uniform.__init__(self, self.space.dtypes_idx_map)
        self.min_sample = min_sample
        self.candidates = None
        # self.avg_best_idx = avg_best_idx
        # self.meta_data_path = meta_data_path
        configs = self.space.get_hyperparameters()
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)

        self.hp_num = len(configs)
        self.bounds_tensor = torch.stack([
            torch.zeros(self.hp_num),
            torch.ones(self.hp_num)
        ])
        self.trials = Trials()
        # self.surrogate = GaussianProcessRegressor(self.hp_num)
        self.surrogate = GaussianProcessRegressorARD_torch(self.hp_num, self.min_sample)
        self.acq_class = MoGP_
        self.noise_std = noise_std
        self.rho = rho
        self.bandwidth = bandwidth
        self.mc_samples = mc_samples
        self.raw_samples = raw_samples
        self.purn = purn
        self.alpha = alpha

    def prepare(self, old_D_x_params, old_D_y, new_D_x_param, sort_idx=None, params=True):
        if params:
            old_D_x = []
            for insts_param in old_D_x_params:
                insts_feature = []
                for inst_param in insts_param:
                    array = Configurations.dictUnwarped_to_array(self.space, inst_param)
                    insts_feature.append(self.array_to_feature(array, self.dense_dimension))
                old_D_x.append(np.asarray(insts_feature))
            insts_feature = []
            if new_D_x_param:
                for inst_param in new_D_x_param:
                    array = Configurations.dictUnwarped_to_array(self.space, inst_param)
                    insts_feature.append(self.array_to_feature(array, self.dense_dimension))
                new_D_x = (np.asarray(insts_feature))
                self.candidates = new_D_x
            else:
                self.candidates = None
        else:
            old_D_x = []
            for insts_param in old_D_x_params:
                # insts_feature = []
                old_D_x.append(insts_param[:, sort_idx])

            if new_D_x_param is not None:
                new_D_x = new_D_x_param[:, sort_idx]
                self.candidates = new_D_x
            else:
                self.candidates = None

        self.old_D_num = len(old_D_x)
        self.gps = []
        self.base_model_best = []
        for d in range(self.old_D_num):
            # self.gps.append(GaussianProcessRegressor())
            # observed_idx = np.random.randint(0, len(old_D_y[d]), size=50)
            # observed_idx = np.random.randint(0, len(old_D_y[d]), size=len())
            observed_idx = list(range(len(old_D_y[d])))
            # observed_idx = np.random.choice(len(old_D_y[d]), size=50, replace=False)
            x = torch.Tensor(old_D_x[d][observed_idx, :])  # TODO
            y = torch.Tensor(old_D_y[d][observed_idx])
            train_yvar = torch.full_like(y, self.noise_std ** 2)
            self.gps.append(get_fitted_model(x, y, train_yvar))
            self.base_model_best.append(np.inf)
            # g = GaussianProcessRegressorARD_torch(self.hp_num)
            # self.gps.append(g.fit(x, y.squeeze()))
        print(1)
        # if new_D_x is not None:
        #     candidates = new_D_x
        # else:  #
        #     raise NotImplemented
        # self.candidates = candidates

    def kendallTauCorrelation(self, base_model_means, y):
        if y is None or len(y) < 2:
            return torch.full(base_model_means.shape[0], self.rho)
        rank_loss = (base_model_means.unsqueeze(-1) < base_model_means.unsqueeze(-2)) ^ (
                y.unsqueeze(-1) < y.unsqueeze(-2))
        t = rank_loss.float().mean(dim=-1).mean(dim=-1) / self.bandwidth
        return (t < 1) * (1 - t * t) * self.rho
        # return self.rho * (1 - t * t) if t < 1 else 0

    def suggest(self, n_suggestions=1):
        # 只suggest 一个
        if (self.trials.trials_num) < self.min_sample:
            # raise NotImplemented
            return self._random_suggest()
        else:
            x_unwarpeds = []
            sas = []
            for n in range(n_suggestions):
                acq = self.acq_class(self.surrogate.gpr, self.surrogate.z_observed.min(), self.gps,
                                     self.base_model_best,
                                     self.weights, maximize=False)
                # acq = self.acq_class(self.surrogate.gpr,
                #                      self.surrogate.transform_outputs(
                #                          np.asarray(self.trials.history_y)[..., None]).min().astype(np.float32), maximize=False)
                if self.candidates is None:
                    # optimize
                    candidate, acq_value = optimize_acqf(
                        acq, bounds=self.bounds_tensor, q=1, num_restarts=5, raw_samples=self.raw_samples,
                    )
                    suggest_array = candidate[0].detach().cpu().numpy()
                    x_array = self.feature_to_array(suggest_array, self.sparse_dimension)
                    x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)

                    sas.append(suggest_array)
                    x_unwarpeds.append(x_unwarped)
                else:
                    with torch.no_grad():
                        ei = acq(torch.Tensor(self.candidates).unsqueeze(dim=-2))
                    rm_id = ei.argmax()
                    suggest_array = self.candidates[rm_id]
                    self.candidates = np.delete(self.candidates, rm_id, axis=0)  # TODO
                    x_array = self.feature_to_array(suggest_array, self.sparse_dimension)
                    x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)

                    sas.append(suggest_array)
                    x_unwarpeds.append(x_unwarped)
        # x = [Configurations.array_to_dictUnwarped(self.space,
        #                                           np.asarray(sa)) for sa in sas]
        self.trials.params_history.extend(x_unwarpeds)
        return x_unwarpeds, sas

    def observe(self, x, y):
        # print(y)
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.surrogate.fit(x, y)
        self.trials.trials_num += 1
        if len(self.trials.history_y) < self.min_sample:
            return
        self.is_fited = True
        self.x = torch.Tensor(self.trials.history)
        self.y = torch.Tensor(self.trials.history_y).unsqueeze(-1)
        with torch.no_grad():
            for d in range(len(self.gps)):
                self.base_model_best[d] = self.gps[d].posterior(self.x).mean.min()
        self.weights = self._get_weight(self.x, self.y.squeeze())
        # self.kendallTauCorrelation(base_model_means.squeeze(), self.y.squeeze())

    def _delete_noise_knowledge(self, ranking_losses):
        if self.purn:
            p_drop = 1 - (1 - self.x.shape[0] / 30) * (ranking_losses[:-1, :] < ranking_losses[-1, :]).sum(axis=-1) / (
                    self.mc_samples * (1 + self.alpha))
            drop_mask = np.random.binomial(1, p_drop) == 1
            # target_loss = ranking_losses[-1]
            # threshold = np.percentile(target_loss, 95)
            # mask_to_remove = np.median(ranking_losses, axis=1) > threshold  # 中位数样本loss 比 95% 的target model结果差
            idx_to_remove = np.argwhere(drop_mask).ravel()
            ranking_losses = np.delete(ranking_losses, idx_to_remove, axis=0)
            # ranking_losses[idx_to_remove] = self.rank_sample_num
            for idx in reversed(idx_to_remove):  # TODO Remove permanent
                self.gps.pop(idx)
            self.old_D_num -= len(idx_to_remove)
        return ranking_losses

    def _compute_ranking_loss(self, f_samples, f_target):
        '''
        f_samples 'n_samples x (n) x n' -dim
        '''
        if f_samples.ndim == 3:  # for target model
            rank_loss = ((f_samples.diagonal(dim1=-2, dim2=-1)[:, :, None] < f_samples) ^ (
                    torch.unsqueeze(f_target, dim=-1) < torch.unsqueeze(f_target, dim=-2)
            )).sum(dim=-1).sum(dim=-1)
        else:
            rank_loss = ((torch.unsqueeze(f_samples, dim=-1) < torch.unsqueeze(f_samples, dim=-2)) ^ (
                    torch.unsqueeze(f_target, dim=-1) < torch.unsqueeze(f_target, dim=-2)
            )).sum(dim=-1).sum(dim=-1)
        return rank_loss

    def _get_min_index(self, array):
        best_model_idxs = np.zeros(array.shape[1], dtype=np.int64)
        is_best_model = (array == array.min(axis=0))
        # idxs = np.argwhere(is_best_model)
        # mod, samp = idxs[:,0], idxs[:, 1]
        # np.random.randint(np.bincount())
        for i in range(array.shape[1]):
            if is_best_model[-1, i]:
                best_model_idxs[i] = self.old_D_num
            else:
                best_model_idxs[i] = np.random.choice(np.argwhere(is_best_model[:, i]).ravel().tolist())
        return best_model_idxs

    def _get_weight(self, t_x, t_y):
        ranking_losses = []
        with torch.no_grad():
            for d in range(self.old_D_num):
                posterior = self.gps[d].posterior(t_x)
                sampler = SobolQMCNormalSampler(num_samples=self.mc_samples)
                base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
                ranking_losses.append(self._compute_ranking_loss(base_f_samps, t_y))
        ranking_losses.append(self._compute_ranking_loss(self._get_loocv_preds(t_x, t_y.unsqueeze(-1)), t_y))
        ranking_losses = torch.stack(ranking_losses).numpy()
        ranking_losses = self._delete_noise_knowledge(ranking_losses)
        # TODO argmin 多个最小处理
        # best_model_idxs = ranking_losses.argmin(axis=0)  # per sample都有一个best idx
        best_model_idxs = self._get_min_index(ranking_losses)

        # rank_weight = np.bincount(best_model_idxs, minlength=ranking_losses.shape[0]) / self.rank_sample_num
        rank_weight = np.bincount(best_model_idxs, minlength=ranking_losses.shape[0])

        return (rank_weight / rank_weight.sum())

    def _get_loocv_preds(self, x, y):
        try_num = len(y)
        masks = ~torch.eye(try_num, dtype=torch.bool)
        x_cv = ([x[m] for m in masks])
        y_cv = ([y[m] for m in masks])
        samples = []
        state_dict = self.surrogate.gpr.state_dict()
        # expand to batch size of batch_mode LOOCV model
        # state_dict_expanded = {
        #     name: t.expand(try_num, *[-1 for _ in range(t.ndim)])
        #     for name, t in state_dict.items()
        # }
        with torch.no_grad():
            for i in range(len(y)):
                # kernel = self.new_gp.kernel.copy()
                model = get_fitted_model(x_cv[i], y_cv[i], None, state_dict=state_dict)

                posterior = model.posterior(x)
                sampler = SobolQMCNormalSampler(num_samples=self.mc_samples)
                samples.append(sampler(posterior).squeeze(-1))
            return torch.stack(samples, dim=1)

    def _random_suggest_explore(self, n_suggestions=1):
        sas = []
        x_unwarpeds = []
        for n in range(n_suggestions):
            rm_id = np.random.choice(len(self.candidates))
            sas.append(self.candidates[rm_id])
            x_array = self.feature_to_array(sas[-1], self.sparse_dimension)
            x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            x_unwarpeds.append(x_unwarped)
            self.candidates = np.delete(self.candidates, rm_id, axis=0)
        return x_unwarpeds, sas

    def _random_suggest(self, n_suggestions=1):
        sas = []
        x_unwarpeds = []
        if self.candidates is not None:
            for n in range(n_suggestions):
                rm_id = np.random.randint(low=0, high=len(self.candidates))
                sas.append(self.candidates[rm_id])
                x_array = self.feature_to_array(sas[-1], self.sparse_dimension)
                x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
                x_unwarpeds.append(x_unwarped)
                self.candidates = np.delete(self.candidates, rm_id, axis=0)  # TODO
        else:
            x_unwarpeds = (self.space.sample_configuration(n_suggestions))
            for n in range(n_suggestions):
                array = Configurations.dictUnwarped_to_array(self.space, x_unwarpeds[-1])
                sas.append(self.array_to_feature(array, self.dense_dimension))
        return x_unwarpeds, sas


opt_class = SMBO
