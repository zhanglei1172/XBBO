import abc
import csv
from curses import A_ATTRIBUTES
import typing
import numpy as np
from xbbo.core.trials import Trials

# from xbbo.surrogate.base import Surrogate
from xbbo.surrogate.gaussian_process import GPR_sklearn
from xbbo.surrogate.transfer.tst import BaseModel
from xbbo.utils.constants import VERY_SMALL_NUMBER

class ABCWeightStategy(metaclass=abc.ABCMeta):
    def __init__(self,
                 cs,
                 base_models,
                 target_model,
                 rng,**kwargs):
        self.space = cs
        self.base_models = base_models
        self.target_model = target_model
        self.rng = rng
    @abc.abstractclassmethod
    def get_weight(self, trials: Trials):
        raise NotImplementedError()




class RankingWeight(ABCWeightStategy):
    def __init__(self,
                 cs,
                 base_models,
                 target_model,
                 rng,
                 budget,
                 rank_sample_num=256,
                 is_purn=False,alpha=0, **kwargs):
        super().__init__(cs, base_models, target_model, rng, **kwargs)
        self.rank_sample_num = rank_sample_num
        # self.iter = 0
        self.budget = budget
        self.is_purn = is_purn
        self.alpha = alpha


    def _compute_ranking_loss(self, f_samples, f_target):
        '''
        f_samples 'n_samples x (n) x n' -dim
        '''
        if f_samples.ndim == 3:  # for target model
            rank_loss = ((f_samples.diagonal(axis1=-2, axis2=-1)[:, :, None] <
                          f_samples) ^
                         (np.expand_dims(f_target, axis=-1) < np.expand_dims(
                             f_target, axis=-2))).sum(axis=-1).sum(axis=-1)
        else:
            rank_loss = ((np.expand_dims(f_samples, axis=-1) < np.expand_dims(
                f_samples, axis=-2)) ^
                         (np.expand_dims(f_target, axis=-1) < np.expand_dims(
                             f_target, axis=-2))).sum(axis=-1).sum(axis=-1)
        return rank_loss

    def _get_min_index(self, array):
        best_model_idxs = np.zeros(array.shape[1], dtype=np.int64)
        is_best_model = (array == array.min(axis=0))
        # idxs = np.argwhere(is_best_model)
        # mod, samp = idxs[:,0], idxs[:, 1]
        # self.rng.randint(np.bincount())
        for i in range(array.shape[1]):
            if is_best_model[-1, i]:
                best_model_idxs[i] = len(self.base_models)
            else:
                best_model_idxs[i] = self.rng.choice(
                    np.argwhere(is_best_model[:, i]).ravel().tolist())
        return best_model_idxs

    def get_weight(self, trials: Trials):
        t_x = trials.get_array()
        t_y = np.asarray(trials._his_observe_value)
        self.try_num = len(t_y)
        ranking_losses = []
        for base_model in self.base_models:
            mean, cov = base_model._predict(t_x, "full_cov")
            ranking_losses.append(
                self._compute_ranking_loss(
                    self.rng.multivariate_normal(mean,
                                                 cov,
                                                 size=self.rank_sample_num),
                    t_y))
        ranking_losses.append(
            self._compute_ranking_loss(self._get_loocv_preds(t_x, t_y), t_y))
        ranking_losses = np.array(ranking_losses)
        ranking_losses = self._delete_noise_knowledge(ranking_losses)
        # TODO argmin 多个最小处理
        # best_model_idxs = ranking_losses.argmin(axis=0)  # per sample都有一个best idx
        best_model_idxs = self._get_min_index(ranking_losses)

        # rank_weight = np.bincount(best_model_idxs, minlength=ranking_losses.shape[0]) / self.rank_sample_num
        rank_weight = np.bincount(best_model_idxs,
                                  minlength=ranking_losses.shape[0])

        return rank_weight  #/ rank_weight.sum()

    def _get_loocv_preds(self, x, y):
        masks = ~np.eye(self.try_num, dtype=np.bool_)
        x_cv = [x[m] for m in masks]
        y_cv = [y[m] for m in masks]
        samples = []
        for i in range(self.try_num):
            # kernel = self.new_gp.kernel.copy()
            model = GPR_sklearn(self.space,
                                rng=self.rng,
                                kernel=self.target_model.kernel,
                                do_optimize=False)
            model.train(x_cv[i], y_cv[i][:, None])
            mean, cov = model.predict(x, "full_cov")  # TODO 使用kernel子块
            # model = GaussianProcessRegressor(self.dim)
            # model.fit(x_cv[i], y_cv[i][:, None])
            # mean, cov = model.predict_with_cov(x)
            samples.append(
                self.rng.multivariate_normal(np.squeeze(mean), cov,
                                             self.rank_sample_num))
        return np.stack(samples, axis=1)

    # def predict_with_sigma(self, newX):
    #     models = [self.gps[d].cached_predict_with_sigma(newX) for d in range(self.old_D_num)]
    #     models.append(self.new_gp.predict_with_sigma(newX))
    #     models = np.asarray(models)
    #     mu = self.rank_weight.dot(models[:, 0])
    #     sigma = (self.rank_weight ** 2).dot(models[:, 1] ** 2)

    #     return mu, np.sqrt(sigma)

    def __delete_noise_knowledge(self, ranking_losses):
        if self.is_purn:
            # p_drop = 1 - (1 - self.x.shape[0] / 30) * (ranking_losses[:-1, :] < ranking_losses[-1, :]).sum(axis=-1) / (
            # self.rank_sample_num * (1 + self.alpha))
            # mask_to_remove = self.rng.binomial(1, p_drop) == 1

            target_loss = ranking_losses[-1]
            threshold = np.percentile(target_loss, 95)
            mask_to_remove = np.median(
                ranking_losses,
                axis=1) > threshold  # 中位数样本loss 比 95% 的target model结果差
            idx_to_remove = np.argwhere(mask_to_remove).ravel()
            ranking_losses = np.delete(ranking_losses, idx_to_remove, axis=0)
            # ranking_losses[idx_to_remove] = self.rank_sample_num
            for idx in reversed(idx_to_remove):  # TODO Remove permanent
                self.base_models.pop(idx)
            # self.old_D_num -= len(idx_to_remove)
        return ranking_losses
    
    def _delete_noise_knowledge(self, ranking_losses):
        if self.is_purn:
            p_drop = 1 - (1 - self.try_num / self.budget) * (ranking_losses[:-1, :] < ranking_losses[-1, :]).sum(axis=-1) / (
            self.rank_sample_num * (1 + self.alpha))
            mask_to_remove = self.rng.binomial(1, p_drop) == 1
            idx_to_remove = np.argwhere(mask_to_remove).ravel()
            # ranking_losses = np.delete(ranking_losses, idx_to_remove, axis=0)
            ranking_losses[idx_to_remove] = self.rank_sample_num
            # ranking_losses[idx_to_remove] = self.rank_sample_num
            # for idx in reversed(idx_to_remove):  # TODO Remove permanent
            #     self.base_models.pop(idx)
            # self.old_D_num -= len(idx_to_remove)
        return ranking_losses

class KernelRegress(ABCWeightStategy):
    def __init__(self, cs, base_models, target_model, rng, bandwidth=0.1,**kwargs):
        super().__init__(cs, base_models, target_model, rng, **kwargs)

        self.bandwidth = bandwidth
        # self.is_purn = is_purn

    def get_weight(self, trials: Trials):
        base_model_means = []
        for model in self.base_models:
            base_model_means.append(
                model._predict_normalize(trials.get_sparse_array(), None)[0])
        if not base_model_means:
            return []
        base_model_means = np.stack(base_model_means)  # [model, obs_num, 1]
        weight = self._naiveVersion(
            base_model_means, np.asarray(trials._his_observe_value))

        return weight  #/ weight.sum()

    def _kendallTauCorrelation(self, base_model_means, y):
        if y is None or len(y) < 2:
            return np.full(base_model_means.shape[0], 1)
        rank_loss = (
            (base_model_means[..., None] < base_model_means[..., None, :]) ^
            (y[..., None] < y[..., None, :])).astype('float')
        base_num, obs_num, _ = rank_loss.shape
        # rank_loss = rank_loss
        # rank_loss = (rank_loss < 1) * (1 - rank_loss**2) * 3 / 4
        num = obs_num * (obs_num - 1) // 2
        l = np.empty((base_num, num))
        idxs = np.triu_indices(obs_num, 1)
        for n in range(base_num):
            l[n] = rank_loss[n][idxs]
        l = np.linalg.norm(
            l / num/2,
            axis=1) / self.bandwidth  # meta feature : |source-target| / b
        l = (l < 1) * (1 - l * l) * 3 / 4
        
        return np.append(l, 3/4)
        # t = rank_loss.mean(axis=(-1, -2)) / self.bandwidth
        # return (t < 1) * (1 - t * t) * 3 / 4
        # return self.rho * (1 - t * t) if t < 1 else 0
    def _naiveVersion(self, base_model_means, y):
        meta_features = []
        for model_predit in base_model_means:
            meta_feature = []
            for i in range(len(y)):
                for j in range(i):
                    meta_feature.append((model_predit[i] > model_predit[j]) /
                                        (len(y) * (len(y) - 1)))
            meta_features.append(meta_feature)
        meta_feature = []
        for i in range(len(y)):
            for j in range(i):
                meta_feature.append((y[i] > y[j]) /
                                    (len(y) * (len(y) - 1)))
        meta_features.append(meta_feature)
        meta_features = np.asarray(meta_features)
        
        def kern(a, b, rho):
            def gamma(x):
                gamma = 3 / 4 * (1 - x ** 2) if x <= 1 else 0.0
                return gamma

            kern = gamma(np.linalg.norm(a - b) / rho)
            return kern
        weights = []
        for meta_feature in meta_features:
            weights.append(kern(meta_feature, meta_features[-1], self.bandwidth))
        return np.asarray(weights)
        
# class ProductExpert():
#     def __init__(self, cs, base_models, target_model, rng):
#         self.space = cs
#         self.base_models = base_models
#         self.target_model = target_model
#         # self.rank_sample_num = rank_sample_num
#         # self.iter = 0
#         self.rng = rng
#         # self.is_purn = is_purn

#     def get_weight(self, trials: Trials):
#         base_model_vars = []
#         for model in self.base_models:
#             base_model_vars.append(
#                 model.predict(trials.get_sparse_array())[1])
#         if not base_model_vars:
#             return []
#         base_model_vars = np.concatenate(base_model_vars)  # [model, obs_num, 1]
#         weight = self._naiveVersion(
#             base_model_vars, self.target_model.target_model_predict(trials.get_sparse_array())[1])

#         return weight  #/ weight.sum()


#     def _naiveVersion(self, base_model_vars, target_model_var):
#         beta = 1. / (len(base_model_vars)+1)
#         weights = [beta / var for var in base_model_vars]
#         weights.append(beta / target_model_var)
#         return np.array(weights)
        
class ZeroWeight(ABCWeightStategy):
    def __init__(self, cs, base_models, target_model, rng, **kwargs):
        super().__init__(cs, base_models, target_model, rng, **kwargs)

    def get_weight(self, trials: Trials):
        weight = np.zeros(len(self.base_models)+1)
        weight[-1] = 1

        return weight  #/ weight.sum()

        
