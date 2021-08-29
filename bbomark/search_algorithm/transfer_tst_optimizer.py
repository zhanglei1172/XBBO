from _curses import meta
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve_triangular, cholesky
from scipy import stats
import tqdm, random
# import hashlib
from ConfigSpace.hyperparameters import (UniformIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         CategoricalHyperparameter,
                                         OrdinalHyperparameter)

from bbomark.configspace.feature_space import FeatureSpace_uniform
from bbomark.core import AbstractOptimizer
from bbomark.configspace.space import Configurations
from bbomark.core.stochastic import Category, Uniform
from bbomark.core.trials import Trials


class EI():
    def __init__(self):
        self.xi = 0.1

    def _getEI(self, mu, sigma, y_best): #
        z = (-y_best + mu - self.xi) / sigma
        ei = (-y_best + mu -
              self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei

    # def argmax(self, y_best, surrogate, candidates):
    #     best_ei = -1
    #     best_candidate = []
    #     for candidate in candidates:
    #         y_hat = surrogate.predict(candidate)
    #         ei = self._getEI(y_hat[0], y_hat[1], y_best)
    #         if ei > best_ei:
    #             best_ei = ei
    #             best_candidate = [candidate]
    #         elif ei == best_ei:
    #             best_candidate.append(candidate)
    #     return np.random.choice(best_candidate)

    def argmax(self, y_best, surrogate, candidates):
        best_ei = -1
        best_candidate = []
        candidates_rm_id = []
        for i, candidate in enumerate(candidates):
            y_hat = surrogate.predict(candidate)
            ei = self._getEI(y_hat[0], y_hat[1], y_best)
            if ei > best_ei:
                best_ei = ei
                best_candidate = [candidate]
                candidates_rm_id = [i]
            elif ei == best_ei:
                best_candidate.append(candidate)
                candidates_rm_id.append(i)

        assert best_candidate
        idx = np.random.choice(len(best_candidate))
        return (best_candidate)[idx], candidates_rm_id[idx]


class SEkernel():
    def __init__(self):
        self.initialize()

    def initialize(self):
        # self.sumF = 0.001
        # self.sumL = 0.001
        # self.sumY = 0.001
        self.sigma_f = 1
        self.sigma_l = 1
        self.sigma_y = 0.001



    def compute_kernel(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            # noise = np.diag([self.sigma_y**2 for _ in range(x1.shape[0])])
            noise = np.eye(x1.shape[0]) * self.sigma_y**2
        else:
            noise = 0
        x2 = np.atleast_2d(x2)
        x1 = np.atleast_2d(x1)
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * (x1 @ x2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.sigma_l ** 2 * dist_matrix) + noise


class Surrogate():
    def __init__(self, dim):
        self.dim = dim

    def predict(self, newX):
        pass

    def fit(self, x, y):
        pass


class GaussianProcessRegressor(Surrogate):
    def __init__(self, dim):
        super().__init__(dim)
        self.kernel = SEkernel()
        self.cached = {}

    def fit(self, x, y):
        self.X = x
        kernel = self.kernel.compute_kernel(x)
        self.L = cholesky(kernel, lower=True)
        _part = solve_triangular(self.L, y, lower=True)
        self.KinvY = solve_triangular(self.L.T, _part, lower=False)

    def predict(self, newX):
        # Kstar = np.squeeze(self.kernel.compute_kernel(self.X, newX))
        Kstar = (self.kernel.compute_kernel(self.X, newX))
        return (Kstar.T @ self.KinvY).item()

    def cached_predict(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        if not hasattr(self, 'X'):
            return 0, np.inf
        else:
            Kstar = self.kernel.compute_kernel(self.X, newX)
            _LinvKstar = solve_triangular(self.L, Kstar, lower=True)
            return (Kstar.T @ self.KinvY).item(), np.sqrt(self.kernel.compute_kernel(newX, newX) - _LinvKstar.T @ _LinvKstar)

class TST_surrogate(Surrogate):
    rho = 0.75
    def __init__(self, dim, bandwidth=0.1):
        super().__init__(dim)

        self.new_gp = GaussianProcessRegressor(dim)
        # self.candidates = None
        self.bandwidth = bandwidth
        # self.history_x = []
        # self.history_y = []

    def get_knowledge(self, old_D_x, old_D_y, new_D_x=None):
        self.old_D_num = len(old_D_x)
        self.gps = []
        for d in range(self.old_D_num):
            self.gps.append(GaussianProcessRegressor(self.dim))
            self.gps[d].fit(old_D_x[d], old_D_y[d])
        if new_D_x is not None:
            candidates = new_D_x
        else: #
            raise NotImplemented
        self.similarity = [self.rho for _ in range(self.old_D_num)]
        return candidates

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self.new_gp.fit(x, y)
        self.similarity = [self.kendallTauCorrelation(d, x, y) for d in range(self.old_D_num)]

    def predict(self, newX):
        denominator = self.rho
        mu, sigma = self.new_gp.predict_with_sigma(newX)
        for d in range(self.old_D_num):
            mu += self.similarity[d] * self.gps[d].cached_predict(newX)
            denominator += self.similarity[d]
        mu /= denominator
        if sigma == np.inf:
            sigma = 1000
        return mu, sigma

    
    def kendallTauCorrelation(self, d, x, y):
        '''
        计算第d个datasets与new datasets的 相关性
        (x, y) 为newdatasets上的history结果
        '''
        if y is None or len(y) < 2:
            return self.rho
        disordered_pairs = total_pairs = 0
        for i in range(len(y)):
            for j in range(len(y)):
                if (y[i] < y[j] != self.gps[d].cached_predict(
                        x[i]) < self.gps[d].cached_predict(x[j])):
                    disordered_pairs += 1
                total_pairs += 1
        t = disordered_pairs / total_pairs / self.bandwidth
        return self.rho * (1 - t * t) if t < 1 else 0

class SMBO_test():

    def __init__(self,dim=6,
                 min_sample=0,
                 data_path='/home/zhang/PycharmProjects/MAC/TST/data/svm',
                 test_data_name='A9A',
                 # avg_best_idx=2.0,
                 # meta_data_path=None,
                 ):
        self.min_sample = min_sample
        self.data_path = data_path
        self.test_data_name = test_data_name
        # self.dim = dim
        self.hp_num = dim
        self.trials = Trials()
        self.surrogate = TST_surrogate(self.hp_num)
        self.acq_func = EI()
        self._load_meta_data()
        self._prepare()

    def cache_compute(self):
        self.cached_new_res = {
            # tuple(sorted(inst_param.items())): self.new_D_y[i] for i, inst_param in enumerate(new_D_x_param)
        }

        for i, inst in enumerate(self.new_D_x):

            key = hash(inst.data.tobytes())
            self.cached_new_res[key] = self.new_D_y[i]


    # def get_knowledge(self):
    #     pass

    def _prepare(self):
        (datasets_hp, datasets_label), filenames = self._load_meta_data()
        new_D_idx = filenames.index(self.test_data_name)
        self.new_D_x, self.new_D_y = datasets_hp.pop(new_D_idx), datasets_label.pop(new_D_idx)
        self.old_D_x, self.old_D_y = datasets_hp, datasets_label
        # sclae -acc
        for d, inst_y in enumerate(self.old_D_y):
            # inst_y = - inst_y_ # minimize problem
            _min = np.min(inst_y)
            _max = np.max(inst_y)
            self.old_D_y[d] = (inst_y - _min) / (_max - _min)

        self.candidates = self.new_D_x



    def _load_meta_data(self):
        file_lists = glob.glob(self.data_path + "/*")
        datasets_hp = []
        datasets_label = []
        filenames = []
        for file in file_lists:
            # data = []
            filename = file.rsplit('/', maxsplit=1)[-1]
            filenames.append(filename)
            with open(file, 'r') as f:
                insts = [] # 2dim
                for line in f.readlines(): # convet categories
                    line_array = list(map(float, line.strip().split(' ')))
                    insts.append(line_array[:1+self.hp_num])

            datasets = np.asarray(insts, dtype=np.float)
            datasets_hp.append(datasets[:, 1:])
            datasets_label.append(datasets[:, 0])
        return (datasets_hp, datasets_label), filenames


    def suggest(self, n_suggestions=1):
        # 只suggest 一个
        if (self.trials.trials_num) < self.min_sample :
            raise NotImplemented
            # return self._random_suggest()
        else:
            sas = []
            for n in range(n_suggestions):
                suggest_array, rm_id = self.acq_func.argmax(max(self.trials.history_y+[-1]), self.surrogate, self.candidates)
                self.candidates = np.delete(self.candidates, rm_id, axis=0)
                sas.append(suggest_array)
        return sas



    def observe(self, x, y):
        self.trials.history.append(x)
        self.trials.history_y.append(y)
        if self.trials.best_y is None or self.trials.best_y < y:
            self.trials.best_y = y
        self.trials.trials_num += 1
        self.surrogate.fit(self.trials.history, self.trials.history_y)

    def print_rank(self):
        rank = 1
        if True:
            for y in self.new_D_y:
                if y > self.trials.history_y[-1]:
                    rank += 1
        print('rank: ', rank)
        return rank

    def print_best_rank(self):
        rank = 1
        if True:
            for y in self.new_D_y:
                if y > self.trials.best_y:
                    rank += 1
        print('rank_best: ', rank)
        return rank


class SMBO(AbstractOptimizer, FeatureSpace_uniform):

    def __init__(self,
                 config_spaces,
                 min_sample=0,
                 # avg_best_idx=2.0,
                 # meta_data_path=None,
                 ):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_uniform.__init__(self, self.space.dtypes_idx_map)
        self.min_sample = min_sample
        # self.avg_best_idx = avg_best_idx
        # self.meta_data_path = meta_data_path
        configs = self.space.get_hyperparameters()
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)

        self.hp_num = len(configs)
        self.trials = Trials()
        self.surrogate = TST_surrogate(self.dense_dimension)
        self.acq_func = EI()

    def prepare(self, old_D_x_params, old_D_y, new_D_x_param):
        old_D_x = []
        for insts_param in old_D_x_params:
            insts_feature = []
            for inst_param in insts_param:
                array = Configurations.dictUnwarped_to_array(self.space, inst_param)
                insts_feature.append(self.array_to_feature(array, self.dense_dimension))
            old_D_x.append(np.asarray(insts_feature))
        insts_feature = []
        for inst_param in new_D_x_param:
            array = Configurations.dictUnwarped_to_array(self.space, inst_param)
            insts_feature.append(self.array_to_feature(array, self.dense_dimension))
        new_D_x = (np.asarray(insts_feature))

        self.candidates = self.surrogate.get_knowledge(old_D_x, old_D_y, new_D_x)


    def suggest(self, n_suggestions=1):
        # 只suggest 一个
        if (self.trials.trials_num) < self.min_sample :
            raise NotImplemented
            # return self._random_suggest()
        else:
            x_unwarpeds = []
            sas = []
            for n in range(n_suggestions):
                suggest_array, rm_id = self.acq_func.argmax(max(self.trials.history_y+[-1]), self.surrogate, self.candidates)
                self.candidates = np.delete(self.candidates, rm_id, axis=0)
                x_array = self.feature_to_array(suggest_array, self.sparse_dimension)
                x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)


                sas.append(suggest_array)
                x_unwarpeds.append(x_unwarped)
        # x = [Configurations.array_to_dictUnwarped(self.space,
        #                                           np.asarray(sa)) for sa in sas]
        self.trials.params_history.extend(x_unwarpeds)
        return x_unwarpeds, sas



    # def _random_suggest(self, n_suggestions=1):
    #     sas = []
    #     for n in range(n_suggestions):
    #         suggest_array = [node.random_sample() for node in self.nodes]
    #         for i in range(self.hp_num):
    #             self.node_trial_num[i] += 1
    #         sas.append(suggest_array)
    #     x = [Configurations.array_to_dictUnwarped(self.space,
    #                                               np.array(sa)) for sa in sas]
    #     self.trials.params_history.extend(x)
    #     return x, sas


    def observe(self, x, y):
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1
        self.surrogate.fit(self.trials.history, self.trials.history_y)



opt_class = SMBO

if __name__ == '__main__':
    try_num = 30
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)
    smbo = SMBO_test()
    smbo.candidates = smbo.surrogate.get_knowledge(smbo.old_D_x, smbo.old_D_y, smbo.new_D_x)
    smbo.cache_compute()
    rank = []
    best_rank = []
    for t in range(try_num):
        print('-'*10)
        print('iter {}: '.format(t+1))
        x = smbo.suggest()[0]


        key = hash(x.data.tobytes())
        y = smbo.cached_new_res[key]

        smbo.observe(x, y)

        print(y)
        rank.append(smbo.print_rank())
        best_rank.append(smbo.print_best_rank())
    plt.subplot(211)
    plt.plot(np.array(smbo.trials.history_y), label='ACC')
    plt.plot(np.maximum.accumulate(smbo.trials.history_y), label='ACC_best')
    plt.legend()
    plt.ylabel('ACC')
    plt.subplot(212)
    plt.plot(rank, label='rank')
    plt.plot(best_rank, label='best_rank')

    plt.legend()
    plt.ylabel('Rank')
    plt.xlabel('iter')
    plt.savefig('./out/TST-R.png')
    plt.show()

