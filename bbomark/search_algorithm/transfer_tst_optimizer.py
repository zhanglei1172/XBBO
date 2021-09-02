import glob
import numpy as np
from matplotlib import pyplot as plt

import tqdm, random

from bbomark.acquisition_function.ei import EI
from bbomark.configspace.feature_space import FeatureSpace_uniform
from bbomark.core import AbstractOptimizer
from bbomark.configspace.space import Configurations

from bbomark.core.trials import Trials
from bbomark.surrogate.tst import TST_surrogate


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


    def suggest(self, n_suggestions=1, enable_random=False):
        # 只suggest 一个
        if (self.trials.trials_num) < self.min_sample and enable_random :
            # raise NotImplemented
            return self._random_suggest()
        else:
            sas = []
            for n in range(n_suggestions):
                suggest_array, rm_id = self.acq_func.argmax(max(self.trials.history_y+[-1]), self.surrogate, self.candidates)
                self.candidates = np.delete(self.candidates, rm_id, axis=0)
                sas.append(suggest_array)
        return sas

    def _random_suggest(self, n_suggestions=1):
        sas = []
        for n in range(n_suggestions):
            rm_id = np.random.choice(len(self.candidates))
            sas.append(self.candidates[rm_id])
            self.candidates = np.delete(self.candidates, rm_id, axis=0)
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
        self.surrogate = TST_surrogate(self.hp_num)
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


    def observe(self, x, y):
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1
        self.surrogate.fit(self.trials.history, self.trials.history_y)


def test_tst_r(try_num, SEED=0):
    np.random.seed(SEED)
    random.seed(SEED)
    smbo = SMBO_test()
    smbo.candidates = smbo.surrogate.get_knowledge(smbo.old_D_x, smbo.old_D_y, smbo.new_D_x)
    smbo.cache_compute()
    rank = []
    best_rank = []
    for t in range(try_num):
        print('-' * 10)
        print('iter {}: '.format(t + 1))
        x = smbo.suggest()[0]

        key = hash(x.data.tobytes())
        y = smbo.cached_new_res[key]

        smbo.observe(x, y)

        print(y)
        rank.append(smbo.print_rank())
        best_rank.append(smbo.print_best_rank())
    return smbo.trials.history_y, np.maximum.accumulate(smbo.trials.history_y), rank, best_rank

def test_gpbo(try_num, SEED=0):
    np.random.seed(SEED)
    random.seed(SEED)
    smbo = SMBO_test()
    smbo.candidates = smbo.surrogate.get_knowledge(smbo.old_D_x, smbo.old_D_y, smbo.new_D_x)
    smbo.cache_compute()
    rank = []
    best_rank = []
    for t in range(try_num):
        print('-' * 10)
        print('iter {}: '.format(t + 1))
        x = smbo.suggest(enable_random=True)[0]

        key = hash(x.data.tobytes())
        y = smbo.cached_new_res[key]

        smbo.observe(x, y)

        smbo.surrogate.similarity = [0 for _ in smbo.old_D_x]
        print(y)
        rank.append(smbo.print_rank())
        best_rank.append(smbo.print_best_rank())
    return smbo.trials.history_y, np.maximum.accumulate(smbo.trials.history_y), rank, best_rank


opt_class = SMBO

if __name__ == '__main__':
    try_num = 30
    SEED = 0
    acc,acc_best, rank, rank_best = test_tst_r(try_num, SEED)
    acc_,acc_best_, rank_, rank_best_ = test_gpbo(try_num, SEED)
    plt.subplot(211)
    plt.plot(acc, 'b-', label='TST-R')
    plt.plot(acc_best, 'b:', label='TST-R_best')

    plt.plot(acc_, 'r-', label='GP-BO')
    plt.plot(acc_best_, 'r:', label='GP-BO_best')
    plt.legend()
    plt.ylabel('ACC')
    # plt.title

    plt.subplot(212)
    plt.plot(rank, 'b-', label='TST-R')
    plt.plot(rank_best, 'b:', label='TST-R_best')

    plt.plot(rank_, 'r-', label='GP-BO')
    plt.plot(rank_best_, 'r:', label='GP-BO_best')

    plt.legend()
    plt.ylabel('Rank')
    plt.xlabel('iter')

    # plt.suptitle('TST-R in A9A datasets(svm)')
    # plt.savefig('./out/TST-R.png')

    # plt.suptitle('TST-R in A9A datasets(svm)-correct')
    # plt.savefig('./out/TST-R-correct.png')
    plt.show()

