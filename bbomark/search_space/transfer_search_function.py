import glob

import pandas as pd

from bbomark.configspace import build_space
from bbomark.core import TestFunction

import numpy as np

class Model(TestFunction):
    _SUPPORT_FUNCTIONS = ('svm')

    def __init__(self, cfg, **kwargs):
        # np.random.seed(cfg.GENERAL.random_seed)
        self.cfg = cfg
        # self.dim = 30
        # assert self.dim % 2 == 0
        super().__init__()

        assert cfg.TEST_PROBLEM.kwargs.func_name in self._SUPPORT_FUNCTIONS
        self.data_path = kwargs.get('data_path')+ kwargs.get('func_name')
        self.test_data_name = kwargs.get('test_data')
        # func_name = cfg.TEST_PROBLEM.kwargs.func_name
        func_name = kwargs.get('func_name')
        if func_name == 'svm':
            self.func = SVM(self.data_path, self.test_data_name)
        else:
            assert False
        self.noise_std = kwargs.get('noise_std', 0)
        self.api_config = self._load_api_config()

    def evaluate(self, params: dict):

        # input_x = []

        f = self.func(params)
        random_noise = np.random.randn() * self.noise_std + 1.
        res_out = {
            # 'rank': (np.searchsorted(self.func.sorted_new_D_y, -f)+1)/len(self.func.sorted_new_D_y),
            'rank': (np.searchsorted(self.func.sorted_new_D_y, -f)+1),
            'regret': (self.func.best_acc-f ) / self.func.acc_range,
        }
        res_loss = {
            'test': f,
            'val': f * random_noise,
        }
        return ([res_out[k] for k in self.cfg.TEST_PROBLEM.func_evals],
                [res_loss[k] for k in self.cfg.TEST_PROBLEM.losses])

    def _load_api_config(self):
        return self.func._load_api_config()



class SVM():
    def __init__(self, data_path, test_data_name=''):
        self.hp_num = 3
        self.hp_indicator_num = [3]
        self.data_path = data_path
        if test_data_name:
            self.test_data_name = test_data_name
        else:
            raise NotImplemented

        self._prepare()
        self.best_acc = max(self.new_D_y)
        self.acc_range = self.best_acc - min(self.new_D_y)
        self.sorted_new_D_y = np.sort(-self.new_D_y, )
        self.api_config = self._load_api_config()

    def __call__(self, hp_param):
        key = tuple(np.round(hp_param[k], 5) for k in self.api_config)
        ret = self.cached_new_res[key]
        # print(key)
        rank = 1
        # if True:
        #     for y in self.new_D_y:
        #         if y > ret:
        #             rank += 1
        # print('rank: ', rank)
        return ret

    def cache(self, new_D_x_param):
        self.cached_new_res = {
            # tuple(sorted(inst_param.items())): self.new_D_y[i] for i, inst_param in enumerate(new_D_x_param)
        }
        for i, inst_param in enumerate(new_D_x_param):
            key = tuple(np.round(inst_param[k], 5) for k in self.api_config)

            self.cached_new_res[key] = self.new_D_y[i]



    def _prepare(self):
        (datasets_hp, datasets_label), filenames = self._load_meta_data()
        new_D_idx = filenames.index(self.test_data_name)
        self.new_D_x, self.new_D_y = datasets_hp.pop(new_D_idx), datasets_label.pop(new_D_idx)
        self.old_D_x, self.old_D_y = datasets_hp, datasets_label
        # scale -acc
        for d, inst_y in enumerate(self.old_D_y):
            # inst_y = - inst_y_ # minimize problem
            _min = np.min(inst_y)
            _max = np.max(inst_y)
            self.old_D_y[d] = (inst_y - _min) / (_max - _min)

    def _inst_to_config(self, inst):
        hp_params = {}
        for i, hp in enumerate(self.api_config):
            if self.api_config[hp]['type'] == 'cat':
                hp_params[hp] = self.api_config[hp]['values'][int(inst[i])]
            else:
                hp_params[hp] = inst[i]
        return hp_params

    def array_to_config(self):
        self.old_D_x_params = []
        for d in range(len(self.old_D_x)):

            self.old_D_x_params.append([self._inst_to_config(inst) for inst in self.old_D_x[d]])

        self.new_D_x_param = [self._inst_to_config(inst) for inst in self.new_D_x]

        return self.old_D_x_params, self.old_D_y, self.new_D_x_param



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
                    line_array_raw = list(map(float, line.strip().split(' ')))
                    idx_start = 1
                    line_array = [line_array_raw[0]]
                    # for ind_num in self.hp_indicator_num:
                    #     line_array.append(line_array_raw[idx_start:idx_start+ind_num].index(1))
                    #     idx_start += ind_num

                    # line_array.extend(line_array_raw[idx_start:self.hp_num+1])
                    line_array.extend(line_array_raw[idx_start:self.hp_num+4])
                    insts.append(line_array)

            datasets = np.asarray(insts, dtype=np.float)
            datasets_hp.append(datasets[:, 1:])
            datasets_label.append(datasets[:, 0])
            mask = datasets_hp[-1][:, 0].astype(np.bool_)  # TODO
            datasets_hp[-1] = datasets_hp[-1][mask, 3:]
            datasets_label[-1] = datasets_label[-1][mask]
        return (datasets_hp, datasets_label), filenames

    def _load_api_config(self):
        return {
            # 'kernel': {
            #     'type': 'cat',
            #     # 'values': ['linear', 'Polynomial', 'RBF']
            #     'values': [0, 1, 2]
            # },
            'C': {
                'type': 'float',
                'range':[-1, 1]
            },
            'gamma':{
                'type': 'float',
                'range':[-1, 1]
            },
            'd':{
                'type': 'float',
                'range':[0, 1]
            }
        }