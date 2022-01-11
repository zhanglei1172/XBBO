import os
from typing import Tuple, List, Callable
import numpy as np
import pandas as pd
from pathlib import Path

from xbbo.core import TestFunction

deepar = 'DeepAR'
fcnet = 'FCNET'
xgboost = 'XGBoost'
nas102 = 'nas_bench102'

svm = 'svm'

metric_error = 'metric_error'
metric_time = 'metric_time'

blackbox_tasks = {
    nas102: [
        'cifar10',
        'cifar100',
        'ImageNet16-120'
    ],
    fcnet: [
        'naval',
        'parkinsons',
        'protein',
        'slice',
    ],
    deepar: [
        'm4-Hourly',
        'm4-Daily',
        'm4-Weekly',
        'm4-Monthly',
        'm4-Quarterly',
        'm4-Yearly',
        'electricity',
        'exchange-rate',
        'solar',
        'traffic',
    ],
    xgboost: [
        'a6a',
        'australian',
        'german.numer',
        'heart',
        'ijcnn1',
        'madelon',
        'skin_nonskin',
        'spambase',
        'svmguide1',
        'w6a'
    ],
}

error_metric = {
    deepar: 'metric_CRPS',
    fcnet: 'metric_error',
    nas102: 'metric_error',
    xgboost: 'metric_error',
}

tasks = [task for bb, tasks in blackbox_tasks.items() for task in tasks]

CACHE_DATA = {}


def load_surrogate_benchmark_data(data_path, test_task_name, min_max_features=False):
    res = CACHE_DATA.get(test_task_name, False)
    if res:
        return res
    df = pd.read_csv(data_path)
    df = df.fillna(0)
    # test_mask = (df['data_id'] == test_task_name).values
    pre = len(df.columns)
    dummy_df = pd.get_dummies(df)
    post = len(dummy_df.columns)
    hp_num = list(dummy_df.columns).index('auc') - 1
    if pre == post:
        data_value = dummy_df.iloc[:, 1:hp_num + 1].values
        # indicator_num = 0
    else:
        indicator_num = post - pre + 1
        data_value = dummy_df.iloc[:, list(range(1, hp_num + 1)) + list(range(post - indicator_num, post))].values
    if min_max_features:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler().fit(data_value)
        data_value = scaler.transform(data_value)
    data_label = -df['accuracy'].values[:, None]
    datasets_hp = []
    datasets_label = []
    for i, data_id in enumerate(df['data_id'].unique()):
        if data_id == test_task_name:
            test_id = i
        datasets_hp.append(data_value[(df['data_id'] == data_id).values])
        datasets_label.append(data_label[(df['data_id'] == data_id).values])
    test_task, test_task_label = datasets_hp.pop(test_id), datasets_label.pop(test_id)

    CACHE_DATA[test_task_name] = (datasets_hp, datasets_label, test_task, test_task_label, [f'hp_{i}' for i in
                                                                                            range(test_task.shape[1])])
    return CACHE_DATA[test_task_name]


def load_svm_data(data_path, test_task_name, hp_num=3, min_max_features=False, sparse=False):
    res = CACHE_DATA.get(test_task_name, False)
    if res:
        return res
    file_lists = os.listdir(data_path)
    file_lists = list(map(lambda x: data_path + x, file_lists))
    datasets_hp = []
    datasets_label = []
    filenames = []
    for file in file_lists:
        # data = []
        filename = file.rsplit('/', maxsplit=1)[-1]
        filenames.append(filename)
        with open(file, 'r') as f:
            insts = []  # 2dim
            for line in f.readlines():  # convet categories
                line_array_raw = list(map(float, line.strip().split(' ')))
                idx_start = 1
                line_array = [line_array_raw[0]]
                # for ind_num in self.hp_indicator_num:
                #     line_array.append(line_array_raw[idx_start:idx_start+ind_num].index(1))
                #     idx_start += ind_num

                # line_array.extend(line_array_raw[idx_start:self.hp_num+1])
                line_array.extend(line_array_raw[idx_start:hp_num + 1+3])
                insts.append(line_array)

        datasets = np.asarray(insts, dtype=np.float)
        if sparse:
            mask = datasets[:, 1] == 1
            datasets_hp.append(datasets[mask, 1+3:])
            # datasets_hp[-1] = datasets_hp[-1][mask]
            datasets_label.append(-datasets[mask, 0:1])  # TODO convet to minimize problem (regret)
        else:
            datasets_hp.append(datasets[:, 1:])
            datasets_label.append(-datasets[:, 0:1]) # TODO convet to minimize problem (regret)
        mask = datasets_hp[-1][:, 0].astype(np.bool_)  # TODO
        datasets_hp[-1] = datasets_hp[-1][mask, 3:]
        datasets_label[-1] = datasets_label[-1][mask]
        # if True:
        #     datasets_label[-1] = datasets_label[-1]
    if min_max_features:
        # min-max scaling of input features
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler().fit(np.vstack(datasets_hp))
        datasets_hp = [scaler.transform(X) for X in datasets_hp]
    test_idx = filenames.index(test_task_name)
    test_task = datasets_hp.pop(test_idx)
    test_task_label = datasets_label.pop(test_idx)
    hp_config = {'C': {
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
            }}
    CACHE_DATA[test_task_name] = (datasets_hp, datasets_label, test_task, test_task_label, hp_config)
    # CACHE_DATA[test_task_name] = (datasets_hp, datasets_label, test_task, test_task_label, [f'hp_{i}' for i in
    #                                                                                         range(test_task.shape[1])])
    return CACHE_DATA[test_task_name]


def evaluations_np(
        blackbox: str,
        test_task: str,
        metric_cols: List[str],
        min_max_features: bool = False
):
    """
    :param blackbox:
    :param test_task:
    :param metric_cols:
    :param min_max_features: whether to apply min-max scaling on input features
    :return: list of features/evaluations on train task and features/evaluations of the test task.
    """

    assert blackbox in [deepar, fcnet, xgboost, nas102]
    res = CACHE_DATA.get(test_task, False)
    if res:
        return res
    df = pd.read_csv(Path(__file__).parent.parent.parent / f"offline_evaluations/{blackbox}.csv.zip")

    assert test_task in df.task.unique()
    for c in metric_cols:
        assert c in df.columns

    Xy_dict = {}
    for task in sorted(df.task.unique()):
        mask = df.loc[:, 'task'] == task
        hp_cols = [c for c in sorted(df.columns) if c.startswith("hp_")]
        X = df.loc[mask, hp_cols].values
        y = df.loc[mask, metric_cols].values
        Xy_dict[task] = X, y

    # todo it would be better done as a post-processing step
    if blackbox in [fcnet, nas102]:
        # applies onehot encoding to *all* hp columns as all hps are categories for those two blackboxes
        # it would be nice to detect column types or pass it as an argument
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        hp_cols = [c for c in sorted(df.columns) if c.startswith("hp_")]
        enc.fit(df.loc[:, hp_cols])
        for task, (X, y) in Xy_dict.items():
            X_features = enc.transform(X)
            Xy_dict[task] = X_features, y

    if min_max_features:
        # min-max scaling of input features
        from sklearn.preprocessing import MinMaxScaler
        X = np.vstack([X for (X, y) in Xy_dict.values()])
        scaler = MinMaxScaler().fit(X)
        Xy_dict = {t: (scaler.transform(X), y) for (t, (X, y)) in Xy_dict.items()}

    Xys_train = [Xy_dict[t] for t in df.task.unique() if t != test_task]
    Xy_test = Xy_dict[test_task]
    hp_names = [f'hp_{i}' for i in range(Xy_test[0].shape[1])]
    CACHE_DATA[test_task] = Xys_train, Xy_test, hp_names
    return CACHE_DATA[test_task]


class Model(TestFunction):

    def __init__(self, cfg, seed, **kwargs):
        self.cfg = cfg
        # self.dim = 30
        # assert self.dim % 2 == 0
        super().__init__(seed=seed)
        name = kwargs.get('func_name', None)
        if name is None:
            test_task = cfg.TEST_PROBLEM.kwargs.test_task if cfg else kwargs['test_task']
            # assert test_task in tasks
            # self.data_path = kwargs.get('data_path')+ kwargs.get('func_name')
            for bb, tasks in blackbox_tasks.items():
                if test_task in tasks:
                    self.blackbox = bb
                    break
            Xys_train, Xy_test, self.hp_names = evaluations_np(
                blackbox=self.blackbox,
                test_task=test_task,
                metric_cols=[error_metric[self.blackbox]],
                min_max_features=True
            )
            self.old_D_x, self.old_D_y = list(zip(*Xys_train))
            self.new_D_x, self.new_D_y = Xy_test
            self.api_config = self._load_api_config()
        elif name == 'svm':
            self.old_D_x, self.old_D_y, self.new_D_x, self.new_D_y, self.hp_config = load_svm_data(
                kwargs.get('data_path'), kwargs.get('test_task'))
            self.api_config = self.hp_config
        elif name == 'surrogate':

            self.old_D_x, self.old_D_y, self.new_D_x, self.new_D_y, self.hp_names = load_surrogate_benchmark_data(
                kwargs.get('data_path'), kwargs.get('test_task'))
            self.api_config = self._load_api_config()
        else:
            raise NotImplemented
        if kwargs.get('normalize_old', False) == True:
            self.old_D_y = [(y - y.min())/(y.max()-y.min()) for y in self.old_D_y]
        self.noise_std = kwargs.get('noise_std', 0)


        self.bbfunc = BlackboxOffline(self.new_D_x, self.new_D_y)
        self.best_err = min(self.new_D_y).item()
        self.err_range = max(self.new_D_y).item() - self.best_err
        self.sorted_new_D_y = np.sort(self.new_D_y).ravel()

    def evaluate(self, params: dict):

        # input_x = []

        f = self.bbfunc(np.asarray([params[k] for k in self.api_config.keys()])).item()
        if self.noise_std == 0:
            random_noise = 1
        else:
            random_noise = self.rng.randn() * self.noise_std + 1.
        # regret = (f - self.best_err) / self.err_range
        # res_out = {
        #     # 'rank': (np.searchsorted(self.func.sorted_new_D_y, -f)+1)/len(self.func.sorted_new_D_y),
        #     # 'rank': (np.searchsorted(self.sorted_new_D_y, f) + 1) / len(self.sorted_new_D_y),
        #     'regret': regret,
        #     'log_regret': np.log10(regret)

        # }
        res_loss = {
            'val': f * random_noise,
        }
        return res_loss['val']

    def _load_api_config(self):
        return {
            hp_name: {
                'type': 'float', 'range': [0, 1]
            } for hp_name in self.hp_names
        }

    def _inst_to_config(self, inst):
        hp_params = {}
        for i, hp in enumerate(self.api_config):
            hp_params[hp] = inst[i]
        return hp_params

    def array_to_config(self, ret_param=True):
        if ret_param:
            self.old_D_x_params = []
            for d in range(len(self.old_D_x)):
                self.old_D_x_params.append([self._inst_to_config(inst) for inst in self.old_D_x[d]])

            self.new_D_x_param = [self._inst_to_config(inst) for inst in self.new_D_x]

            return self.old_D_x_params, self.old_D_y, self.new_D_x_param
        else:
            return self.old_D_x, self.old_D_y, self.new_D_x


class Blackbox:
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            eval_fun: Callable[[np.array], np.array],
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eval_fun = eval_fun

    def __call__(self, x: np.array) -> np.array:
        """
        :param x: shape (input_dim,)
        :return: shape (output_dim,)
        """
        assert x.shape == (self.input_dim,)
        y = self.eval_fun(x)
        assert y.shape == (self.output_dim,)
        return y


class BlackboxOffline(Blackbox):
    def __init__(
            self,
            X: np.array,
            y: np.array,
    ):
        """
        A blackbox whose evaluations are already known.
        To evaluate a new point, we return the value of the closest known point.
        :param input_dim:
        :param output_dim:
        :param X: list of arguments evaluated, shape (n, input_dim)
        :param y: list of outputs evaluated, shape (n, output_dim)
        """
        assert len(X) == len(y)
        n, input_dim = X.shape
        n, output_dim = y.shape

        from sklearn.neighbors import KNeighborsRegressor
        proj = KNeighborsRegressor(n_neighbors=1).fit(X, y)

        super(BlackboxOffline, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            eval_fun=lambda x: proj.predict(x.reshape(1, -1))[0]
        )
