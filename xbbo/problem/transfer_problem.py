from abc import abstractmethod
import os
from enum import Enum
from typing import Tuple, List, Callable
import numpy as np
import pandas as pd
from pathlib import Path
from ConfigSpace import ConfigurationSpace
import ConfigSpace as CS
from ConfigSpace.conditions import InCondition, LessThanCondition
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from xbbo.core.constants import MAXINT, Key
from xbbo.problem.base import AbstractBenchmark

class BenchName(Enum):
    TST = 0
    # surrogate = 1
    Table_deepar = 1
    Table_fcnet = 2
    Table_xgboost = 3
    Table_nas102 = 4

class TransferData():
    def __init__(self, bench_name:int, data_path_root:str, data_base_name:str, target_task_name:str) -> None:
        self.bench_name = bench_name
        self.data_base_name = data_base_name
        self.data_path_root = data_path_root
        self.target_task_name = target_task_name
    
    def load_data(self,):
        key = str(self.__class__) + "_" + self.target_task_name
        res = CACHE_DATA.get(key, False)
        if res:
            return res
        CACHE_DATA[key] = self._load_data()
        return CACHE_DATA[key]

    def _load_data(self):
        pass
    
    def get_configuration_space(self,):
        pass
    
    @abstractmethod
    def download_data(self,):
        pass
    
    @property
    def hp_names(self,):
        return None

# prevent duplicate load data
CACHE_DATA = {}


class TST_Data(TransferData):
    def __init__(self, bench_name:int,data_path_root:str, data_base_name:str,target_task_name:str, download=True, sparse=False, hp_num=3,min_max_features=False, rng=np.random.RandomState(), **kwargs) -> None:
        super().__init__(bench_name,data_path_root, data_base_name, target_task_name)
        self.data_path = os.path.join(self.data_path_root, data_base_name)
        self.min_max_features = min_max_features
        self.sparse = sparse
        self.hp_num = hp_num
        self.download = download
        self.url = "https://git.openi.org.cn/isleizhang/BBO-Datasets/datasets"
        self.rng = rng
        self.hp_keys = ['C', 'gamma', 'd']
        
    def _load_data(self):
        if not os.path.exists(self.data_path):
            assert self.download, 'ERROR: "{}" not exits.'.format(self.data_path)
            self.download_data()
        file_lists = os.listdir(self.data_path)
        file_lists = list(map(lambda x: os.path.join(self.data_path,x), file_lists))
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
                    line_array.extend(line_array_raw[idx_start:self.hp_num + 1+3])
                    insts.append(line_array)

            datasets = np.asarray(insts, dtype=np.float)
            if self.sparse:
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
        if self.min_max_features:
            # min-max scaling of input features
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler().fit(np.vstack(datasets_hp))
            datasets_hp = [scaler.transform(X) for X in datasets_hp]
        test_idx = filenames.index(self.target_task_name)
        test_task = datasets_hp.pop(test_idx)
        test_task_label = datasets_label.pop(test_idx)
        
        return (datasets_hp, datasets_label, test_task, test_task_label)

    def get_configuration_space(self):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        x0 = UniformFloatHyperparameter("C", -1, 1)
        x1 = UniformFloatHyperparameter("gamma", -1, 1)
        x2 = UniformFloatHyperparameter("d", 0, 1)
        self.configuration_space.add_hyperparameters([x0, x1, x2])
        return self.configuration_space
    
    @property
    def hp_names(self,):
        return self.hp_keys

    def download_data(self):
        raise NotImplementedError("plese download {} in {}".format(self.url, self.data_path))

class Table_Data(TransferData):
    blackbox_tasks = {
    BenchName.Table_nas102: [
        'cifar10',
        'cifar100',
        'ImageNet16-120'
    ],
    BenchName.Table_fcnet: [
        'naval',
        'parkinsons',
        'protein',
        'slice',
    ],
    BenchName.Table_deepar: [
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
    BenchName.Table_xgboost: [
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
        BenchName.Table_deepar: 'metric_CRPS',
        BenchName.Table_fcnet: 'metric_error',
        BenchName.Table_nas102: 'metric_error',
        BenchName.Table_xgboost: 'metric_error',
    }
    
    def __init__(self, bench_name:int,data_path_root:str, data_base_name:str, target_task_name:str, download=True, sparse=False, hp_num=3,min_max_features=False, rng=np.random.RandomState(), **kwargs) -> None:
        super().__init__(bench_name,data_path_root, data_base_name, target_task_name)
        self.data_base_name = data_base_name
        self.data_path = os.path.join(data_path_root, data_base_name)
        self.min_max_features = min_max_features
        self.sparse = sparse
        self.hp_num = hp_num
        self.download = download
        self.url = "https://git.openi.org.cn/isleizhang/BBO-Datasets/datasets"
        self.rng = rng
        self._metric_col = self.error_metric[bench_name]
        
    def _load_data(self):
        if not os.path.exists(self.data_path):
            assert self.download, 'ERROR: "{}" not exits.'.format(self.data_path)
            self.download_data()
        df = pd.read_csv(self.data_path)

        assert self.target_task_name in df.task.unique()
        assert self._metric_col in df.columns

        Xy_dict = {}
        for task in sorted(df.task.unique()):
            mask = df.loc[:, 'task'] == task
            hp_cols = [c for c in sorted(df.columns) if c.startswith("hp_")]
            X = df.loc[mask, hp_cols].values
            y = df.loc[mask, self._metric_col].values
            if len(y.shape) == 1:
                y = np.expand_dims(y, axis=1)
            Xy_dict[task] = X, y

        # todo it would be better done as a post-processing step
        if self.bench_name in [BenchName.Table_fcnet, BenchName.Table_nas102]:
            # applies onehot encoding to *all* hp columns as all hps are categories for those two blackboxes
            # it would be nice to detect column types or pass it as an argument
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            hp_cols = [c for c in sorted(df.columns) if c.startswith("hp_")]
            enc.fit(df.loc[:, hp_cols])
            for task, (X, y) in Xy_dict.items():
                X_features = enc.transform(X)
                Xy_dict[task] = X_features, y

        if self.min_max_features:
            # min-max scaling of input features
            from sklearn.preprocessing import MinMaxScaler
            X = np.vstack([X for (X, y) in Xy_dict.values()])
            scaler = MinMaxScaler().fit(X)
            Xy_dict = {t: (scaler.transform(X), y) for (t, (X, y)) in Xy_dict.items()}

        Xys_train = [Xy_dict[t] for t in df.task.unique() if t != self.target_task_name]
        Xy_test = Xy_dict[self.target_task_name]
        self.hp_keys = [f'hp_{i}' for i in range(Xy_test[0].shape[1])]
        L = list(zip(*Xys_train))
        L.extend(Xy_test)
        return L

    def get_configuration_space(self):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for name in self.hp_keys:
            x = UniformFloatHyperparameter(name, 0, 1)
            self.configuration_space.add_hyperparameter(x)
        return self.configuration_space

    @property
    def hp_names(self,):
        return self.hp_keys

    def download_data(self):
        raise NotImplementedError("plese download {} in {}".format(self.url, self.data_path))
        # download_and_extract_archive(self.url, download_root=self.data_path_root, filename=self.data_path, remove_finished=True)

class TransferBenchmark(AbstractBenchmark):

    def __init__(self, bench_name:int, data_base_name:str, target_task_name:str, rng=np.random.RandomState(), normalize_y=False, data_path_root='./data',**kwargs):
        # np.random.seed(cfg.GENERAL.random_seed)
        self.bench_name = bench_name
        self.target_task_name = target_task_name
        self.normalize_y = normalize_y
        self.data_path_root = data_path_root
        super().__init__(rng)
        
        if bench_name == BenchName.TST:
            self.data_loader = TST_Data(bench_name=bench_name,data_path_root=data_path_root, data_base_name=data_base_name,target_task_name=target_task_name, rng=self.rng,**kwargs)
            # self.old_D_x, self.old_D_y, self.new_D_x, self.new_D_y, self.hp_config = 
            # self.api_config = self.hp_config
        elif bench_name in [BenchName.Table_deepar, BenchName.Table_fcnet, BenchName.Table_nas102, BenchName.Table_xgboost]:
            self.data_loader = Table_Data(bench_name=bench_name,data_path_root=data_path_root, data_base_name=data_base_name,target_task_name=target_task_name,min_max_features=True, rng=self.rng,**kwargs)
        else:
            raise NotImplemented
        if normalize_y:
            self.old_D_y = [(y - y.min())/(y.max()-y.min()) for y in self.old_D_y]

        self._old_D_x, self._old_D_y, _new_D_x, _new_D_y = self.data_loader.load_data()
        self._bbfunc = BlackboxOffline(_new_D_x, _new_D_y)
        self._best_f = min(_new_D_y).item()
        self._f_range = max(_new_D_y).item() - self._best_f
        self._sorted_new_D_y = np.sort(_new_D_y).ravel()
        self.get_configuration_space()
        self.idxs = []
        for name in self.data_loader.hp_names:
            self.idxs.append(self.configuration_space.get_idx_by_hyperparameter_name(name))
        self.idxs = np.argsort(self.idxs)
        
    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        f = self._bbfunc(np.asarray([config[k] for k in self.data_loader.hp_names])).item()
        y = (f - self._best_f) / self._f_range if self.normalize_y else f

        return {Key.FUNC_VALUE: y}
    
    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = self.data_loader.get_configuration_space()
        return self.configuration_space

    def get_old_data(self):
        return np.take(self._old_D_x, self.idxs, axis=-1), self._old_D_y
    
    def get_old_configurations(self,):
        return CS.Configuration(self.configuration_space, vector=np.take(self._old_D_x,self.idxs, axis=-1)), self._old_D_y
        
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Transfer blackbox benchmark'}

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

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            eval_fun=lambda x: proj.predict(x.reshape(1, -1))[0]
        )
        
        
if __name__ == "__main__":
    bench = TransferBenchmark(BenchName.Table_deepar, 'DeepAR.csv.zip', target_task_name="m4-Hourly", data_path_root='./data/offline_evaluations')
    cs = bench.get_configuration_space()
    bench(cs.get_default_configuration())