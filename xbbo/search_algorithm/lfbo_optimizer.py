import logging
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize, OptimizeResult
from sklearn.ensemble import RandomForestClassifier
from xbbo.core.constants import MAXINT


from xbbo.core.trials import Trial, Trials
from xbbo.initial_design import ALL_avaliable_design

from . import alg_register
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration

logger = logging.getLogger(__name__)


@alg_register.register('lfbo')
class LFBO(AbstractOptimizer):
    '''
    ref: https://github.com/lfbo-ml/lfbo
    '''
    def __init__(self,
                 space,
                 seed: int = 42,
                 random_rate: float = 0.1,
                 initial_design: str = 'sobol',
                 suggest_limit: int = np.inf,
                 classify: str = 'rf',
                 **kwargs):

        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='one-hot',
                                   encoding_ord='bin',
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)

        # self.multi_start = multi_start(minimizer_fn=minimize)
        if self.space.get_conditions():
            raise NotImplementedError(
                "BORE optimizer currently does not support conditional space!")

        self.dimension = self.space.get_dimensions()
        self.classifier = Classfify(classify=classify, dim=self.dimension,rng=self.rng)
        bounds = self.space.get_bounds()
        self.bounds = Bounds(bounds.lb, bounds.ub)  #(bounds.lb, bounds.ub)
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=suggest_limit, **kwargs)
        self.init_budget = kwargs.get('init_budget')
        if self.init_budget is None:
            self.init_budget = self.initial_design.init_budget
            
        # self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )[:self.init_budget]

        self.trials = Trials(space, dim=self.dimension)
        self.random_rate = random_rate
        self.num_starts = kwargs.get("num_starts", 5)
        self.num_samples = kwargs.get("num_samples", 1024)
        self.method = kwargs.get("method", "L-BFGS-B")
        self.options = kwargs.get('options', dict(maxiter=1000, ftol=1e-9))
        self.quantile = kwargs.get("quantile", 0.33)

    def _suggest(self, n_suggestions=1):
        dataset_size = self.trials.trials_num

        # Insufficient training data
        if dataset_size < self.init_budget:
            logger.debug(f"Completed {dataset_size}/{self.init_budget}"
                         " initial runs. Suggesting random candidate...")
            return [
                Trial(configuration=config,
                      config_dict=config.get_dictionary(),
                      array=config.get_array(sparse=False),
                      origin='Random') for config in
                self.initial_design_configs[dataset_size:dataset_size +
                                            n_suggestions]
            ]

        # targets: historical y-values
        targets = self.trials.get_history()[0]

        X, Y, W = self._make_clf_data(self.trials.get_array(), targets)

        # update classifier
        self.classifier.fit(X, Y, W)

        X_init = self.rng.uniform(low=self.bounds.lb,
                                  high=self.bounds.ub,
                                  size=(self.num_samples, self.dimension))
        f_init = self.classifier.predict(X_init)  # to minimize

        #Calculate the minimum value of the fitted function
        trial_list = []
        for n_ in range(n_suggestions):
            results = []
            if self.num_starts > 0:
                ind = np.argpartition(f_init,
                                      kth=self.num_starts - 1,
                                      axis=None)
                for i in range(self.num_starts):
                    x0 = X_init[ind[i]]
                    result = minimize(self.classifier.predict,
                                      x0=x0,
                                      method=self.method,
                                      jac=False,
                                      bounds=self.bounds,
                                      options=self.options)
                    results.append(result)
                    # # TODO(LT): Make this message a customizable option.
                    # print(f"[Maximum {i+1:02d}: value={result.fun:.3f}] "
                    #         f"success: {result.success}, "
                    #         f"iterations: {result.nit:02d}, "
                    #         f"status: {result.status} ({result.message})")
            else:
                i = np.argmin(f_init, axis=None)
                result = OptimizeResult(x=X_init[i],
                                        fun=f_init[i],
                                        success=True)
                results.append(result)
            best_v = np.inf
            best_config = None
            for res in results:
                if (res.success or res.status == 1) and res.fun < best_v:
                    config = DenseConfiguration.from_array(self.space, res.x)
                    if not self.trials.is_contain(config):
                        best_config = config
            assert best_config is not None
            trial_list.append(
                Trial(configuration=config,
                      config_dict=config.get_dictionary(),
                      array=config.get_array(sparse=False)))

        return trial_list

    def _observe(self, trial_list):
        # for xx, yy in zip(features, y):
        for trial in trial_list:
            self.trials.add_a_trial(trial)

    def _is_unique(self, res, rtol=1e-5, atol=1e-8):
        is_duplicate = any(
            np.allclose(x_prev, res.x, rtol=rtol, atol=atol)
            for x_prev in self.trials.get_array())
        if is_duplicate:
            logger.warn("Duplicate detected! Skipping...")
        return not is_duplicate

    def _make_clf_data(self, X, Y, eta=1.0):
        '''
        Uility default use EI version(eta = 1.0)
        when eta == 0, i.e. PI version
        '''
        X = np.asarray(X)
        Y = np.asarray(Y)
        tau = np.quantile(Y, self.quantile)

        z = np.less(Y, tau)
        x1, y1 = X[z], z[z]
        x0, y0 = X, np.zeros_like(z)
        w1 = (tau - Y)[z]  # Utility
        w1 = w1**eta / np.mean(w1)  # scale constant
        w0 = 1 - y0
        s1 = x1.shape[0]
        s0 = x0.shape[0]

        X = np.concatenate([x1, x0], axis=0)
        Y = np.concatenate([y1, y0], axis=0)
        W = np.concatenate([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0],
                           axis=0)  # 正负样本数量均衡
        W = W / W.mean()
        return X, Y, W


class Classfify():
    def __init__(self, classify: str = 'rf', dim=0, rng=np.random.RandomState()):
        self.classify = classify
        if classify == 'rf':
            self.model = RFClassify(n_estimators=1000,
                                                min_samples_split=2,random_state=rng)
        elif classify == 'xgb':
            XGBClassify = _load_class('XGBClassify')
            self.model = XGBClassify(objective='binary:logistic',
                                        min_child_weight=1,
                                        learning_rate=0.3,
                                n_estimators=100,random_state=rng)
        elif classify == 'mlp':
            SequentialNN = _load_class('SequentialNN')
            self.model = SequentialNN(dim,random_state=rng
                )
        else:
            raise NotImplementedError()

    def fit(self, X, z, w=None):
        z = z.ravel()
        if w is None:
            w = np.ones_like(z, dtype='float')
        self.model.fit(X, z, sample_weight=w)

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return 1 - self.model.predict_proba(x)


def from_bounds(bounds):

    if isinstance(bounds, Bounds):
        low = bounds.lb
        high = bounds.ub
        dim = len(low)
        assert dim == len(high), "lower and upper bounds sizes do not match!"
    else:
        # assumes `bounds` is a list of tuples
        low, high = zip(*bounds)
        dim = len(bounds)

    return (low, high), dim


class RFClassify(RandomForestClassifier):
    def predict_proba(self,*args, **kwargs):
        return super().predict_proba(*args, **kwargs)[:,-1]

def _load_class(classname='XGBClassify'):
    if classname == 'XGBClassify':
        from xgboost import XGBClassifier
        class XGBClassify(XGBClassifier):
            def fit(self,*args, **kwargs):
                return super().fit(*args, eval_metric='logloss', callbacks=[], verbose=False, **kwargs)
            def predict_proba(self,*args, **kwargs):
                return super().predict_proba(*args, **kwargs)[:,-1]
        return XGBClassify
    elif classname == 'SequentialNN':
        import torch.nn as nn
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        class Network(nn.Module):
            def __init__(self,input_dim,
                output_dim=1,
                num_layers=2,
                num_units=32,) -> None:
                super().__init__()
                self.layers = nn.ModuleList()

                for i in range(num_layers):
                    if not i:
                        self.layers.append(nn.Linear(input_dim, num_units))
                        self.layers.append(nn.ReLU())
                    self.layers.append(nn.Linear(num_units, num_units))
                    self.layers.append(nn.ReLU())

                self.layers.append(nn.Linear(num_units, output_dim))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
            
        class SequentialNN():
            def __init__(self,input_dim,
                output_dim=1,
                num_layers=2,
                num_units=32,random_state=np.random.RandomState()) -> None:
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.num_layers = num_layers
                self.num_units = num_units
                seed = random_state.randint(MAXINT)
                torch.manual_seed(seed)
                self.net = Network(self.input_dim,self.output_dim, self.num_layers, self.num_units)
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=0.0)


            def fit(self, X, z, sample_weight):
                batch_size = 64

                X = torch.tensor(X, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float).unsqueeze(-1)
                sample_weight = torch.tensor(sample_weight, dtype=torch.float).unsqueeze(-1)

                dataset = TensorDataset(X, z, sample_weight)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

                for i in range(100//len(loader)):
                    for x, y, w in loader:
                        self.optimizer.zero_grad()
                        y_ = self.net(x)
                        loss = nn.BCEWithLogitsLoss(weight=w)(y_, y)
                        loss.backward()
                        self.optimizer.step()
            
            def predict_proba(self, x):
                x = torch.tensor(x, dtype=torch.float)
                with torch.no_grad():
                    y = torch.sigmoid(self.net(x))
                return y.squeeze().numpy()
            
        return SequentialNN

opt_class = LFBO