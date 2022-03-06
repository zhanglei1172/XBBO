# License: MIT
# ref: https://github.com/thomas-young-2013/open-box/blob/master/openbox/surrogate/skrf.py
import logging
import typing
import numpy as np
from typing import List, Optional, Tuple, Union

from xbbo.surrogate.base import BaseRF
from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.utils.constants import MAXINT
from xbbo.utils.util import get_types


logger = logging.getLogger(__name__)


class RandomForestSurrogate(BaseRF):

    def __init__(
            self,
            configspace: DenseConfigurationSpace,
            ensemble_size: int = 10,
            normalize_y: bool = True,
            instance_features: typing.Optional[np.ndarray] = None,
            pca_components: typing.Optional[int] = None,
            rng: np.random.RandomState = np.random.RandomState(42),
            **kwargs
    ):
    
        self.model_config = dict()
        self.model_config["n_estimators"] = 10
        self.model_config["criterion"] = "mse"
        self.model_config["max_depth"] = 12
        self.model_config["min_samples_split"] = 3
        self.model_config["min_samples_leaf"] = 3
        self.model_config["min_weight_fraction_leaf"] = 0.
        self.model_config["max_features"] = 5. / 6.
        self.model_config["max_leaf_nodes"] = None
        self.model_config["n_jobs"] = -1
        # self.model_config["random_state"] = -1
        # self.model_config["max_samples"] = 1.

        self.ensemble_size = ensemble_size
        self.models = list()
        self.configspace = configspace
        self.rng = rng
        self.random_seeds = self.rng.randint(low=1, high=MAXINT, size=self.ensemble_size)
        types, bounds = get_types(configspace)
        super().__init__(
            configspace=configspace,
            types=types,
            bounds=bounds,
            instance_features=instance_features,
            pca_components=pca_components,
            **kwargs
        )

        self.normalize_y = normalize_y
        self.is_trained = False

    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train a Random Forest Regression model on X and y

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        """
        from sklearn.ensemble import RandomForestRegressor
        X = self._impute_inactive(X)
        if self.normalize_y:
            y = self._normalize_y(y)

        self.models = list()
        for i in range(self.ensemble_size):
            configs = self.model_config.copy()
            configs["random_state"] = self.random_seeds[i]
            rf_model = RandomForestRegressor(**configs)
            rf_model.fit(X, y)
            self.models.append(rf_model)

        self.is_trained = True
        return self

    def _predict(self, X_test: np.ndarray, **kwargs):
        r"""
        Returns the predictive mean and variance of the objective function

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        X_test = self._impute_inactive(X_test)

        predictions = list()
        for i, model in enumerate(self.models):
            pred = model.predict(X_test)
            # print(i)
            predictions.append(pred)

        m = np.mean(predictions, axis=0)
        v = np.var(predictions, axis=0)

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_y:
            m, v = self._untransform_y(m, v)

        return m, v

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean unit standard deviation.
        """
        self.mean_y_ = np.mean(y)
        self.std_y_ = np.std(y)
        if self.std_y_ == 0:
            self.std_y_ = 1
        return (y - self.mean_y_) / self.std_y_

    def _untransform_y(
            self,
            y: np.ndarray,
            var: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform zeromean unit standard deviation data into the regular space.
        """
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_ ** 2
            return y, var
        return y

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[~np.isfinite(X)] = -1
        return X
