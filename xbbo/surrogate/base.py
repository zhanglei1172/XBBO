from typing import Dict, List, Optional, Tuple, Union
import typing
import numpy as np
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter,Constant
import sklearn.gaussian_process.kernels
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError
from skopt.learning.gaussian_process.kernels import Kernel
from skopt.learning.gaussian_process import GaussianProcessRegressor

from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.surrogate.gp_prior import Prior, SoftTopHatPrior, TophatPrior


class SurrogateModel(object):
    """Abstract implementation of the Model API.

    **Note:** The input dimensionality of Y for training and the output dimensions
    of all predictions (also called ``n_objectives``) depends on the concrete
    implementation of this abstract class.

    Attributes
    ----------
    instance_features : np.ndarray(I, K)
        Contains the K dimensional instance features
        of the I different instances
    pca : sklearn.decomposition.PCA
        Object to perform PCA
    pca_components : float
        Number of components to keep or None
    n_feats : int
        Number of instance features
    n_params : int
        Number of parameters in a configuration (only available after train has
        been called)
    scaler : sklearn.preprocessing.MinMaxScaler
        Object to scale data to be withing [0, 1]
    var_threshold : float
        Lower bound vor variance. If estimated variance < var_threshold, the set
        to var_threshold
    types : list
        If set, contains a list with feature types (cat,const) of input vector
    """
    def __init__(self,
                 types: np.ndarray,
                 bounds: typing.List[typing.Tuple[float, float]],
                 instance_features: np.ndarray = None,
                 pca_components: float = None,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        types : np.ndarray (D)
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds : list
            Specifies the bounds for continuous features.
        instance_features : np.ndarray (I, K)
            Contains the K dimensional instance features
            of the I different instances
        pca_components : float
            Number of components to keep when using PCA to reduce
            dimensionality of instance features. Requires to
            set n_feats (> pca_dims).
        """
        self.instance_features = instance_features
        self.pca_components = pca_components

        if instance_features is not None:
            self.n_feats = instance_features.shape[1]
        else:
            self.n_feats = 0

        self.n_params = None  # will be updated on train()

        self.pca = None
        self.scaler = None
        if self.pca_components and self.n_feats > self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            self.scaler = MinMaxScaler()

        # Never use a lower variance than this
        self.var_threshold = 10**-5

        self.bounds = bounds
        self.types = types
        # Initial types array which is used to reset the type array at every call to train()
        self._initial_types = types.copy()
        self.do_optimize = kwargs.get('do_optimize', True)

    def train(self, X: np.ndarray, Y: np.ndarray) -> 'SurrogateModel':
        """Trains the Model on X and Y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, n_objectives]
            The corresponding target values. n_objectives must match the
            number of target names specified in the constructor.

        Returns
        -------
        self : AbstractModel
        """
        self.types = self._initial_types.copy()

        if len(X.shape) != 2:
            raise ValueError('Expected 2d array, got %dd array!' %
                             len(X.shape))
        if X.shape[1] != len(self.types):
            raise ValueError(
                'Feature mismatch: X should have %d features, but has %d' %
                (X.shape[1], len(self.types)))
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X.shape[0] (%s) != y.shape[0] (%s)' %
                             (X.shape[0], Y.shape[0]))

        self.n_params = X.shape[1] - self.n_feats

        # reduce dimensionality of features of larger than PCA_DIM
        if self.pca and X.shape[0] > self.pca.n_components:
            X_feats = X[:, -self.n_feats:]
            # scale features
            X_feats = self.scaler.fit_transform(X_feats)
            X_feats = np.nan_to_num(X_feats)  # if features with max == min
            # PCA
            X_feats = self.pca.fit_transform(X_feats)
            X = np.hstack((X[:, :self.n_params], X_feats))
            if hasattr(self, "types"):
                # for RF, adapt types list
                # if X_feats.shape[0] < self.pca, X_feats.shape[1] ==
                # X_feats.shape[0]
                self.types = np.array(
                    np.hstack((self.types[:self.n_params],
                               np.zeros((X_feats.shape[1])))),
                    dtype=np.uint,
                )

        return self._train(X, Y)

    def _train(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> 'SurrogateModel':
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, n_objectives]
            The corresponding target values. n_objectives must match the
            number of target names specified in the constructor.

        Returns
        -------
        self
        """
        raise NotImplementedError

    def predict(
        self,
        X: np.ndarray,
        cov_return_type: typing.Optional[str] = 'diagonal_cov'
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance features)]
            Training samples

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            if len(X.shape) != 2:
                raise ValueError('Expected 2d array, got %dd array!' %
                                 len(X.shape))
            if X.shape[1] != len(self._initial_types):
                raise ValueError(
                    'Rows in X should have %d entries but have %d!' %
                    (len(self._initial_types), X.shape[1]))

            if self.pca:
                try:
                    X_feats = X[:, -self.n_feats:]
                    X_feats = self.scaler.transform(X_feats)
                    X_feats = self.pca.transform(X_feats)
                    X = np.hstack((X[:, :self.n_params], X_feats))
                except NotFittedError:
                    pass  # PCA not fitted if only one training sample

            if X.shape[1] != len(self.types):
                raise ValueError(
                    'Rows in X should have %d entries but have %d!' %
                    (len(self.types), X.shape[1]))

            mean, var = self._predict(X, cov_return_type=cov_return_type)
            if cov_return_type is None:
                return mean, var
            if len(mean.shape) == 1:
                mean = mean.reshape((-1, 1))
            if len(var.shape) == 1:
                var = var.reshape((-1, 1))

            return mean, var

    def _predict(
        self,
        X: np.ndarray,
        cov_return_type: typing.Optional[str] = 'diagonal_cov'
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config + instance features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        """
        raise NotImplementedError()

    def predict_marginalized_over_instances(
            self, X: np.ndarray, cov_return_type: typing.Optional[str] = 'diagonal_cov') -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if len(X.shape) != 2:
            raise ValueError('Expected 2d array, got %dd array!' %
                             len(X.shape))
        if X.shape[1] != len(self.types):
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (len(self.types), X.shape[1]))

        if self.instance_features is None or \
                len(self.instance_features) == 0:
            mean, var = self.predict(X, cov_return_type)
            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean, var
        else:
            n_instances = len(self.instance_features)

        mean = np.zeros(X.shape[0])
        var = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            X_ = np.hstack((np.tile(x,
                                    (n_instances, 1)), self.instance_features))
            means, vars = self.predict(X_, cov_return_type)
            # VAR[1/n (X_1 + ... + X_n)] =
            # 1/n^2 * ( VAR(X_1) + ... + VAR(X_n))
            # for independent X_1 ... X_n
            var_x = np.sum(vars) / (len(vars)**2)
            if var_x < self.var_threshold:
                var_x = self.var_threshold

            var[i] = var_x
            mean[i] = np.mean(means)

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var
    
    def update_weight(self, w, rho=None):
        pass


class BaseGP(SurrogateModel):
    def __init__(self,
                 configspace: DenseConfigurationSpace,
                 types: List[int],
                 bounds: List[Tuple[float, float]],
                 rng: np.random.RandomState,
                 normalize_y: bool = True,
                 instance_features: Optional[np.ndarray] = None,
                 pca_components: Optional[int] = None,
                 **kwargs):
        """
        Abstract base class for all Gaussian process models.
        """
        super().__init__(types=types,
                         bounds=bounds,
                         instance_features=instance_features,
                         pca_components=pca_components,
                         **kwargs)

        self.configspace = configspace
        self.rng = rng
        self.normalize_y = normalize_y
        kernel = kwargs.get('kernel')
        self.kernel = kernel if kernel else self._get_kernel()
        self.gp = self._get_gp()

    def _get_kernel(self) -> Kernel:
        raise NotImplementedError()

    def _get_gp(self) -> GaussianProcessRegressor:
        raise NotImplementedError()

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean unit standard deviation.

        Parameters
        ----------
        y : np.ndarray
            Targets for the Gaussian process

        Returns
        -------
        np.ndarray
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

        This function should be used after a prediction with the Gaussian process which was trained on normalized data.

        Parameters
        ----------
        y : np.ndarray
            Normalized data.
        var : np.ndarray (optional)
            Normalized variance

        Returns
        -------
        np.ndarray on Tuple[np.ndarray, np.ndarray]
        """
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_**2
            return y, var
        return y

    def _get_all_priors(
        self,
        add_bound_priors: bool = True,
        add_soft_bounds: bool = False,
    ) -> List[List[Prior]]:
        # Obtain a list of all priors for each tunable hyperparameter of the kernel
        all_priors = []
        to_visit = []
        # to_visit.append(self.gp.kernel.k1)
        # to_visit.append(self.gp.kernel.k2)
        to_visit.append(self.gp.kernel)  # fix single kernel
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param,
                          sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                continue
            elif isinstance(current_param,
                            sklearn.gaussian_process.kernels.Kernel):
                hps = current_param.hyperparameters
                assert len(hps) == 1
                hp = hps[0]
                if hp.fixed:
                    continue
                bounds = hps[0].bounds
                for i in range(hps[0].n_elements):
                    priors_for_hp = []
                    if current_param.prior is not None:
                        priors_for_hp.append(current_param.prior)
                    if add_bound_priors:
                        if add_soft_bounds:
                            priors_for_hp.append(
                                SoftTopHatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    rng=self.rng,
                                    exponent=2,
                                ))
                        else:
                            priors_for_hp.append(
                                TophatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    rng=self.rng,
                                ))
                    all_priors.append(priors_for_hp)
        return all_priors

    def _set_has_conditions(self) -> None:
        has_conditions = len(self.configspace.get_conditions()) > 0
        to_visit = []
        to_visit.append(self.kernel)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param,
                          sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                current_param.has_conditions = has_conditions
            elif isinstance(current_param,
                            sklearn.gaussian_process.kernels.Kernel):
                current_param.has_conditions = has_conditions
            else:
                raise ValueError(current_param)

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[~np.isfinite(X)] = -1
        return X


class Surrogate():
    def __init__(self, cs, min_sample):
        self.cs = cs
        self.min_sample = min_sample
        pass

    def predict(self, newX):
        pass

    def fit(self, x, y):
        pass

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        self.mean_y = np.mean(y)
        self.std_y = np.std(y)
        if self.std_y == 0:
            self.std_y = 1
        return (y - self.mean_y) / self.std_y

    def _untransform_y(
        self,
        y: np.ndarray,
        var: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform zeromean unit standard deviation data into the regular space.

        This function should be used after a prediction with the Gaussian process which was trained on normalized data.

        Parameters
        ----------
        y : np.ndarray
            Normalized data.
        var : np.ndarray (optional)
            Normalized variance

        Returns
        -------
        np.ndarray on Tuple[np.ndarray, np.ndarray]
        """
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_**2
            return y, var
        return y

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[~np.isfinite(X)] = -1
        return X


class BaseRF(SurrogateModel):
    def __init__(self,
                 configspace: DenseConfigurationSpace,
                 types: typing.List[int],
                 bounds: List[Tuple[float, float]],
                 instance_features: Optional[np.ndarray] = None,
                 pca_components: Optional[int] = None,
                 **kwargs) -> None:
        """
        Abstract base class for all random forest models.
        """
        self.configspace = configspace
        super().__init__(types=types,
                         bounds=bounds,
                         instance_features=instance_features,
                         pca_components=pca_components,
                         **kwargs)

        self.conditional = dict()  # type: Dict[int, bool]
        self.impute_values = dict()  # type: Dict[int, float]

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for idx, hp in enumerate(self.configspace.get_hyperparameters()):
            if idx not in self.conditional:
                parents = self.configspace.get_parents_of(hp.name)
                if len(parents) == 0:
                    self.conditional[idx] = False
                else:
                    self.conditional[idx] = True
                    if isinstance(hp, CategoricalHyperparameter):
                        self.impute_values[idx] = len(hp.choices)
                    elif isinstance(hp, (UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)):
                        self.impute_values[idx] = -1
                    elif isinstance(hp, Constant):
                        self.impute_values[idx] = 1
                    else:
                        raise ValueError

            if self.conditional[idx] is True:
                nonfinite_mask = ~np.isfinite(X[:, idx])
                X[nonfinite_mask, idx] = self.impute_values[idx]

        return X
