from collections import OrderedDict
import numpy as np
import abc
from typing import Iterable, List, Tuple, Union
# import logging

from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace, convert_denseConfigurations_to_array
from xbbo.core.trials import Trials
from xbbo.surrogate.base import SurrogateModel


class AbstractAcquisitionFunction(object, metaclass=abc.ABCMeta):
    """Abstract base class for acquisition function

    Attributes
    ----------
    model
    logger
    """

    # def __str__(self):
    #     return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, surrogate_model: Union[SurrogateModel,
                                              List[SurrogateModel]], **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            Models the objective function.
        """
        self.surrogate_model = surrogate_model
        # self.logger = logging.getLogger(
        #     self.__module__ + "." + self.__class__.__name__)

    def update(self, **kwargs):
        """Update the acquisition functions values.

        This method will be called if the surrogate is updated. E.g.
        entropy search uses it to update its approximation of P(x=x_min),
        EI uses it to update the current optimizer.

        The default implementation takes all keyword arguments and sets the
        respective attributes for the acquisition function object.

        Parameters
        ----------
        kwargs
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self,
                 configurations: Union[List[DenseConfiguration], np.ndarray],
                 convert=True,
                 **kwargs):
        """Computes the acquisition value for a given X

        Parameters
        ----------
        configurations : list
            The configurations where the acquisition function
            should be evaluated.
        convert : bool

        Returns
        -------
        np.ndarray(N, 1)
            acquisition values for X
        """
        if convert:
            X = convert_denseConfigurations_to_array(configurations)
        else:
            X = configurations  # to be compatible with multi-objective acq to call single acq
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X, **kwargs)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the acquisition value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Acquisition function values wrt X
        """
        raise NotImplementedError()
    
    def update_weight(self, w, rho=None):
        pass


class AcquisitionFunctionMaximizer(object, metaclass=abc.ABCMeta):
    """Abstract class for acquisition maximization.

    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.

    Parameters
    ----------
    acquisition_function : ~xbbo.acquisition_function.acquisition.AbstractAcquisitionFunction

    config_space : ~xbbo.config_space.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """
    def __init__(self,
                 acquisition_function: AbstractAcquisitionFunction,
                 config_space: DenseConfigurationSpace,
                 rng: np.random.RandomState = np.random.RandomState(42)):
        self.acquisition_function = acquisition_function
        self.config_space = config_space
        # self.logger = logging.getLogger(self.__module__ + "." +
                                        # self.__class__.__name__)
        self.rng = rng

    def maximize(self,
                 trials: Trials,
                 num_points: int,
                 drop_self_duplicate: bool = False,
                 **kwargs) -> Iterable[DenseConfiguration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        trials: ~xbbo.utils.history_container.HistoryContainer
            trials object
        stats: ~xbbo.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consisting of :class:`xbbo.config_space.DenseConfiguration`.
        """
        configs = [t[1] for t in self._maximize(trials, num_points, **kwargs)]
        return self.unique(configs=configs) if drop_self_duplicate else configs

    @staticmethod
    def unique(configs: Iterable[DenseConfiguration]):
        return list(OrderedDict.fromkeys(configs))

    @abc.abstractmethod
    def _maximize(self, trials: Trials, num_points: int,
                  **kwargs) -> Iterable[Tuple[float, DenseConfiguration]]:
        """Implements acquisition function maximization.

        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the DenseConfiguration. This
        allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        trials: ~xbbo.utils.history_container.HistoryContainer
            trials object
        stats: ~xbbo.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`xbbo.config_space.DenseConfiguration`).
        """
        raise NotImplementedError()

    def _sort_configs_by_acq_value(
        self, configs: List[DenseConfiguration]
    ) -> List[Tuple[float, DenseConfiguration]]:
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        configs : list(DenseConfiguration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """

        acq_values = self.acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
        # seen = set()
        # seen_add = seen.add

        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]
