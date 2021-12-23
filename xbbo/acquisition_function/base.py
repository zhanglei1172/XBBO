import numpy as np
import abc
from typing import List, Union
from xbbo.configspace.space import DenseConfiguration, convert_denseConfigurations_to_array
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

    def __init__(self, surrogate_model: Union[SurrogateModel, List[SurrogateModel]], **kwargs):
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

    def __call__(self, configurations: Union[List[DenseConfiguration], np.ndarray], convert=True, **kwargs):
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

