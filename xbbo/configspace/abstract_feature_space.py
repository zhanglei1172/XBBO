from abc import ABC
import numpy as np
import warnings
from scipy import stats



class AbstractFeatureSpace(ABC):
    '''
    for optimizer using

    sparse array => features
    '''


    def __init__(self):
        pass

    # @abstractmethod
    def array_to_feature(self, array):
        '''
        array: expressed by configspace
        feature: expressed by optimizer
        '''
        pass

    # @abstractmethod
    def feature_to_array(self, feature):
        pass

class Identity():
    '''
    uniform to Gaussian
    '''

    def __init__(self):
        '''
        U(0, 1) to std gaussian feature
        define target feature's mean and std
        '''
        super().__init__()

    def sparse_array_to_feature(self, sparse_array):

        return sparse_array

    def feature_to_sparse_array(self, feature):
        return feature

class Cat2Onehot():

    def __init__(self):
        super().__init__()

    def sparse_array_to_feature(self, sparse_array, cat_num):
        '''
        sparse_array: int, index of max value
        return: one-hot code
        '''
        feat = np.zeros(cat_num)
        feat[int(sparse_array)] = 1
        return feat

    def feature_to_sparse_array(self, feature, cat_num):
        return np.argmax(feature).item()

class Ord2Uniform():
    '''
    ordinal to uniform(0, 1)
    '''

    def __init__(self):
        super().__init__()

    def sparse_array_to_feature(self, sparse_array, seqnum):
        '''
        sparse_array: int , one of 0, 1, 2 ... (seqnum-1)
        return a float in [0, 1]
        '''

        return sparse_array / (seqnum-1)

    def feature_to_sparse_array(self, feature, seqnum):
        return np.rint(feature * (seqnum-1))

class U2gaussian():
    '''
    uniform to Gaussian
    '''

    def __init__(self):
        '''
        U(0, 1) to std gaussian feature
        define target feature's mean and std
        '''
        super().__init__()

    def sparse_array_to_feature(self, sparse_array):
        '''
        convert to Gaussian distribution
        '''
        return stats.norm.ppf(sparse_array)

    def feature_to_sparse_array(self, feature):
        return stats.norm.cdf(feature)

class U2Onehot():
    def __init__(self):
        AbstractFeatureSpace.__init__(self)

    def sparse_array_to_feature(self, sparse_array, cat_num):
        '''
        sparse_array: 0~1
        return: one-hot code
        '''
        feat = np.zeros(cat_num)
        feat[np.uintp(sparse_array*cat_num)] = 1
        return feat

    def feature_to_sparse_array(self, feature, cat_num):
        return np.argmax(feature, -1).item() / (cat_num-1)


class Ordinal():

    def __init__(self):
        super().__init__()

    
    def feature_to_sparse_array(self, feature, seq_num):
        '''
        return int, one of 0, 1, 2 ... (seqnum-1)
        '''
        return threshold_discretization(feature, arity=seq_num)

    def sparse_array_to_feature(self, sparse_array, seq_num):
        '''
        feature： int , one of 0, 1, 2 ... (seqnum-1)
        return： N(0, 1)
        '''
        return inverse_threshold_discretization(sparse_array, arity=seq_num)


class Gaussian():

    def __init__(self, mean=0, std=1):
        '''
        std gaussian to specify gaussian
        define target feature's mean and std
        '''
        super().__init__()
        self.mean = mean
        self.std = std

    def sparse_array_to_feature(self, sparse_array):
        '''
        convert to Gaussian distribution
        '''
        return self.std * sparse_array + self.mean

    def feature_to_sparse_array(self, feature):
        return (feature - self.mean) / self.std

class Category():

    def __init__(self, deterministic=False):
        super().__init__()
        # self.cat_num = cat_num
        self.deterministic = deterministic

    def feature_to_sparse_array(self, feature, cat_num):
        '''
        return a int index
        '''

        return int(softmax_discretization(feature, cat_num, deterministic=self.deterministic))

    def sparse_array_to_feature(self, sparse_array, cat_num):
        '''
        convert to Gaussian distribution
        '''
        return inverse_softmax_discretization(sparse_array, cat_num)



def threshold_discretization(x, arity: int = 2):
    """Discretize by casting values from 0 to arity -1, assuming that x values
    follow a normal distribution.

    Parameters
    ----------
    x: list/array
       values to discretize
    arity: int
       the number of possible integer values (arity n will lead to values from 0 to n - 1)

    Note
    ----
    - nans are processed as negative infs (yields 0)
    """
    x = np.array(x, copy=True)
    if np.any(np.isnan(x)):
        warnings.warn("Encountered NaN values for discretization")
        x[np.isnan(x)] = -np.inf
    if arity == 2:  # special case, to have 0 yield 0
        return (np.array(x) > 0).astype(int)#.tolist()  # type: ignore
    else:
        return np.clip(arity * stats.norm.cdf(x), 0, arity - 1).astype(int)#.tolist()  # type: ignore


def inverse_threshold_discretization(indexes, arity: int = 2):
    '''
    to N(0,1)
    '''
    indexes_arr = np.array(indexes, copy=True)
    pdf_bin_size = 1 / arity
    # We take the center of each bin (in the pdf space)
    return stats.norm.ppf(indexes_arr * pdf_bin_size + (pdf_bin_size / 2))


def softmax_discretization(x, arity: int = 2, deterministic: bool = False):
    """Discretize a list of floats to a list of ints based on softmax probabilities.
    For arity n, a softmax is applied to the first n values, and the result
    serves as probability for the first output integer. The same process it
    applied to the other input values.

    Parameters
    ----------
    x: list/array
        the float values from a continuous space which need to be discretized
    arity: int
        the number of possible integer values (arity 2 will lead to values in {0, 1})
    deterministic: bool
        removes all randomness by returning the last mast probable output (highest values)

    Notes
    -----
    - if one or several inf values are present, only those are considered
    - in case of tie, the deterministic value is the first one (lowest) of the tie
    - nans and -infs are ignored, except if all are (then uniform random choice)
    """
    # data = np.array(x, copy=True, dtype=float).reshape((-1, arity))
    # if np.any(np.isnan(data)):
    #     warnings.warn("Encountered NaN values for discretization")
    #     data[np.isnan(data)] = -np.inf
    # if deterministic:
    #     output = np.argmax(data, axis=1).tolist()
    #     return output
    # return [np.random.choice(arity, p=softmax_probas(d)) for d in data]
    data = np.array(x, copy=True, dtype=float)
    if np.any(np.isnan(data)):
        warnings.warn("Encountered NaN values for discretization")
        data[np.isnan(data)] = -np.inf
    if deterministic:
        output = np.argmax(data, axis=1)
        return output
    return np.random.choice(arity, p=softmax_probas(data))
    # data = np.array(x, copy=True, dtype=float)#.reshape((-1, arity))
    # output = np.zeros_like(data)
    # if np.any(np.isnan(data)):
    #     warnings.warn("Encountered NaN values for discretization")
    #     data[np.isnan(data)] = -np.inf
    # if deterministic:
    #     max_idx = np.argmax(data, axis=1)
    #     output[np.arange(output.shape[0]), max_idx] = 1
    #     return output
    # for i, d in enumerate(data):
    #     max_idx = np.random.choice(arity, p=softmax_probas(d))
    #     output[i, max_idx] = 1
    # return output

def softmax_probas(data: np.ndarray) -> np.ndarray:
    # TODO: test directly? (currently through softmax discretization)
    # TODO: move nan case here?
    maxv = np.max(data)
    if np.abs(maxv) == np.inf or np.isnan(maxv):
        maxv = 0
    data = np.exp(data - maxv)
    if any(x == np.inf for x in data):  # deal with infinite positives special case
        data = np.array([int(x == np.inf) for x in data])
    if not sum(data):
        data = np.ones(len(data))
    return data / np.sum(data)

def inverse_softmax_discretization(_x: int, arity: int):
    # p is an arbitrary probability that the provided arg will be sampled with the returned point
    p = (1 / arity) * 1.5
    # x = np.zeros(arity)
    x = np.zeros(arity)
    x[int(_x)] = np.log((p * (arity - 1)) / (1 - p))
    return x