import numpy as np
from scipy.interpolate import interp1d
from scipy.special import expit as logistic  # because nobody calls it expit
from scipy.special import logit
from copy import deepcopy
import warnings
import ConfigSpace.hyperparameters as CSH
# from xbbo.utils.util import clip_chk
'''
warp for dict
'''
M = 1e7

WARPED_DTYPE = np.float_
N_GRID_DEFAULT = 8


def identity(x):
    """Helper function that perform warping in linear configspace. Sort of a no-op.

    Parameters
    ----------
    x : scalar
        Input variable in linear configspace. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : scalar
        Same as input `x`.
    """
    y = x
    return y


def bilog(x):
    """Bilog warping function. Extension of log to work with negative numbers.

    ``Bilog(x) ~= log(x)`` for large `x` or ``-log(abs(x))`` if `x` is negative. However, the bias term ensures good
    behavior near 0 and ``bilog(0) = 0``.

    Parameters
    ----------
    x : scalar
        Input variable in linear configspace. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : float
        The bilog of `x`.
    """
    y = np.sign(x) * np.log(1.0 + np.abs(x))
    return y


def biexp(x):
    """Inverse of :func:`.bilog` function.

    Parameters
    ----------
    x : scalar
        Input variable in linear configspace. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : float
        The biexp of `x`.
    """
    y = np.sign(x) * (np.exp(np.abs(x)) - 1.0)
    return y


WARP_DICT = {"linear": identity, "log": np.log, "logit": logit, "bilog": bilog, 'log10': np.log10}
UNWARP_DICT = {"linear": identity, "log": np.exp, "logit": logistic, "bilog": biexp, 'log10': lambda x: 10**x}
ROUND_DICT = {"linear": identity, "round": np.rint}

class Warp():

    def __init__(self):
        '''
        all_warp: a dict, {hp_name: warp_name}
        all_round: a dict, {hp_name: （round_func_name param_values）} (need encode)      or
                            {hp_name: （round_func_name None）} (no need encode)
        '''
        self.all_warp = {}
        # self.all_round = {}
        self.space_warped = {}
        self.all_types = {}
        # self.dtype_map = {}
        # self.need_encodes = {}

    def warp_space(self,
                   dtype,
                   param_name,
                   param_range=None,
                   param_values=None,
                   warp='linear',
                   # discrete_method='linear',
                   ):
        '''
        type: one of {'cat', 'ord', 'float'}
        return range: [lower_warped, upper_warped] or values
        '''
        self.all_warp[param_name] = warp
        # self.dtype_map[param_name] = dtype
        if dtype == 'cat':

            assert warp == 'linear', "离散超参没有必要warp！！！"
            assert not (param_values is None)
            # assert discrete_method == 'linear'
            # self.all_round[param_name] = (discrete_method, None)
            param_values = np.sort(np.unique(param_values))
            self.space_warped[param_name] = param_values
            # return param_values
            self.all_types[param_name] = 'cat'
            return CSH.CategoricalHyperparameter(name=param_name, choices=param_values)
        elif dtype == 'ord':
            assert not (param_values is None)
            assert warp == 'linear', "离散超参没有必要warp！！！"
            # param_values = np.sort(np.unique(np.asarray(param_values, dtype='str')))
            # self.all_round[param_name] = (discrete_method, param_values)
            # UniformIntegerHyperparameter(linear)、UniformFloatHyperparameter(round)
            self.space_warped[param_name] = param_values
            # if discrete_method != 'linear':
            #     warnings.warn("You should use UniformIntegerHyperparameter with discrete_method='linear' instead")
            #     return CSH.UniformFloatHyperparameter(name=param_name, lower=0, upper=len(param_values) - 1)

            # return 0, len(param_values) - 1
            self.all_types[param_name] = 'ord'
            return CSH.OrdinalHyperparameter(name=param_name, sequence=param_values)
        elif dtype in ('float', 'int'): # discrete_method actually use, int
            assert (param_values is None)
            assert not (param_range is None)
            param_range = np.asarray(param_range)
            if not np.isfinite(param_range[0]):
                param_range[0] = -M
            if not np.isfinite(param_range[1]):
                param_range[1] = M

            # self.all_round[param_name] = (discrete_method, None)
            # return self.space_warped[param_name]
            if dtype == 'float': # discrete_method can be round
                warp_func = WARP_DICT[warp]
                self.space_warped[param_name] = warp_func(param_range)
                self.all_types[param_name] = 'float'
                return CSH.UniformFloatHyperparameter(
                    name=param_name,
                    lower=self.space_warped[param_name][0],
                    upper=self.space_warped[param_name][1]
                )
            elif dtype == 'int':
                assert warp in ('linear', 'log'), "int type only support linear or log space"
                self.all_warp[param_name] = 'linear' # 强制修改为'linear'
                warp_func = WARP_DICT['linear']
                self.space_warped[param_name] = warp_func((param_range))
                # assert discrete_method == 'linear'
                # assert warp == 'linear', 'type=int, configspace:{} must be "linear"'.format(warp)
                self.all_types[param_name] = 'int'
                return CSH.UniformIntegerHyperparameter(
                    name=param_name,
                    lower=self.space_warped[param_name][0],
                    upper=self.space_warped[param_name][1],
                    log=warp=='log'
                )
            else:
                pass
        else:
            raise ValueError('param: type=%s not in {"cat", "ord", "con"}')

    def unwarp_space(self, param_name):

        warp = self.all_warp[param_name]
        unwarp_func = UNWARP_DICT[warp]

        return unwarp_func(self.space_warped[param_name])

    def warp(self, xx): # TODO
        '''
        xx： a dict, valid
        return: [dict]
        '''
        # xx_warped_list = []
        # for xx in xx_list:
        xx_warped = deepcopy(xx)
        for param_key in self.all_warp:
            warp_func = WARP_DICT[self.all_warp[param_key]]
            xx_warped[param_key] = warp_func(xx_warped[param_key])
            # discrete_method, param_values = self.all_round[param_key]
            # if not (param_values is None):
            #
            #     xx_warped[param_key] = np.searchsorted(param_values, xx_warped[param_key])
            # xx_warped_list.append(xx_warped)
        return xx_warped

    def unwrap(self, xx_warped): # TODO
        '''
        xx_warped: a dict
        '''
        # xx_list = []
        xx = deepcopy(xx_warped)
        for param_key in self.all_warp:
            unwarp_func = UNWARP_DICT[self.all_warp[param_key]]
            xx[param_key] = unwarp_func(xx[param_key])
            # xx_list.append(xx)
            # TODO
            # discrete_method, param_values = self.all_round[param_key]

            # round_func = ROUND_DICT[discrete_method]
            # xx_rounded = round_func(xx[param_key])
            # if discrete_method != 'linear':
            #     round_func = ROUND_DICT[discrete_method]
            #     xx_rounded = round_func(xx[param_key])
            # else:
            #     xx_rounded = xx[param_key]
            # if param_values is None:
            #     xx[param_key] = xx_rounded
            # else:
            #     idx = np.clip(xx_rounded, 0, len(param_values) - 1)
            #     xx[param_key] = param_values[idx] # TODO round索引类型
        return xx

    def warp_s(self, xx_list): # TODO
        '''
        xx_list: [:dict: hp]
        return: [dict]
        '''
        xx_warped_list = []
        for xx in xx_list:
            # xx_warped = deepcopy(xx)
            # for param_key in self.all_warp:
            #     warp_func = WARP_DICT[self.all_warp[param_key]]
            #     xx_warped[param_key] = warp_func(xx_warped[param_key])
            xx_warped_list.append(self.warp(xx))
        return xx_warped_list

    def unwrap_s(self, xx_warped_list): # TODO
        xx_list = []
        for xx_w in xx_warped_list:
            # xx = deepcopy(xx_w)
            # for param_key in self.all_warp:
            #     unwarp_func = UNWARP_DICT[self.all_warp[param_key]]
            #     xx[param_key] = unwarp_func(xx[param_key])
            xx_list.append(self.unwrap(xx_w))
        return xx_list

