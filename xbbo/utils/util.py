from typing import Optional
import numpy as np
import pickle, os, json
from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         OrdinalHyperparameter, Constant,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)


def dumpOBJ(path, filename, obj):
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(obj, f)

def loadOBJ(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def dumpJson(path, filename, j):
    with open(os.path.join(path, filename), 'w') as f:
        json.dump(j, f)

def loadJson(filename):
    with open(filename, 'r') as f:
        j = json.load(f)
    return j

def get_types(config_space, instance_features=None):
    """TODO"""
    # Extract types vector for rf from config space and the bounds
    types = [0] * len(config_space.get_hyperparameters())
    bounds = [(np.nan, np.nan)] * len(types)

    for i, param in enumerate(config_space.get_hyperparameters()):
        parents = config_space.get_parents_of(param.name)
        if len(parents) == 0:
            can_be_inactive = False
        else:
            can_be_inactive = True

        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            if can_be_inactive:
                n_cats = len(param.choices) + 1
            types[i] = n_cats
            bounds[i] = (int(n_cats), np.nan)

        elif isinstance(param, (OrdinalHyperparameter)):
            n_cats = len(param.sequence)
            types[i] = 0
            if can_be_inactive:
                bounds[i] = (0, int(n_cats))
            else:
                bounds[i] = (0, int(n_cats) - 1)

        elif isinstance(param, Constant):
            # for constants we simply set types to 0 which makes it a numerical
            # parameter
            # pass
            if can_be_inactive:
                bounds[i] = (2, np.nan)
                types[i] = 2
            else:
                bounds[i] = (0, np.nan)
                types[i] = 0
            # and we leave the bounds to be 0 for now
        elif isinstance(param, UniformFloatHyperparameter):
            # Are sampled on the unit hypercube thus the bounds
            # are always 0.0, 1.0
            if can_be_inactive:
                bounds[i] = (-1.0, 1.0)
            else:
                bounds[i] = (0, 1.0)
        elif isinstance(param, UniformIntegerHyperparameter):
            if can_be_inactive:
                bounds[i] = (-1.0, 1.0)
            else:
                bounds[i] = (0, 1.0)
        elif not isinstance(
                param,
            (UniformFloatHyperparameter, UniformIntegerHyperparameter,
             OrdinalHyperparameter, CategoricalHyperparameter)):
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    if instance_features is not None:
        types = types + [0] * instance_features.shape[1]

    types = np.array(types, dtype=np.uint)
    bounds = np.array(bounds, dtype=object)
    return types, bounds

def create_rng(rng):
    """ ref: hpolib
    helper to create rng from RandomState or int
    :param rng: int or RandomState
    :return: RandomState
    """
    if rng is None:
        return np.random.RandomState()
    elif type(rng) == np.random.RandomState:
        return rng
    elif int(rng) == rng:
        # As seed is sometimes -1 (e.g. if SMAC optimizes a
        # deterministic function
        rng = np.abs(rng)
        return np.random.RandomState(rng)
    else:
        raise ValueError("%s is neither a number nor a RandomState. "
                         "Initializing RandomState failed")
        
def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    os.system('wget {} -O {} && unzip -d {} {}'.format(url,filename, extract_root, filename))
    if remove_finished:
        os.system('rm {}'.format(filename))
