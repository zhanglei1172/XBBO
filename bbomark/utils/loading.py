import json
import os
import pickle
import shelve

from bbomark.constants import OPTIMIZERS_FILE, ARG_DELIM
from bbomark.optimizers.config import CONFIG
from bbomark.utils import cmd_parse as cmd
from bbomark.utils.cmd_parse import infer_settings
from bbomark.utils.path_util import absopen


def load_optimizer_kwargs(optimizer_name, opt_root):  # pragma: io
    """Load the kwarg options for this optimizer being tested.

    This is part of the general experiment setup before a study.

    Parameters
    ----------
    optimizer_name : str
        Name of the optimizer being tested. This optimizer name must be present in optimizer config file.
    opt_root : str
        Absolute path to folder containing the config file.

    Returns
    -------
    kwargs : dict(str, object)
        The kwargs setting to pass into the optimizer wrapper constructor.
    """
    if optimizer_name in CONFIG:
        _, kwargs = CONFIG[optimizer_name]
    else:
        settings = load_optimizer_settings(opt_root)
        assert optimizer_name in settings, "optimizer %s not found in settings file %s" % optimizer_name
        _, kwargs = settings[optimizer_name]
    return kwargs

def load_history(filename):
    with open(filename, 'rb') as db:
        dct = pickle.load(db)
        if 'features' in dct:
            return dct['features'], dct['y'], True
        else:
            return dct['params'], dct['y'], False
    # with shelve.open(filename) as db:
    #     if 'features' in db:
    #         return db['features'], db['y'], True
    #     else:
    #         return db['params'], db['y'], False

def save_history(filename, dct):
    with open(filename, 'wb') as db:
        pickle.dump(dct, db)
    # with shelve.open(filename) as db:
    #     for k in dct:
    #         db[k] = dct[k]

def load_optimizer_settings(opt_root):
    try:
        with absopen(os.path.join(opt_root, OPTIMIZERS_FILE), "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        # Search for optimizers instead
        settings = infer_settings(opt_root)

    assert isinstance(settings, dict)
    assert not any((ARG_DELIM in opt) for opt in settings), "optimizer names violates name convention"
    return settings