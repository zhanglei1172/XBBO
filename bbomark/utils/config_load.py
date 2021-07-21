from bbomark.optimizers.config import CONFIG
from bbomark.utils import cmd_parse as cmd

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
        settings = cmd.load_optimizer_settings(opt_root)
        assert optimizer_name in settings, "optimizer %s not found in settings file %s" % optimizer_name
        _, kwargs = settings[optimizer_name]
    return kwargs