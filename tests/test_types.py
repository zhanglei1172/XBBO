import numpy as np
import pytest
import ConfigSpace as CS

from xbbo.configspace.space import Space, Configurations
from xbbo.configspace.warp import Warp


@pytest.fixture
def seed():
    return 8888


@pytest.fixture
def config_space(seed):

    cs = CS.ConfigurationSpace(seed=seed)
    warp = Warp()
    cs.add_hyperparameter(warp.warp_space('int',"n_units_1", param_range=[0, 5]))
    cs.add_hyperparameter(warp.warp_space('int', "n_units_2", param_range=[0, 5]))
    cs.add_hyperparameter(warp.warp_space('float',"dropout_1", param_range=[0, 0.9]))
    cs.add_hyperparameter(warp.warp_space('float',"dropout_2", param_range=[0, 0.9]))
    cs.add_hyperparameter(warp.warp_space('cat', "activation_fn_1", param_values=["tanh", "relu"])) # [relu, tanh]
    cs.add_hyperparameter(warp.warp_space('cat', "activation_fn_2", param_values=["tanh", "relu"]))
    cs.add_hyperparameter(
        warp.warp_space('int', "init_lr", param_range=[0, 5]))
    cs.add_hyperparameter(warp.warp_space('cat', "lr_schedule", param_values=["cosine", "const"]))
    cs.add_hyperparameter(warp.warp_space('int', "batch_size", param_range=[0, 3]))

    return cs, warp


def test_shapes(config_space, seed):

    SPARSE_DIM = 9
    DENSE_DIM = 12

    myspace = Space(*config_space, seed=seed)

    assert myspace.get_dimensions(sparse=True) == SPARSE_DIM
    assert myspace.get_dimensions(sparse=False) == DENSE_DIM

    bounds = myspace.get_bounds()

    np.testing.assert_array_equal(bounds.lb, np.zeros(DENSE_DIM))
    np.testing.assert_array_equal(bounds.ub, np.ones(DENSE_DIM))

    assert isinstance(myspace.sample_configuration()[0], CS.Configuration)
    assert isinstance(myspace.sample_configuration(size=1)[0], CS.Configuration)

    configs = myspace.sample_configuration(size=5)

    assert len(configs) == 5

    for config in configs:
        assert isinstance(config, CS.Configuration)
        assert config.get_array().shape == (SPARSE_DIM,)


def test_dense_encoding(config_space, seed):

    ind = 0

    myspace = Space(*config_space, seed=seed)

    name = myspace.get_hyperparameter_by_idx(ind)
    # hp = cs_dense.get_hyperparameter(name)
    assert name == "activation_fn_1"

    dct = {
        'activation_fn_1': 'relu',
        'activation_fn_2': 'tanh',
        'batch_size': 2,
        'dropout_1': 0.39803953082292726,
        'dropout_2': 0.022039062686389176,
        'init_lr': 0,
        'lr_schedule': 'cosine',
        'n_units_1': 5,
        'n_units_2': 1
    }
    dict_warped = myspace.warp.warp(dct)
    config = Configurations(myspace, warped_values=dict_warped)
    array = config.get_array()

    # # should always be between 0 and 1
    # assert np.less_equal(0., array).all()
    # assert np.less_equal(array, 1.).all()

    np.testing.assert_array_almost_equal(array, [0., 1., 0.62500063, 0.44226615, 0.02448785, 0.08333194, 1., 0.91666806, 0.24999917])

    # make sure we recover original dictionary exactly
    config_recon = Configurations.from_array(myspace, array)
    dct_recon = config_recon.get_dict_unwarped()

    assert dct == dct_recon

    # # in one-hot encoding scheme, if all entries are "hot" (i.e. nonzero),
    # # we should take the argmax
    # array[0] = 0.8  # tanh
    # array[1] = 0.6  # relu
    #
    # config_recon = Configurations.from_array(myspace, array)
    # dct_recon = config_recon.get_dict_unwarped()
    #
    # assert dct_recon["activation_fn_1"] == "relu"
