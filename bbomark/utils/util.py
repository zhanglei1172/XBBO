# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General utilities that should arguably be included in Python.
"""
import shlex
import numpy as np

random = np.random.random.__self__
# np seed must be in [0, 2**32 - 1] = [0, uint32 max]
SEED_MAX_INCL = np.iinfo(np.uint32).max

def in_or_none(x, L):
    """Check if item is in list of list is None."""
    return (L is None) or (x in L)


def all_unique(L):
    """Check if all elements in a list are unique.

    Parameters
    ----------
    L : list
        List we would like to check for uniqueness.

    Returns
    -------
    uniq : bool
        True if all elements in `L` are unique.
    """
    uniq = len(L) == len(set(L))
    return uniq


def strict_sorted(L):
    """Return a strictly sorted version of `L`. Therefore, this raises an error if `L` contains duplicates.

    Parameters
    ----------
    L : list
        List we would like to sort.

    Returns
    -------
    S : list
        Strictly sorted version of `L`.
    """
    assert all_unique(L), "Cannot strict sort because list contains duplicates."
    S = sorted(L)
    return S


def range_str(stop):
    """Version of ``range(stop)`` that instead returns strings that are zero padded so the entire iteration is of the
    same length.

    Parameters
    ----------
    stop : int
        Stop value equivalent to ``range(stop)``.

    Yields
    ------
    x : str
        String representation of integer zero padded so all items from this generator have the same ``len(x)``.
    """
    str_len = len(str(stop - 1))  # moot if stop=0

    def map_(x):
        ss = str(x).zfill(str_len)
        return x, ss

    G = map(map_, range(stop))
    return G


def str_join_safe(delim, str_vec, append=False):
    """Version of `str.join` that is guaranteed to be invertible.

    Parameters
    ----------
    delim : str
        Delimiter to join the strings.
    str_vec : list(str)
        List of strings to join. A `ValueError` is raised if `delim` is present in any of these strings.
    append : bool
        If true, assume the first element is already joined and we are appending to it. So, `str_vec[0]` can contain
        `delim`.

    Returns
    -------
    joined_str : str
        Joined version of `str_vec`, which is always recoverable with ``joined_str.split(delim)``.

    Examples
    --------
    Append is required because,

    .. code-block:: pycon

        ss = str_join_safe('_', ('foo', 'bar'))
        str_join_safe('_', (ss, 'baz', 'qux'))

    would fail because we are appending ``'baz'`` and ``'qux'`` to the already joined string ``ss = 'foo_bar'``.

    In this case, we use

    .. code-block:: pycon

        ss = str_join_safe('_', ('foo', 'bar'))
        str_join_safe('_', (ss, 'baz', 'qux'), append=True)
    """
    chk_vec = str_vec[1:] if append else str_vec

    for ss in chk_vec:
        if delim in ss:
            raise ValueError("%s cannot contain delimeter %s" % (ss, delim))

    joined_str = delim.join(str_vec)
    return joined_str


def shell_join(argv, delim=" "):
    """Join strings together in a way that is an inverse of `shlex` shell parsing into `argv`.

    Basically, if the resulting string is passed as a command line argument then `sys.argv` will equal `argv`.

    Parameters
    ----------
    argv : list(str)
        List of arguments to collect into command line string. It will be escaped accordingly.
    delim : str
        Whitespace delimiter to join the strings.

    Returns
    -------
    cmd : str
        Properly escaped and joined command line string.
    """
    vv = [shlex.quote(vv) for vv in argv]
    cmd = delim.join(vv)
    assert shlex.split(cmd) == list(argv)
    return cmd


def chomp(str_val, ext="\n"):
    """Chomp a suffix off a string.

    Parameters
    ----------
    str_val : str
        String we want to chomp off a suffix, e.g., ``"foo.log"``, and we want to chomp the file extension.
    ext : str
        The suffix we want to chomp. An error is raised if `str_val` doesn't end in `ext`.

    Returns
    -------
    chomped : str
        Version of `str_val` with `ext` removed from the end.
    """
    n = len(ext)
    assert n > 0

    chomped, ext_ = str_val[:-n], str_val[-n:]
    assert ext == ext_, "%s must end with %s" % (repr(str_val), repr(ext))
    return chomped


def preimage_func(f, x):
    """Pre-image a funcation at a set of input points.

    Parameters
    ----------
    f : typing.Callable
        The function we would like to pre-image. The output type must be hashable.
    x : typing.Iterable
        Input points we would like to evaluate `f`. `x` must be of a type acceptable by `f`.

    Returns
    -------
    D : dict(object, list(object))
        This dictionary maps the output of `f` to the list of `x` values that produce it.
    """
    D = {}
    for xx in x:
        D.setdefault(f(xx), []).append(xx)
    return D

def isclose_lte(x, y):
    """Check that less than or equal to (lte, ``x <= y``) is approximately true between all elements of `x` and `y`.

    This is similar to :func:`numpy:numpy.allclose` for equality. Shapes of all input variables must be broadcast
    compatible.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        Lower limit in ``<=`` check.
    y : :class:`numpy:numpy.ndarray`
        Upper limit in ``<=`` check.

    Returns
    -------
    lte : bool
        True if ``x <= y`` is approximately true element-wise.
    """
    # Use np.less_equal to ensure always np type consistently
    lte = np.less_equal(x, y) | np.isclose(x, y)
    return lte

def clip_chk(x, lb, ub, allow_nan=False):
    """Clip all element of `x` to be between `lb` and `ub` like :func:`numpy:numpy.clip`, but also check
    :func:`numpy:numpy.isclose`.

    Shapes of all input variables must be broadcast compatible.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        Array containing elements to clip.
    lb : :class:`numpy:numpy.ndarray`
        Lower limit in clip.
    ub : :class:`numpy:numpy.ndarray`
        Upper limit in clip.
    allow_nan : bool
        If true, we allow ``nan`` to be present in `x` without out raising an error.

    Returns
    -------
    x : :class:`numpy:numpy.ndarray`
        An array with the elements of `x`, but where values < `lb` are replaced with `lb`, and those > `ub` with `ub`.
    """
    assert np.all(lb <= ub)  # np.clip does not do this check

    x = np.asarray(x)

    # These are asserts not exceptions since clip_chk most used internally.
    if allow_nan:
        assert np.all(isclose_lte(lb, x) | np.isnan(x))
        assert np.all(isclose_lte(x, ub) | np.isnan(x))
    else:
        assert np.all(isclose_lte(lb, x))
        assert np.all(isclose_lte(x, ub))
    x = np.clip(x, lb, ub)
    return x

def random_seed(random=random):
    """Draw a random seed compatible with :class:`numpy:numpy.random.RandomState`.

    Parameters
    ----------
    random : :class:`numpy:numpy.random.RandomState`
        Random stream to use to draw the random seed.

    Returns
    -------
    seed : int
        Seed for a new random stream in ``[0, 2**32-1)``.
    """
    # np randint is exclusive on the high value, py randint is inclusive. We
    # must use inclusive limit here to work with both. We are missing one
    # possibility here (2**32-1), but I don't think that matters.
    seed = random.randint(0, SEED_MAX_INCL)
    return seed
