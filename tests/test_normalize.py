import pytest
from docs2stubs.normalize import check_normalizer
from hamcrest import assert_that, equal_to


def tcheck(input: str, output: str):
      assert_that(check_normalizer(input), equal_to(f'(Trivial) {output}'))    


def ntcheck(input: str, output: str):
      assert_that(check_normalizer(input), equal_to(f'(Non-trivial) {output}'))  


def test_simple_normalizations():
    for typ in ['int', 'float', 'complex', 'bool', 'str', 'set', 'frozenset', 'range', 'dict', 'list', 'None']:
        tcheck(typ, typ)
    tcheck('object', 'Any')
    tcheck('array', 'ArrayLike')


def test_restricted_values():
    tcheck("{'lar', 'lasso'}", "Literal['lar', 'lasso']")
    tcheck("{'linear', 'poly'} or callable", "Literal['linear', 'poly']|Callable")
    tcheck("float or {'scale', 'auto'}", "float|Literal['scale', 'auto']")


def test_unions():
    tcheck('int, or str', 'int|str')
    tcheck('str or callable', 'str|Callable')
    tcheck('float or None', 'float|None')
    tcheck('scalar or array-like', 'Scalar|ArrayLike')
    tcheck('str or path-like or file-like', 'str|PathLike|FileLike')
    tcheck('float or array-like, shape (n, )', 'float|ArrayLike')
    tcheck('array-like or scalar', 'ArrayLike|Scalar')


def test_array_likes():
    tcheck('array-like', 'ArrayLike')
    tcheck('ndarray', 'np.ndarray')
    tcheck('array of shape (n_samples, n_dimensions)','ArrayLike')
    tcheck('array-like of shape (n_samples,)', 'ArrayLike')
    tcheck('array-like, shape (n_samples,)', 'ArrayLike')
    tcheck('ndarray of shape (n_samples,)', 'np.ndarray')
    tcheck('None or array of shape (n_samples, n_classes)', 'None|ArrayLike')

    tcheck('int or float or ndarray of shape (n_samples,)', 'int|float|np.ndarray')


def test_tuples():
    tcheck('tuple of float', 'tuple[float, ...]')


def test_lists():
    tcheck('dict or list of dictionaries', 'dict|Sequence[dict]')
    tcheck('int or list of floats', 'int|Sequence[float]')
    # Should probably distinguish between param or return type
    # and keep list if return type
    tcheck('list of ndarray', 'Sequence[np.ndarray]')

    # Ambiguous so not trivial
    ntcheck('list of str or bool', 'Sequence[str]|bool')


def test_classes():
    tcheck(':class:`~sklearn.utils.Bunch`', 'Bunch')
    tcheck('list of `~matplotlib.axes.Axes`', 'Sequence[Axes]')