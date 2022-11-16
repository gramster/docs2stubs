import pytest
from docs2stubs.type_normalizer import check_normalizer
from hamcrest import assert_that, equal_to


def tcheck(input: str, expected_type: str, expected_imports: dict|None, \
        modname: str|None=None):
      trivial, type, imports = check_normalizer(input, modname)
      assert_that(trivial)
      assert_that(type, equal_to(expected_type))
      assert_that(imports, equal_to(expected_imports))

      
def ntcheck(input: str, expected_type: str, expected_imports: dict|None, \
        modname: str|None=None):
      trivial, type, imports = check_normalizer(input, modname)
      assert_that(not trivial)
      assert_that(type, equal_to(expected_type))
      assert_that(imports, equal_to(expected_imports))

      
def test_simple_normalizations():
    for typ in ['int', 'float', 'complex', 'bool', 'str', 'set', 'frozenset', 'dict', 'list', 'None']:
        tcheck(typ, typ, None)
    ntcheck("Any", "Any", {'typing': ['Any']})
    tcheck('object', 'Any', {'typing': ['Any']})
    tcheck('array', "NDArray", {'numpy.typing': ['NDArray']})
    tcheck("function", "Callable", {'typing': ['Callable']})
    tcheck("list of tuple", "list[tuple]", None)
    tcheck("slice", "slice", None)
    tcheck("None, `numpy.ndarray`", "NDArray|None", {'numpy.typing': ['NDArray']})


def test_restricted_values():
    tcheck("{'lar', 'lasso'}", "Literal['lar','lasso']",  {'typing': ['Literal']})
    tcheck("{'linear', 'poly'} or callable", "Literal['linear','poly']|Callable", {'typing': ['Callable', 'Literal']})
    tcheck("float or {'scale', 'auto'}", "float|Literal['scale','auto']", {'typing': ['Literal']})
    tcheck("'all' or 'wcs'", "Literal['all','wcs']", None)


def test_unions():
    tcheck('int, or str', 'int|str', None)
    tcheck('str or callable', "str|Callable", {'typing': ['Callable']})
    tcheck('float or None', 'float|None', None)
    tcheck('scalar or array-like', "Scalar|ArrayLike", {'Scalar': ['mod._typing'], 'numpy.typing': ['ArrayLike']})
    tcheck('str or path-like or file-like', 'str|PathLike|IO', {'PathLike': ['os'], 'IO': ['typing']})
    ntcheck('float or array-like, shape (n, )', 'float|ArrayLike', {'numpy.typing': ['ArrayLike']})
    tcheck('array-like or scalar', 'ArrayLike|Scalar', {'numpy.typing': ['ArrayLike'], 'Scalar': ['mod._typing']})


def test_array_likes():
    tcheck('array-like', 'ArrayLike', {'numpy.typing': ['ArrayLike']})
    tcheck('ndarray', 'NDArray', {'numpy.typing': ['NDArray']})
    ntcheck('array of shape (n_samples, n_dimensions)','NDArray', {'numpy.typing': ['NDArray']})
    ntcheck('array-like of shape (n_samples,)', 'ArrayLike', {'numpy.typing': ['ArrayLike']})
    ntcheck('array-like, shape (n_samples,)', 'ArrayLike', {'numpy.typing': ['ArrayLike']})
    ntcheck('ndarray of shape (n_samples,)', 'NDArray', {'numpy.typing': ['NDArray']})
    ntcheck('None or array of shape (n_samples, n_classes)', 'NDArray|None', {'numpy.typing': ['NDArray']})

    ntcheck('int or float or ndarray of shape (n_samples,)', 'int|float|NDArray', {'numpy.typing': ['NDArray']})


def test_tuples():
    tcheck('tuple of float', 'tuple[float, ...]', None)
    ntcheck('tuple of float or int', 'tuple[float, ...]|int', None)
    tcheck('tuple of 2 float', 'tuple[float,float]', None)
    tcheck('tuple (float, int)', 'tuple[float,int]', None)
    tcheck("tuple of ints", "tuple[int, ...]", None)
    tcheck("(float, float, int)", "tuple[float, float, int]", None)


def test_lists():
    tcheck('dict or list of dictionaries', 'dict|list[dict]', None)
    tcheck('int or list of floats', 'int|list[float]', None)
    # Should probably distinguish between param or return type
    # and keep list if return type. For now we can just use list[]
    # and in the location it gets used we can replace that with Sequence 
    # if appropriate.
    tcheck('list of ndarray', 'list[NDArray]', {'numpy.typing': ['NDArray']})

    # Ambiguous so not trivial
    ntcheck('list of str or bool', 'list[str]|bool', None)


def test_classes():
    ntcheck(':class:`~sklearn.utils.Bunch`', 'Bunch', {'sklearn.utils': ['Bunch']}, 'sklearn')
    ntcheck('list of `~matplotlib.axes.Axes`', 'list[Axes]', {'matplotlib.axes': ['Axes']}, 'matplotlib')

    
