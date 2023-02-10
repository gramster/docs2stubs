import pytest
from docs2stubs.type_normalizer import check_normalizer
from hamcrest import assert_that, equal_to, has_entries


def tcheck(input: str, expected_type: str, expected_imports: dict|None=None, \
        modname: str|None=None, is_param: bool=False):
      trivial, type, imports = check_normalizer(input, is_param, modname)
      assert_that(trivial)
      assert_that(type, equal_to(expected_type))
      if expected_imports is None:
          expected_imports = {}
      assert_that(imports, has_entries(expected_imports))

      
def ntcheck(input: str, expected_type: str, expected_imports: dict|None=None, \
        modname: str|None=None, is_param: bool=False):
      trivial, type, imports = check_normalizer(input, is_param, modname)
      assert_that(not trivial)
      assert_that(type, equal_to(expected_type))
      if expected_imports is None:
          expected_imports = {}
      assert_that(imports,  has_entries(expected_imports))

      
def test_simple_normalizations():
    for typ in ['int', 'float', 'complex', 'bool', 'str', 'set', 'frozenset', 'dict', 'list', 'None']:
        tcheck(typ, typ, {})
    tcheck("Any", "Any", {'typing': ['Any']})
    tcheck('object', 'Any', {'typing': ['Any']})
    tcheck('array', "np.ndarray", {'numpy': []})
    tcheck('array', "ArrayLike", {'numpy.typing': ['ArrayLike']}, is_param=True)
    ntcheck("function", "Callable", {'typing': ['Callable']}, is_param=True)
    tcheck("list of tuple", "list[tuple]", None)
    ntcheck("slice", "slice", None)
    ntcheck("None, `numpy.ndarray`", "np.ndarray|None", {'numpy':[]})


def test_restricted_values():
    tcheck("{'lar', 'lasso'}", "Literal['lar','lasso']",  {'typing': ['Literal']})
    tcheck("{'linear', 'poly'} or callable", "Literal['linear','poly']|Callable", {'typing': ['Callable', 'Literal']})
    tcheck("float or {'scale', 'auto'}", "float|Literal['scale','auto']", {'typing': ['Literal']})
    tcheck("'all' or 'wcs'", "Literal['all','wcs']", {'typing': ['Literal']})


def test_unions():
    tcheck('int, or str', 'int|str', None)
    tcheck('str or callable', "str|Callable", {'typing': ['Callable']})
    tcheck('float or None', 'float|None', None)
    tcheck('scalar or array-like', "Scalar|ArrayLike", {'._typing': ['Scalar'], 'numpy.typing': ['ArrayLike']})
    tcheck('str or path-like or file-like', 'str|PathLike|IO', {'os':['PathLike'], 'typing': ['IO']})
    ntcheck('float or array-like, shape (n, )', 'float|ArrayLike', {'numpy.typing': ['ArrayLike']})
    tcheck('array-like or scalar', 'ArrayLike|Scalar', {'numpy.typing': ['ArrayLike'], '._typing': ['Scalar']})


def test_array_likes():
    tcheck('array-like', 'ArrayLike', {'numpy.typing': ['ArrayLike']})
    tcheck('ndarray', 'np.ndarray', {'numpy': []})
    tcheck('array of shape (n_samples, n_dimensions)','np.ndarray', {'numpy': []})
    tcheck('array-like of shape (n_samples,)', 'ArrayLike', {'numpy.typing': ['ArrayLike']})
    ntcheck('array-like, shape (n_samples,)', 'ArrayLike', {'numpy.typing': ['ArrayLike']})
    tcheck('ndarray of shape (n_samples,)', 'np.ndarray', {'numpy': []})
    tcheck('None or array of shape (n_samples, n_classes)', 'np.ndarray|None', {'numpy': []})
    tcheck('int or float or ndarray of shape (n_samples,)', 'int|float|np.ndarray', {'numpy': []})


def test_tuples():
    tcheck('tuple of float', 'tuple[float, ...]', None)
    ntcheck('tuple of float or int', 'tuple[float, ...]|int', None)
    tcheck('tuple of 2 float', 'tuple[float,float]', None)
    tcheck('tuple (float, int)', 'tuple[float,int]', None)
    tcheck("tuple of ints", "tuple[int, ...]", None)
    ntcheck("(float, float, int)", "tuple[float,float,int]", None)


def test_lists():
    tcheck('dict or list of dictionaries', 'dict|list[dict]', None)
    tcheck('int or list of floats', 'int|list[float]', None)
    # Should probably distinguish between param or return type
    # and keep list if return type. For now we can just use list[]
    # and in the location it gets used we can replace that with Sequence 
    # if appropriate.
    tcheck('list of ndarray', 'list[np.ndarray]', {'numpy': []})

    # Ambiguous so not trivial
    ntcheck('list of str or bool', 'list[str]|bool', None)


def test_classes():
    tcheck(':class:`~sklearn.utils.Bunch`', 'Bunch', {'sklearn.utils': ['Bunch']}, 'sklearn')
    ntcheck('list of `~matplotlib.axes.Axes`', 'list[Axes]', {'matplotlib.axes': ['Axes']}, 'matplotlib')

    
