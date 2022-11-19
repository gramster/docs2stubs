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
        modname: str = "sklearn"):
      trivial, type, imports = check_normalizer(input, modname)
      assert_that(not trivial)
      assert_that(type, equal_to(expected_type))
      assert_that(imports, equal_to(expected_imports))

  
def test_basic_type():
    ntcheck("float in [0., 1.]", "float", None)

    
def test_array_types():
    ntcheck("1d boolean nd-array", "NDArray[bool]", {'numpy.typing': ['NDArray']})

    ntcheck("{array, sparse matrix}", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix}", "NDArray", {'numpy.typing': ['NDArray']})

    ntcheck("array-like or sparse matrix of  shape (n_samples, sum_n_components)", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("array-like, shape [n_samples, n_components]", "ArrayLike", {'numpy.typing': ['ArrayLike']})
    ntcheck("array of shape=(n_features, n_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("array, shape=(n_features, n_features)", "NDArray", {'numpy.typing': ['NDArray']})

    tcheck("float or ndarray of floats", "float|NDArray[float]", {'numpy.typing': ['NDArray']})
    ntcheck("float or ndarray of shape (n_unique_labels,), dtype=np.float64", "float|NDArray[np.float64]", {'numpy.typing': ['NDArray']})
    ntcheck("float or array of shape [n_classes]", "float|NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("float or array of float, shape = [n_unique_labels]", "float|NDArray[float]", {'numpy.typing': ['NDArray']})
    tcheck("int or ndarray of int", "int|NDArray[int]", {'numpy.typing': ['NDArray']})

    ntcheck("list of 1d ndarrays", "list[NDArray]", {'numpy.typing': ['NDArray']})
    ntcheck("list of shape (n_alphas,), dtype=float", "list[float]", None)
    ntcheck("list of {ndarray, sparse matrix, dataframe} or None", "list[NDArray|dataframe]|None", {'numpy.typing': ['NDArray']})
    ntcheck("list of int of size n_outputs", "list[int]", None)
    ntcheck("list, length n", "list", None)
    ntcheck("list of shape (n_alphas,) of ndarray of shape    (n_features, n_features)", "list[NDArray]", {'numpy.typing': ['NDArray']})

    tcheck("ndarray of str objects", "NDArray[str]", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples,) of arrays", "NDArray[ArrayLike]", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("ndarray of shape coef.shape", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape = (n_thresholds,)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray, dtype=np.intp", "NDArray[np.intp]", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples, n_components) or None", "NDArray|None", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of str or `None`", "NDArray[str]|None", {'numpy.typing': ['NDArray']})    
    ntcheck("ndarray of shape (n_samples, n_components),", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_components, n_features) or None", "NDArray|None", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (2, 2), dtype=np.int64", "NDArray[np.int64]", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape of (n_samples, k)", "NDArray", {'numpy.typing': ['NDArray']})
    tcheck("ndarray of int", "NDArray[int]", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int", "NDArray[int]", {'numpy.typing': ['NDArray']})

    ntcheck("np.array", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("numpy.ndarray", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("numpy array of shape [n_samples]", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("numpy scalar or array of shape (n_classes,)", "Scalar|NDArray", {'numpy.typing': ['NDArray'], 'Scalar': ['sklearn._typing']})
 
    ntcheck("sequence of array-like of shape (n_samples,) or  (n_samples, n_outputs)", "Sequence[ArrayLike]", {'typing': ['Sequence'], 'numpy.typing': ['ArrayLike']})
    ntcheck("sparse matrix of shape (n_samples_transform, n_samples_fit)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("sparse-matrix of shape (n_queries, n_samples_fit)", "NDArray", {'numpy.typing': ['NDArray']})  
    ntcheck("strided ndarray", "NDArray", {'numpy.typing': ['NDArray']})


def test_array_like():
   tcheck("array-like or tuple of array-like", "ArrayLike|tuple[ArrayLike, ...]", {'numpy.typing': ['ArrayLike']})  


def test_class_types():
    tcheck("ColumnTransformer", "ColumnTransformer", {'sklearn.compose._column_transformer': ['ColumnTransformer']})       
    ntcheck("_CalibratedClassifier instance", "_CalibratedClassifier", {'sklearn.calibration': ['_CalibratedClassifier']})
    ntcheck("a cross-validator instance.", "cross-validator", None)
    ntcheck(":class:`numpy:numpy.random.RandomState`", "RandomState", {'numpy.random': ['RandomState']})

       
def test_dict_types():
   ntcheck("mapping of string to any", "Mapping[str, Any]", {'typing': ['Mapping'], 'Any': ['typing']})

   
def test_restricted_value_types():
    tcheck("{'int', 'str', 'bool', None}", "None|Literal['int','str','bool']", {'typing': ['Literal']})
    ntcheck("one of {'multilabel-indicator', 'multiclass', 'binary'}", "Literal['multilabel-indicator','multiclass','binary']", {'typing': ['Literal']})


def test_iterable_types():
    ntcheck("iterator over dict of str to any", "Iterator[dict[str, Any]]", {'Iterator': ['collections.abc'], 'Any': ['typing']})
    ntcheck("generator of ndarray of shape (n_samples, k)", "Generator[NDArray, None, None]", {'numpy.typing': ['NDArray'], 'Generator': ['collections.abc']})
    ntcheck("iter(Matrices)", "Iterator[Matrices]", {'Iterator': ['collections.abc']})
   
def test_tuple_types():
    ntcheck("list of (objective, dual_gap) pairs", "list[tuple[objective,dual_gap]]", None)

    
def test_scikit_returns():
    ntcheck("returns an instance of self.", "self", None)
    ntcheck("array of integers, shape: n_samples", "NDArray[int]|n_samples", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_bins,) or smaller", "NDArray|smaller", {'numpy.typing': ['NDArray']})
    ntcheck("(doc_topic_distr, suff_stats)", "tuple[doc_topic_distr,suff_stats]", None)
    ntcheck("ndarray of shape (n_samples, n_classes) or list of such arrays", "NDArray|list[ArrayLike]", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape                 (n_samples, n_encoded_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("mask", "mask", None)
    ntcheck("ndarray of shape (n_features,) or (n_samples,), dtype=floating", "NDArray[floating]", {'numpy.typing': ['NDArray']})
    ntcheck("array-like, sparse matrix or list", "ArrayLike|NDArray|list", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("list of size n_outputs of ndarray of size (n_classes,)", "list[NDArray]", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape (n_samples, n_features)         or (n_samples, n_features_with_missing)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_features, n_alphas) or             (n_targets, n_features, n_alphas)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{array-like, sparse matrix}", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape (n_components, n_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("json", "json", None)
    ntcheck("ndarray of shape `shape`", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{array-like, sparse matrix} of                 shape (n_samples, sum_n_components)", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("arrays of shape (n_samples,) or (n_samples, n_classes)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("sparse matrix of (n_samples, n_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("sparse matrix of shape (n_samples, n_nodes)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples, n_classes), or a list of such arrays", "NDArray|list[ArrayLike]", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("sparse matrix of shape (n_samples, n_out)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{array-like, sparse matrix} of shape (n_samples_X, n_features)", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("{array-like, sparse matrix} of shape (n_samples_Y, n_features)", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("ndarray of shape (n_queries,) or (n_queries, n_outputs),                 dtype=double", "NDArray[double]", {'numpy.typing': ['NDArray']})
    ntcheck("array of the same shape as ``dist``", "NDArray[the]", {'numpy.typing': ['NDArray']})
    ntcheck("{sparse matrix, array-like}, (n_samples, n_samples)", "NDArray|ArrayLike|tuple[n_samples,n_samples]", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("X, Y, x_mean, y_mean, x_std, y_std", "X|Y|x_mean|y_mean|x_std|y_std", None)
    ntcheck("{ndarray, sparse matrix} or tuple of these", "NDArray|tuple[these, ...]", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape (n_samples,)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("sparse matrix of shape (n_samples, n_classes)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape (n_samples, NP)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_knots, n_features), dtype=np.float64", "NDArray[np.float64]", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_infrequent_categories,) or None", "NDArray|None", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape (n_samples, n_features + 1)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of (n_samples, n_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("array, (n_components, n_features)", "NDArray|tuple[n_components,n_features]", {'numpy.typing': ['NDArray']})
    ntcheck(":class:`~sklearn.utils.Bunch` or dict of such instances", "Bunch|dict[instances, instances]", {'sklearn.utils': ['Bunch']})
    ntcheck("array of shape [n_samples]", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("array, shape = [n_samples, n_classes] or [n_samples]", "NDArray|tuple[n_samples]", {'numpy.typing': ['NDArray']})
    ntcheck("array of shape [n_samples, n_selected_features]", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("array of shape [n_samples, n_original_features]", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("sparse matrix.", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("numbers.Number", "Number", {'numbers': ['Number']})
    tcheck("ndarray or None", "NDArray|None", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_features,) or (n_samples,), dtype=integral", "NDArray[integral]", {'numpy.typing': ['NDArray']})
    ntcheck("boolean array", "NDArray[bool]", {'numpy.typing': ['NDArray']})
    ntcheck("result", "result", None)
    ntcheck("converted_container", "converted_container", None)
    ntcheck("X, y", "X|y", None)
    ntcheck("`pytest.mark.parametrize`", "parametrize", {'pytest.mark': ['parametrize']})
    ntcheck("generator", "Generator", {'Generator': ['collections.abc']})
    tcheck("ndarray of float", "NDArray[float]", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape                 (n_samples, n_features_out)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape         (n_samples, n_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape (n_samples,         n_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("array, shape = [n_samples] or [n_samples, n_targets]", "NDArray|tuple[n_samples,n_targets]", {'numpy.typing': ['NDArray']})
    ntcheck("{array-like, sparse matrix} of shape (n_samples, n_features)", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("ndarray of shape (n_samples,) or             (n_samples, n_classes)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape = (n_features,)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("intercept_decay", "intercept_decay", None)
    ntcheck("{ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{array-like, sparse matrix} of shape (n_samples, n_outputs)", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("sparse matrix of shape (n_components, n_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{ndarray, sparse matrix} of shape (n_samples, n_components)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("stream", "stream", None)
    ntcheck("np.array or scipy.sparse.csr_matrix", "NDArray|csr_matrix", {'numpy.typing': ['NDArray'], 'scipy.sparse': ['csr_matrix']})
    ntcheck("np.ndarray", "ndarray", {'np': ['ndarray']})
    ntcheck("1-D arrays", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_features, n_components) or             (n_components, n_features)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("sparse matrix of shape (dim, dim)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("3D array", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples, n_samples) or                 (n_samples, n_samples, n_targets)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples_X, n_samples), or             (n_samples_X, n_targets, n_samples)", "NDArray|tuple[n_samples_X,n_targets,n_samples]", {'numpy.typing': ['NDArray']})
    ntcheck("(sparse) array-like of shape (n_samples,) or (n_samples, n_classes)", "ArrayLike", {'numpy.typing': ['ArrayLike']})
    ntcheck("(sparse) array-like of shape (n_samples, n_classes)", "ArrayLike", {'numpy.typing': ['ArrayLike']})
    ntcheck("C-contiguous array of shape (n_samples,) or array of shape             (n_samples, n_classes)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("C-contiguous array of shape (n_samples,), array of shape", "NDArray|NDArray[shape]", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray or a sparse matrix class", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("np.ndarray or a sparse matrix class", "ndarray|NDArray", {'numpy.typing': ['NDArray'], 'np': ['ndarray']})
    ntcheck("array of shape (n_patches, patch_height, patch_width) or         (n_patches, patch_height, patch_width, n_channels)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape image_size", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("array of shape (n_patches, patch_height, patch_width) or              (n_patches, patch_height, patch_width, n_channels)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("list of arrays of shape (n_samples,)", "list[ArrayLike]", {'numpy.typing': ['ArrayLike']})
    ntcheck("list of dict_type objects of shape (n_samples,)", "list[dict_type]", None)
    ntcheck("list of length (n_features,)", "list", None)
    ntcheck("module", "module", None)
    ntcheck("{array-like, sparse matrix} of shape (n_samples, n_clusters)", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("sparse matrix", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_nodes,) or None", "NDArray|None", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_nodes, ) or None", "NDArray|None", {'numpy.typing': ['NDArray']})
    ntcheck("array [n_samples]", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples, n_estimators) or                 (n_samples, n_classes * n_estimators)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("A TreePredictor object.", "TreePredictor", {'sklearn.ensemble._hist_gradient_boosting.predictor': ['TreePredictor']})
    ntcheck("ndarray, shape (n_samples,) or                 (n_samples, n_trees_per_iteration)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples, n_classes, n_outputs) or                 (n_samples, 1, n_outputs)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples_X * n_samples_Y, n_features) or             (n_samples_X, n_samples_Y)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("ndarray of shape (n_samples_X, n_samples_X) or             (n_samples_X, n_samples_Y)", "NDArray", {'numpy.typing': ['NDArray']})
    ntcheck("{array-like, sparse}, shape=[n_classes_true, n_classes_pred]", "ArrayLike|NDArray", {'numpy.typing': ['ArrayLike', 'NDArray']})
    ntcheck("list of artists", "list[rtists]", None) 