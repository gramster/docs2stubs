import pytest
from docs2stubs.type_normalizer import check_normalizer
from hamcrest import assert_that, equal_to
from .test_normalize import tcheck, ntcheck

  
def test_basic_type():
    ntcheck("float in [0., 1.]", "float", None)

def test_sk_params():
    tcheck("array-like of shape (n_samples,) or (n_samples, n_outputs)", "ArrayLike|MatrixLike",
        {'._typing': ['MatrixLike'], 'numpy.typing': ['ArrayLike']}, "sklearn", True)
    tcheck("ndarray of shape (n_samples,)", "ArrayLike",\
        {'numpy.typing': ['ArrayLike']}, "sklearn", True)
    tcheck("array-like of shape (n_samples,)", "ArrayLike", {'numpy.typing': ['ArrayLike']}, "sklearn", True)
    ntcheck("{array-like, sparse matrix} of shape (n_samples, n_features)", "MatrixLike", \
        {'._typing': ['MatrixLike']}, "sklearn", True)
    tcheck("array-like of shape (n_samples, n_features)", "MatrixLike", \
        {'._typing': ['MatrixLike']}, "sklearn", True)
    ntcheck("int, RandomState instance or None", "int|RandomState|None", {}, 'sklearn', True)
    ntcheck("int, RandomState instance", "int|RandomState", {}, 'sklearn', True)
    tcheck("ndarray of shape (n_samples, n_features)", "MatrixLike", \
                {'._typing': ['MatrixLike']}, "sklearn", True)
    tcheck("array-like", "ArrayLike",
                   {'numpy.typing': ['ArrayLike']}, "sklearn", True)
    ntcheck("{array-like, sparse matrix}, shape (n_samples, n_features)", "MatrixLike",
        {'._typing': ['MatrixLike']}, "sklearn", True)
    tcheck("array-like of shape (n_classes,)", "ArrayLike",
                   {'numpy.typing': ['ArrayLike']}, "sklearn", True)
    ntcheck("array-like of str or None", "ArrayLike|None",
                    {'numpy.typing': ['ArrayLike']}, "sklearn", True)
    tcheck("ndarray", "ArrayLike", \
                               {'numpy.typing': ['ArrayLike']}, "sklearn", True)
    tcheck("array of shape (n_samples,)", "ArrayLike", \
                               {'numpy.typing': ['ArrayLike']}, "sklearn", True)
    ntcheck("array-like of shape (n_samples_X, n_features) or list of object", "MatrixLike|ArrayLike", \
                    {'._typing': ['MatrixLike'], 'numpy.typing': ['ArrayLike']}, "sklearn", True)
    tcheck("ndarray of shape (n_samples, K)", "MatrixLike", \
                   {'._typing': ['MatrixLike']}, "sklearn", True)
    tcheck("array-like of shape (n_samples,) or (n_samples, n_targets)", "ArrayLike|MatrixLike", \
                   {'._typing': ['MatrixLike'], 'numpy.typing': ['ArrayLike']}, "sklearn", True)
    ntcheck("array-like, shape (n_samples, n_features)", "MatrixLike", \
                    {'._typing': ['MatrixLike']}, "sklearn", True)
    tcheck("ndarray of shape (n_samples_Y, n_features)", "MatrixLike", \
                    {'._typing': ['MatrixLike']}, "sklearn", True)
    ntcheck("estimator instance", "estimator", {}, "sklearn", True)
    ntcheck("estimator object", "estimator", {}, "sklearn", True)
    ntcheck("dict of string -> object", "Mapping[str, Any]", {'typing': ['Any', 'Mapping']}, "sklearn", True)
    ntcheck("sparse matrix of shape (n_samples, n_features)", "MatrixLike",  \
                    {'._typing': ['MatrixLike']}, "sklearn", True)
"""
10#non-negative float#float
@9#array-like of shape (n_samples, n_components)#MatrixLike
9#int array, shape = [n_samples]#ArrayLike
8#{array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples, n_samples) if metric='precomputed'#MatrixLike
8#'raise' or numeric#Float|Literal['raise']
@8#float or None#Float|None
@8#list of str#Sequence[str]
8#{array-like, sparse matrix}#ArrayLike|MatrixLike
8#array-like, shape (n_samples,)#ArrayLike
@8#{'cyclic', 'random'}#Literal['cyclic','random']
8#array-like of shape (n_samples, n_features) or list of object#ArrayLike|MatrixLike
8#(sparse) array-like of shape (n_samples, n_features)#MatrixLike
8#None or C-contiguous array of shape (n_samples,)#ArrayLike|None
@8#ndarray of shape (n_samples, n_labels)#MatrixLike
7#array-like of shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'#MatrixLike
@7#ndarray of shape (n_components, n_features)#MatrixLike
@7#ndarray of shape (n_samples, n_components)#MatrixLike
!7#int, cross-validation generator or iterable#int, cross-validation generator or iterable
6#indexable, length n_samples#ArrayLike
@6#ndarray of shape (n_classes,)#ArrayLike
6#{ndarray, sparse matrix} of shape (n_samples, n_features)#MatrixLike
@6#float or ndarray of shape (n_samples,)#float|ArrayLike
6#C-contiguous array of shape (n_samples,)#ArrayLike
6#"warn", 0 or 1#Literal["warn"]|int
@5#array-like of shape (n_components, n_features)#ArrayLike
@5#ndarray of shape (n_features,)#ArrayLike
5#{0, 1}#int
5#str or object with the joblib.Memory interface#str|Memory
5#{array-like, dataframe} of shape (n_samples, n_features)#MatrixLike
5#{array-like}, shape (n_samples, n_features)#MatrixLike
5#array-like or label indicator matrix#ArrayLike|MatrixLike
@5#ndarray of shape (n_features, n_features)#MatrixLike
5#{ndarray, sparse matrix}#ArrayLike|MatrixLike
@5#bool or 'allow-nan'#bool|Literal['allow-nan']
5#estimator#Estimator
@5#ndarray of shape (n_samples,) or (n_samples, n_targets)#ArrayLike|MatrixLike
5#str or module#str|module
@5#Kernel#Kernel
5#C-contiguous array of shape (n_samples,) or array of             shape (n_samples, n_classes)#ArrayLike|MatrixLike
5#int > 1 or float between 0 and 1#int|float
5#{array-like, sparse matrix} of shape (n_samples_X, n_features)#MatrixLike
5#{array-like, sparse matrix} of shape (n_samples_Y, n_features)#MatrixLike
@5#array-like of shape (n_samples_X, n_features)#MatrixLike
5#{'raw_values', 'uniform_average'} or array-like of shape             (n_outputs,)#Literal['raw_values','uniform_average']|ArrayLike
4#{array-like, sparse matrix} of shape (n_samples,) or                 (n_samples, n_outputs)#ArrayLike|MatrixLike
@4#{'connectivity', 'distance'}#Literal['connectivity','distance']
@4#{'auto', 'QR', 'LU', 'none'}#Literal['auto','QR','LU','none']
@4#int or 'auto'#Int|int|Literal['auto']
4#estimator object implementing 'fit'#Estimator
@4#array-like of shape (n_samples, n_features) or                 (n_samples, n_samples)#MatrixLike
@4#{'auto', 'arpack', 'dense'}#Literal['auto','arpack','dense']
@4#{'l1', 'l2'}#str|Literal['l1','l2']
@4#array of shape [n_samples, n_features]#MatrixLike
@4#{'cd', 'lars'}#Literal['cd','lars']
@4#float or array-like of shape (n_samples,)#ArrayLike
4#'auto', bool or array-like of shape             (n_features, n_features)#bool|MatrixLike|Literal['auto']
4#ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)#ArrayLike|MatrixLike
4#contiguous array of shape (n_samples,)#ArrayLike
4#None or contiguous array of shape (n_samples,)#None|MatrixLike
@4#array-like of shape (n_samples, n_classes)#MatrixLike
4#array-like, shape (n_samples,) or (n_samples, n_outputs)#ArrayLike|MatrixLike
@4#{'strict', 'ignore', 'replace'}#Literal['strict','ignore','replace']
4#float > 0#Float
4#None or C-contiguous array of shape (n_samples,) or array             of shape (n_samples, n_classes)#ArrayLike|MatrixLike|None
@4#{"gini", "entropy", "log_loss"}#Literal["gini","entropy","log_loss"]
4#{"sqrt", "log2", None}, int or float#Literal["sqrt","log2"]|int|float
4#str or matplotlib Colormap#str|Colormap
@4#{'vertical', 'horizontal'} or float#Literal['vertical','horizontal']|float
@4#{'true', 'pred', 'all'}#Literal['true','pred','all']
@4#array-like of shape (n_samples,), dtype=integral#ArrayLike
@4#{'micro', 'macro', 'samples', 'weighted', 'binary'} or None#Literal['micro','macro','samples','weighted','binary']|None
@3#'auto' or float#float|Literal['auto']|str
@3#{'uniform', 'quantile'}#Literal['uniform','quantile']
@3#ndarray of shape (n_components, n_components)#MatrixLike
@3#{'lasso_lars', 'lasso_cd', 'lars', 'omp',             'threshold'}#Literal['lasso_lars','lasso_cd','lars','omp','threshold']
@3#int or bool#int|bool
@3#{'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}#Literal['random','nndsvd','nndsvda','nndsvdar','custom']
@3#float or {'frobenius', 'kullback-leibler',             'itakura-saito'}#float|Literal['frobenius','kullback-leibler','itakura-saito']
@3#float or "same"#float|Literal["same"]
3#int or RandomState instance#None|int|RandomState
@3#None or array-like of shape (n_samples, n_features)#MatrixLike|None
@3#ndarray of shape (n_samples,) or (n_samples, n_classes)#ArrayLike|MatrixLike
3#iterable of iterables#Iterable[Iterable]
@3#{'C', 'F'}#str|Literal['C','F']
3#list of str of shape (n_features,)#Sequence[str]
3#int or np.nan#int|np.nan
3#iterable or array-like, depending on transformers#Iterable|ArrayLike
@3#array-like of shape (n_samples, n_outputs)#MatrixLike
@3#{'arpack', 'lobpcg', 'amg'}#Literal['arpack','lobpcg','amg']
@3#ndarray of shape (n_samples, n_features) or (n_samples, n_samples)#MatrixLike
3#array-like, shape (n_samples, n_dim)#MatrixLike
@3#BaseEstimator#Estimator
@3#{'auto', 'predict_proba', 'decision_function'}#Literal['auto','predict_proba','decision_function']
@3#tuple of float#tuple[float, ...]
3#Matplotlib axes or array-like of Matplotlib axes#Axes|Sequence[Axes]
@3#float, int or None#float|int|None
@3#ndarray of shape (grid_resolution, grid_resolution)#MatrixLike
3#array-like or sparse matrix, shape (n_samples, n_features)#MatrixLike
3#{array-like or sparse matrix} of shape (n_samples, n_features)#MatrixLike
@3#array-like of shape (n_features, n_features)#MatrixLike
3#ndarray of shape (n_features,), dtype={np.float32, np.float64}#MatrixLike
@3#str, callable#str|Callable
3#numpy array of shape [n_samples, ]#ArrayLike
@3#{'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg',             'sag', 'saga', 'lbfgs'}#Literal['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']
@3#ndarray of shape (n_samples, n_targets)#MatrixLike
3#array-like of shape (n_samples,) default=None#ArrayLike|None
3#{array-like, sparse matrix} of shape (n_samples, n_outputs)#MatrixLike
@3#array-like of shape (n_samples,) or (n_samples, 1)#ArrayLike
@3#{'train', 'test', 'all'}#Literal['train','test','all']
3#kernel instance#Kernel
@3#{'knn', 'rbf'} or callable#Literal['knn','rbf']|Callable
@3#{'filename', 'file', 'content'}#Literal['filename','file','content']
@3#{'ascii', 'unicode'}#Literal['ascii','unicode']
@3#{'english'}, list#Literal['english']|ArrayLike
3#tuple (min_n, max_n)#tuple[float,float]
@3#{'word', 'char', 'char_wb'} or callable#Literal['word','char','char_wb']|Callable
3#{'k-means++', 'random'}, callable or array-like of shape             (n_clusters, n_features)#Literal['k-means++','random']|Callable|MatrixLike
@3#ndarray of shape (n,)#ArrayLike
3#list of (str, estimator) tuples#list[tuple[str, Estimator]]
@3#array-like of shape (n_samples,) or (n_samples, n_classes)#ArrayLike|MatrixLike
@3#array-like of shape (n_samples_Y, n_features)#MatrixLike
3#matplotlib Axes#Axes
3#int array-like of shape (n_samples,)#ArrayLike
2#array-like of shape (n_samples, n_features) or BallTree#MatrixLike|BallTree
@2#{'distance', 'connectivity'}#Literal['distance','connectivity']
@2#array-like of shape (n_samples_transform, n_features)#MatrixLike
@2#array-like of shape (n_samples, n_targets)#MatrixLike
@2#{'sigmoid', 'isotonic'}#Literal['sigmoid','isotonic']
@2#ndarray of shape (n_bins,)#ArrayLike
2#Matplotlib Axes#Axes
@2#{'parallel', 'deflation'}#Literal['parallel','deflation']
@2#str or bool#str|bool
@2#{'logcosh', 'exp', 'cube'} or callable#Literal['logcosh','exp','cube']|Callable
2#{array-like, sparse matrix} of shape (n_samples, n_components)#MatrixLike
@2#{'cd', 'mu'}#Literal['cd','mu']
@2#dict or list of dictionaries#Mapping|Sequence[dict]
2#``'n_samples'`` or str#str
@2#{'exhaust', 'smallest'} or int#Literal['exhaust','smallest']|int
@2#str, callable, or None#str|Callable|None
@2#str, callable, list, tuple, or dict#str|Callable|ArrayLike|tuple|Mapping
2#array-like of shape (n_samples,) or (n_samples, n_outputs) or None#ArrayLike|MatrixLike|None
!2#object type that implements the "fit" and "predict" methods#LinearSVC|object type that implements the "fit" and "predict" methods|SVC
@2#array-like of shape (n_samples, n_output)             or (n_samples,)#ArrayLike|MatrixLike
2#dict of str -> object#Mapping[str, Any]
@2#str, callable, list, tuple or dict#str|Callable|ArrayLike|tuple|Mapping
@2#bool, str, or callable#bool|str|Callable
@2#int, or str#int|str
@2#{'text', 'diagram'}#str|Literal['text','diagram']
2#{ndarray, sparse matrix} of shape (n_samples, n_classes)#MatrixLike
@2#int or array-like of shape (n_features,)#Int|ArrayLike
2#'auto' or a list of array-like#Sequence[ArrayLike]|Literal['auto']
2#tuple (min, max)#tuple[int, int]
2#tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0#tuple[float,float]
@2#{'l1', 'l2', 'max'}#Literal['l1','l2','max']
@2#ndarray of shape (n_samples, n_samples)#MatrixLike
@2#{'uniform', 'normal'}#Literal['uniform','normal']
@2#{'yeo-johnson', 'box-cox'}#Literal['yeo-johnson','box-cox']
@2#{'full', 'tied', 'diag', 'spherical'}#Literal['full','tied','diag','spherical']
@2#{'kmeans', 'k-means++', 'random', 'random_from_data'}#Literal['kmeans','k-means++','random','random_from_data']
@2#array of shape (n_samples, n_dimensions)#MatrixLike
@2#{'euclidean', 'precomputed'}#Literal['euclidean','precomputed']
@2#{'auto', 'FW', 'D'}#Literal['auto','FW','D']
@2#{'auto', 'brute', 'kd_tree', 'ball_tree'}#Literal['auto','brute','kd_tree','ball_tree']
@2#str, or callable#str|Callable
@2#float or 'auto'#float|Literal['auto']
!2#{array-like, NearestNeighbors}#NearestNeighbors|NearestNeighbors|ArrayLike|ArrayLike
@2#{'standard', 'hessian', 'modified', 'ltsa'}#Literal['standard','hessian','modified','ltsa']
@2#{'ovo', 'ovr'}#Literal['ovo','ovr']
2#{array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples_test, n_samples_train)#MatrixLike
2#list of {int, str, pair of int, pair of str}#Sequence[int|str|Sequence[int|str]]
@2#array-like of shape (n_features,), dtype=str#ArrayLike
2#{'average', 'individual', 'both'} or list of such str#Literal['average','individual','both']|Sequence[Literal['average','individual','both']]
@2#{'average', 'individual', 'both'}#Literal['average','individual','both']
@2#{'contourf', 'contour', 'pcolormesh'}#Literal['contourf','contour','pcolormesh']
2#Matplotlib axes#None|Axes
@2#"auto", int or float#int|float|Literal["auto"]
@2#{'auto', bool, array-like}#Literal['auto']|ArrayLike|bool]
2#``Estimator`` instance#Estimator
@2#any#Any
2#tuple, length = n_layers - 2#Sequence[int]
@2#{'identity', 'logistic', 'tanh', 'relu'}#Literal['identity','logistic','tanh','relu']
@2#{'lbfgs', 'sgd', 'adam'}#Literal['lbfgs','sgd','adam']
@2#{'constant', 'invscaling', 'adaptive'}#Literal['constant','invscaling','adaptive']
2#list, length = len(coefs_) + len(intercepts_)#Sequence
@2#array of shape (n_features, n_features)#MatrixLike
2#str, bool or list/tuple of str#Sequence[str]|str|bool
2#'numeric', type, list of type or None#Literal['numeric']|type|Sequence[type]|None
2#str or estimator instance#str|Estimator
2#{ndarray, list, sparse matrix}#ArrayLike|MatrixLike
@2#ndarray of shape (n_samples,) or (n_features,)#ArrayLike
@2#ndarray of shape (n_features,) or (n_samples,), dtype=floating#ArrayLike
@2#array-like of shape (M, N) or (M,)#MatrixLike|ArrayLike
2#array_like#ArrayLike
2#float, optional#float|None
2#bool, optional#bool|None
@2#float or array-like of shape (n_features,)#float|ArrayLike
2#int, float, str, np.nan or None#int|float|str|np.nan|None
@2#float or array-like of shape (n_targets,)#float|ArrayLike
@2#ndarray of shape (n_classes, n_features)#MatrixLike
@2#{'l2', 'l1', 'elasticnet'}#Literal['l2','l1','elasticnet']
2#dict, {class_label: weight} or "balanced"#Mapping[str,float]|Literal["balanced"]
2#numpy array of shape (n_samples,)#ArrayLike
@2#ndarray of shape (n_alphas,)#ArrayLike
2#{array-like, sparse matrix} of shape (n_samples,) or         (n_samples, n_targets)#ArrayLike|MatrixLike
@2#array-like of shape (n_features,) or (n_features, n_targets)#ArrayLike|MatrixLike
@2#ndarray of shape (n_features, )#ArrayLike
@2#bool or array-like of shape (n_features, n_features)#bool|MatrixLike
@2#float or list of float#float|Sequence[float]
@2#{'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}#Literal['newton-cg','lbfgs','liblinear','sag','saga']
2#numpy array of shape [n_samples]#ArrayLike
@2#{'log', 'squared', 'multinomial'}#Literal['log','squared','multinomial']
@2#{'lar', 'lasso'}#Literal['lar','lasso']
@2#bool, 'auto' or array-like #bool|Literal['auto']|ArrayLike
@2#bool, 'auto' or array-like#bool|Literal['auto']|ArrayLike
2#int or None, optional (default=None)#int|None
2#array-like of shape (n_outputs,) or 'random'#ArrayLike|Literal['random']
2#int, RandomState instance or None, optional (default=None)#int|RandomState|None
2#ndarray, float32 or float64#ArrayLike|Float
2#{dense matrix, sparse matrix, LinearOperator}#MatrixLike|LinearOperator
@2#tuple#tuple
2#numpy data type#DType
@2#bool or "auto"#bool|Literal["auto"]
2#tuple of slice#tuple[slice, ...]
2#float, ndarray of shape (n_features,) or None#float|ArrayLike|None
2#int or tuple of shape (2,), dtype=int#int|int|tuple[int,int]
2#{'drop', 'passthrough'} or estimator#Literal['drop','passthrough']|Estimator
2#column dtype or list of column dtypes#DType|Sequence[DType]
2#function#Callable
2#array-like of shape (n_samples_Y, n_features) or list of object#MatrixLike|Sequence
@2#float or ndarray of shape (n_features,)#float|ArrayLike
@2#'fmin_l_bfgs_b' or callable#Literal['fmin_l_bfgs_b']|Callable
@2#array-like of shape (n_kernel_params,)#ArrayLike
2#(sparse) array-like of shape (n_samples,) or (n_samples, n_classes)#ArrayLike|MatrixLike
2#array, shape (n_classes, )#ArrayLike
2#{int, array of shape (n_samples,)}#ArrayLike
2#array, shape (n_samples,)#ArrayLike
@2#array of shape (n_samples, n_classes)#MatrixLike
@2#None or array of shape (n_samples, n_classes)#None|MatrixLike
2#np.ndarray or a sparse matrix class#ArrayLike|MatrixLike
2#tuple of int (patch_height, patch_width)#tuple[int, int]
@2#ndarray of shape [n_samples, n_features]#MatrixLike
2#iterable over raw text documents, length = n_samples#Iterable[str]
2#float in range [0.0, 1.0] or int#float|int
2#Mapping or iterable#Mapping|Iterable
2#Mapping or iterable over Mappings#Mapping|Iterable[Mapping]
2#(ignored)#Any
2#array-like of shape (n_samples,) or float#ArrayLike|float
2#{array-like, sparse matrix} of shape (n_samples, n_features), or                 array-like of shape (n_samples, n_samples)#MatrixLike
@2#{'kmeans', 'discretize', 'cluster_qr'}#Literal['kmeans','discretize','cluster_qr']
2#{array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples, n_samples)#MatrixLike
@2#{"lloyd", "elkan", "auto", "full"}#Literal["lloyd","elkan","auto","full"]
@2#{'randomized', 'arpack'}#Literal['randomized','arpack']
2#{array-like, sparse matrix} of shape (n_samples, n_features), or             (n_samples, n_samples)#MatrixLike
2#float between 0 and 1#float
2#sparse matrix#MatrixLike
@2#array-like or callable#ArrayLike|Callable
2#dict of str -> obj#Mapping[str, obj]
2#list of (str, estimator)#Sequence[tuple[str, Estimator]]
!2#int, cross-validation generator, iterable, or "prefit"#int|cross-validation generator|Iterable|Literal["prefit"]
2#tree.Tree#Tree
2#ndarray of bool of shape (n_features,)#ArrayLike
2#array-like of {bool, int} of shape (n_features)             or shape (n_categorical_features,)#ArrayLike
2#array-like of int of shape (n_features)#ArrayLike
@2#str or callable or None#str|Callable|None
@2#int or float or None#int|float|None
2#ndarray of shape (n_categorical_splits, 8),             dtype=uint32#MatrixLike
2#ndarray, shape (n_samples, n_features)#MatrixLike
2#Estimator#Estimator
@2#{'friedman_mse', 'squared_error', 'mse'}#Literal['friedman_mse','squared_error','mse']
2#estimator or 'zero'#Estimator|Literal['zero']
2#{'auto', 'sqrt', 'log2'}, int or float#Literal['auto','sqrt','log2']|int|float
2#{"balanced", "balanced_subsample"}, dict or list of dicts#Literal["balanced","balanced_subsample"]|Mapping|Sequence[Mapping]
2#{ndarray, sparse matrix} of shape (n_samples, n_labels)#MatrixLike
@2#ndarray of shape (n_samples_X, n_samples_X) or             (n_samples_X, n_features)#MatrixLike
2#{'raw_values', 'uniform_average'}  or array-like of shape             (n_outputs,)#Literal['raw_values','uniform_average']|ArrayLike
2#array-like of shape = (n_samples) or (n_samples, n_outputs)#ArrayLike|MatrixLike
2#{'predict_proba', 'decision_function', 'auto'}                 default='auto'#Literal['predict_proba','decision_function','auto']
2#{'predict_proba', 'decision_function', 'auto'}             default='auto'#Literal['predict_proba','decision_function','auto']
@2#{'predict_proba', 'decision_function', 'auto'}#Literal['predict_proba','decision_function','auto']
2#array-like of shape (n_samples_a, n_samples_a) if metric ==             "precomputed" or (n_samples_a, n_features) otherwise#MatrixLike
2#(rows, columns)#tuple[int,int]
2#{array-like, sparse matrix} of shape (n_samples, n_outputs) or             (n_samples,)#MatrixLike|ArrayLike
2#array, shape = [n_samples]#ArrayLike
2#1d array-like#ArrayLike
@2#{"best", "random"}#Literal["best","random"]
@2#int, float or {"auto", "sqrt", "log2"}#int|float|Literal["auto","sqrt","log2"]
2#dict, list of dict or "balanced"#Mapping|Sequence[Mapping]|Literal["balanced"]
@2#{"random", "best"}#Literal["random","best"]
2#int, float, {"auto", "sqrt", "log2"} or None#int|float|Literal["auto","sqrt","log2"]|None
2#list of str or bool#Sequence[str|bool]
@2#{'all', 'root', 'none'}#Literal['all','root','none']
@1#{'kd_tree', 'ball_tree', 'auto'}#Literal['kd_tree','ball_tree','auto']
@1#{'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear',                  'cosine'}#Literal['gaussian','tophat','epanechnikov','exponential','linear','cosine']
1#array-like, shape (n_queries, n_features),             or (n_queries, n_indexed) if metric == 'precomputed'#MatrixLike
1#array-like of (n_samples, n_features)#MatrixLike
1#{'auto', 'pca', 'lda', 'identity', 'random'} or ndarray of shape             (n_features_a, n_features_b)#Literal['auto','pca','lda','identity','random']|MatrixLike
1#int or numpy.RandomState#int|RandomState
!1#{manual label, 'most_frequent'}#Label|Literal['most_frequent']
@1#{'nipals', 'svd'}#Literal['nipals','svd']
1#int, cross-validation generator, iterable or "prefit"#int|cross-validation generator|Iterable|Literal["prefit"]
1#list of fitted estimator instances#list[IsotonicRegression|_SigmoidCalibration]|list of fitted estimator instances
@1#{'batch', 'online'}#Literal['batch','online']
@1#ndarray of shape (n_components, n_samples)#MatrixLike
@1#{'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}#Literal['lasso_lars','lasso_cd','lars','omp','threshold']
1#tuple of (A, B) ndarrays#tuple[ArrayLike, ArrayLike]
@1#{'arpack', 'randomized'}#Literal['arpack','randomized']
@1#int, float or 'mle'#Int|float|Literal['mle']
@1#{'auto', 'full', 'arpack', 'randomized'}#Literal['auto','full','arpack','randomized']
@1#{'linear', 'poly',             'rbf', 'sigmoid', 'cosine', 'precomputed'}#Literal['linear','poly','rbf','sigmoid','cosine','precomputed']
@1#{'auto', 'dense', 'arpack', 'randomized'}#Literal['auto','dense','arpack','randomized']
1#int >= 0, or 'auto'#int|Literal['auto']
@1#{'both', 'components', 'transformation'}#Literal['both','components','transformation']
@1#{'both', 'components', 'transformation', None}#Literal['both','components','transformation']|None
1#{ndarray, sparse matrix} of shape (n_samples, n_components)#MatrixLike
@1#{'lapack', 'randomized'}#Literal['lapack','randomized']
@1#{'varimax', 'quartimax'}#Literal['varimax','quartimax']
1#{list, tuple, set} of estimator instance or a single             estimator instance#Iterable[Estimator]|Estimator
@1#{"most_frequent", "prior", "stratified", "uniform",             "constant"}#Literal["most_frequent","prior","stratified","uniform","constant"]
@1#int or str or array-like of shape (n_outputs,)#int|str|ArrayLike
1#{array-like, object with finite length or shape}#ArrayLike
@1#{"mean", "median", "quantile", "constant"}#Literal["mean","median","quantile","constant"]
@1#int or float or array-like of shape (n_outputs,)#int|float|ArrayLike
1#float in [0.0, 1.0]#float
@1#float, int#float|int
@1#array-like of shape (n_samples,) or (n_samples, n_labels)#ArrayLike
!1#sequence of indexables with same length / shape[0]#Sequence[Indexed]
1#array-like, shape (n_samples,) or (n_samples, n_output)#ArrayLike|MatrixLike
1#estimator object implementing 'fit' and 'predict'#Estimator
@1#{'predict', 'predict_proba', 'predict_log_proba',               'decision_function'}#Literal['predict','predict_proba','predict_log_proba','decision_function']
1#array-like of shape at least 2D#MatrixLike
1#array-like of  shape (n_samples,)#ArrayLike
@1#array-like of shape (n_ticks,)#ArrayLike
@1#array-like of shape (n_values,)#ArrayLike
1#dict of str to sequence, or sequence of such#Mapping[str,Sequence]|Sequence[Mapping[str Sequence]]
1#dict or list of dicts#Mapping|Sequence[Mapping]
@1#callable, 'one-to-one' or None#Callable|Literal['one-to-one']|None
1#{ndarray, sparse matrix} of shape (n_samples,) or                 (n_samples, n_classes)#ArrayLike|MatrixLike
1#{array, sparse matrix} of shape (n_samples,) or                 (n_samples, n_classes)#ArrayLike|MatrixLike
1#int or tuple (min_degree, max_degree)#int|tuple[int,int]
1#{'uniform', 'quantile'} or array-like of shape         (n_knots, n_features)#Literal['uniform','quantile']|MatrixLike
@1#{'error', 'constant', 'linear', 'continue', 'periodic'}#Literal['error','constant','linear','continue','periodic']
@1#{'onehot', 'onehot-dense', 'ordinal'}#Literal['onehot','onehot-dense','ordinal']
@1#{'uniform', 'quantile', 'kmeans'}#Literal['uniform','quantile','kmeans']
1#{np.float32, np.float64}#Float
1#int or None (default='warn')#int|Literal['warn']|None
1#{'first', 'if_binary'} or an array-like of shape (n_features,)#Literal['first','if_binary']|ArrayLike
@1#{'error', 'ignore', 'infrequent_if_exist'}#Literal['error','ignore','infrequent_if_exist']
1#{array-like, sparse matrix} of shape                 (n_samples, n_encoded_features)#MatrixLike
@1#{'error', 'use_encoded_value'}#Literal['error','use_encoded_value']
@1#array-like of shape (n_samples, n_encoded_features)#MatrixLike
1#{array-like, sparse matrix of shape (n_samples, n_features)#MatrixLike
1#{array-like, sparse matrix} of shape (n_sample, n_features)#MatrixLike
@1#ndarray of shape (n_samples1, n_samples2)#MatrixLike
@1#array-like of shape (n_components, )#ArrayLike
1#array-like, shape (n_features,)#ArrayLike
@1#float or array-like#float|ArrayLike
@1#array-like of shape (n_samples, n_dimensions)#MatrixLike
@1#{'svd', 'lsqr', 'eigen'}#str|Literal['svd','lsqr','eigen']
1#covariance estimator#Estimator
@1#list of tuple#Sequence[tuple]
@1#array-like of shape (n_samples, n_transformed_features)#MatrixLike
1#list of Estimator objects#Sequence[Estimator]
1#list of (str, transformer) tuples#Sequence[tuple[str, Transformer]]
1#list of estimators#Sequence[Estimator]
1#{array-like, sparse graph} of shape (n_samples, n_samples)#MatrixLike
@1#{'nearest_neighbors', 'rbf', 'precomputed',                 'precomputed_nearest_neighbors'} or callable#Literal['nearest_neighbors','rbf','precomputed','precomputed_nearest_neighbors']|Callable
!1#{array-like, sparse graph, BallTree, KDTree, NearestNeighbors}#csr_matrix|{array-like, sparse graph, BallTree, KDTree, NearestNeighbors}
!1#{array-like, sparse graph, BallTree, KDTree}#ndarray|{array-like, sparse graph, BallTree, KDTree}
1#array-like, shape (n_queries, n_features)#MatrixLike
@1#{'random', 'pca'} or ndarray of shape (n_samples, n_components)#Literal['random','pca']|MatrixLike
1#True#bool
1#{array, matrix, sparse matrix, LinearOperator}#MatrixLike|LinearOperator
@1#{'hinge', 'squared_hinge'}#str|Literal['hinge','squared_hinge']
@1#{'ovr', 'crammer_singer'}#Literal['ovr','crammer_singer']
@1#{'epsilon_insensitive', 'squared_epsilon_insensitive'}#Literal['epsilon_insensitive','squared_epsilon_insensitive']
@1#{dict, 'balanced'}#Literal['balanced'|Mapping]
@1#{'squared_hinge', 'log'}#Literal['squared_hinge','log']
1#{array-like, sparse matrix} of shape (n_samples, n_features)                 or (n_samples, n_samples)#MatrixLike
@1#array-like of shape (n_samples, n_features) or                 (n_samples_test, n_samples_train)#MatrixLike
1#ndarray or DataFrame, shape (n_samples, n_features)#MatrixLike|DataFrame
1#array-like or None, shape (n_samples, ) or (n_samples, n_classes)#ArrayLike|None|MatrixLike
@1#list of Bunch#Sequence[Bunch]
1#list of (int,) or list of (int, int)#Sequence[int]|Sequence[tuple[int,int]]
@1#dict or None#Mapping|None
1#{array-like, sparse matrix, dataframe} of shape (n_samples, 2)#MatrixLike|DataFrame
@1#{'auto', 'predict_proba', 'decision_function',                 'predict'}#Literal['auto','predict_proba','decision_function','predict']
1#{array-like or dataframe} of shape (n_samples, n_features)#MatrixLike|DataFrame
@1#array-like of {int, str}#Sequence[tuple[int, str]]
@1#{'auto', 'recursion', 'brute'}#Literal['auto','recursion','brute']
@1#{'forward', 'backward'}#Literal['forward','backward']
1#str, callable, list/tuple or dict#str|Callable|Sequence|tuple|dict
@1#int or "all"#int|Literal["all"]
@1#{'percentile', 'k_best', 'fpr', 'fdr', 'fwe'}#Literal['percentile','k_best','fpr','fdr','fwe']
1#float or int depending on the feature selection mode#float|int 
@1#str or float#str|float
1#non-zero int, inf, -inf#int|np.inf|-np.inf
@1#int, callable#int|Callable
@1#array of shape [n_samples]#ArrayLike
@1#str, callable or None#str|Callable|None
1#array-like of shape (n_samples,) or None#ArrayLike|None
@1#array of shape [n_samples, n_selected_features]#MatrixLike
1#ndarray or sparse matrix of shape (n_samples, n_features)#MatrixLike
@1#ndarray of shape (n_samples,) or (n_samples, n_outputs)#ArrayLike|MatrixLike
@1#array of shape (n_classes,)#ArrayLike
1#list of length = len(coefs_) + len(intercepts_)#Sequence
1#list of length = len(params)#Sequence
@1#{'constant', 'adaptive', 'invscaling'}#str|Literal['constant','adaptive','invscaling']
1#array-like of float, shape = (n_samples, n_classes)#MatrixLike
1#array-like of float, shape = (n_samples, 1)#MatrixLike
1#tuple of shape (2,)#tuple[Any,Any]
1#int or tuple of shape (2,)#int|tuple[Any,Any]
@1#{"frobenius", "spectral"}#str|Literal["frobenius","spectral"]
@1#array-like of shape (n_alphas,)#ArrayLike
@1#array of shape (n_test_samples, n_features)#MatrixLike
@1#int or array-like of shape (n_alphas,), dtype=float#int|ArrayLike
1#sequence of array-like of shape (n_samples,) or             (n_samples, n_outputs)#Sequence[ArrayLike|MatrixLike]
!1#sequence of indexable data-structures#Sequence[Indexable]
1#{array-like, ndarray, sparse matrix}#ArrayLike|MatrixLike
1#list-like#Sequence
1#any type#Any
1#{"classifier", "regressor", "cluster", "transformer"}             or list of such str#Literal["classifier","regressor","cluster","transformer"]|Sequence[Literal["classifier","regressor","cluster","transformer"]]
1#None, str or object with the joblib.Memory interface#None|str|Memory
!1#list or tuple of input objects.#Sequence|tuple[input, ...]
1#{lists, dataframes, ndarrays, sparse matrices}#ArrayLike|MatrixLike
@1#{'F', 'C'} or None#Literal['F','C']|None
@1#str, bool or list of str#str|bool|Sequence[str]
@1#{'F', 'C'}#Literal['F','C']
1#None, int or instance of RandomState#None|int|RandomState
@1#str, list or tuple of str#str|Sequence|tuple[str, ...]
1#callable, {all, any}#Callable
1#type or tuple#type|tuple
1#ndarray of float of shape (n_samples,)#ArrayLike
1#CSR or CSC sparse matrix of shape (n_samples, n_features)#MatrixLike
1#float or ndarray of shape (n_features,) or (n_samples,),             dtype=floating#float|ArrayLike
1#sparse matrix of shape (n_samples, n_labels)#MatrixLike
1#2D array#MatrixLike
@1#list of array-like#Sequence[ArrayLike]
@1#ndarray of shape (M, len(arrays))#MatrixLike
1#array-like of float of shape (M, N)#MatrixLike
1#warning class#Warning
1#tuple of warning class#tuple[Warning]
1#exception or tuple of exception#Exception|tuple[Exception, ...]
1#str, optional#str|None
@1#list#Sequence
1#Exception or list of Exception#Exception|Sequence[Exception]
@1#str or list of str#str|Sequence[str]
1#str, list of str or tuple of str#str|Sequence[str]|tuple[str, ...]
@1#{'serial', 'parallel', 'single'}#Literal['serial','parallel','single']
1#list of estimators or `_VisualBlock`s or a single estimator#Sequewnce[Estimator]|Sequence[_VisualBlock]|Estimator
1#list of str, str, or None#Sequene[str]|str|None
@1#dict, 'balanced' or None#str|Mapping|Literal['balanced']|None
1#{array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)#ArrayLike|MatrixLike
@1#array-like of shape (n_subsample,)#ArrayLike
@1#"all" or list of str#Literal["all"]|Sequence[str]
@1#array-like of shape (n_samples, n_outputs) or (n_samples,)#MatrixLike|ArrayLike
@1#array-like of shape (n_samples, n_output) or (n_samples,)#MatrixLike|ArrayLike
1#{sparse matrix, ndarray} of shape (n, n)#MatrixLike
1#arraylike or sparse matrix, shape = (N,N)#MatrixLike
@1#boolean#bool
1#array-likes#ArrayLike
1#{array-like, sparse matrix} of size (n_samples, n_outputs)#MatrixLike
1#list of estimators instances#Sequence[Estimator]
1#int, dict, 'sequential_blas_under_openmp' or None (default=None)#int|Mapping|Literal['sequential_blas_under_openmp']|None
1#"blas", "openmp" or None (default=None)#str|Literal["blas","openmp"]|None
@1#{'mean', 'median', 'most_frequent', 'constant'}#Literal['mean','median','most_frequent','constant']
@1#{'ascending', 'descending', 'roman', 'arabic',             'random'}#Literal['ascending','descending','roman','arabic','random']
@1#{'nan_euclidean'} or callable#Literal['nan_euclidean']|Callable
1#array-like shape of (n_samples, n_features)#MatrixLike
1#int, float, str, np.nan, None or pandas.NA#int|float|str|np.nan|pd.NA|None
1#str or numerical value#str|int|float
1#array-like of shape                 (n_samples, n_features + n_features_missing_indicator)#MatrixLike
@1#{'missing-only', 'all'}#Literal['missing-only','all']
1#mapping of str to any#Mapping[str, Any]
1#int (>= 1) or float ([0, 1])#int|float
1#float in range [0, 1]#float
1#(array-like or sparse matrix} of shape (n_samples, n_features)#MatrixLike
1#float, greater than 1.0#float
1#Estimator object#Estimator
1#numpy array or sparse matrix of shape [n_samples,n_features]#MatrixLike
@1#{'hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge',        'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',        'squared_epsilon_insensitive'}#str|Literal['hinge','log_loss','log','modified_huber','squared_hinge','perceptron','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']
@1#ndarray of shape (1,)#ArrayLike
@1#{'constant', 'optimal', 'invscaling', 'adaptive'}#Literal['constant','optimal','invscaling','adaptive']
1#array, shape (n_classes, n_features)#MatrixLike
1#array, shape (n_classes,)#ArrayLike
1#{ndarray, sparse matrix, LinearOperator} of shape         (n_samples, n_features)#MatrixLike|LinearOperator
1#{float, ndarray of shape (n_targets,)}#float|ArrayLike
1#{array-like, spare matrix} of shape (n_samples, n_features)#MatrixLike
@1#{'auto', 'svd', 'eigen'}#Literal['auto','svd','eigen']
1#{ndarray, sparse matrix} of (n_samples, n_features)#MatrixLike
1#{ndarray, sparse matrix} of shape (n_samples,) or             (n_samples, n_targets)#ArrayLike|MatrixLike
1#instance of class BaseLoss from sklearn._loss.#BaseLoss
@1#{'l2','l1','elasticnet'}#Literal['l2','l1','elasticnet']
@1#ndarray of shape (n_features,) or (n_features, n_targets)#ArrayLike|MatrixLike
@1#array-like of shape (n_targets,)#ArrayLike
@1#{'l1', 'l2', 'elasticnet', 'none'}#Literal['l1','l2','elasticnet','none']
@1#{'auto', 'ovr', 'multinomial'}#Literal['auto','ovr','multinomial']
@1#int or list of floats#int|Sequence[float]
!1#int or cross-validation generator#int|cross-validation generator
@1#{'l1', 'l2', 'elasticnet'}#Literal['l1','l2','elasticnet']
1#{'auto, 'ovr', 'multinomial'}#{'auto, 'ovr', 'multinomial'}
@1#list of float#Sequence[float]
1#dict, {class_label: weight} or "balanced" or None#Mapping[str, float]|Literal["balanced"]|None
1#array, shape = [n_features]#ArrayLike
1#array, shape = [1]#ArrayLike
@1#'lbfgs'#Literal['lbfgs']
@1#{'auto', 'identity', 'log'}#Literal['auto','identity','log']
@1#None or array-like of shape (n_samples,)#None|ArrayLike
1#None, 'auto', array-like of shape (n_features, n_features)#None|Literal['auto']|MatrixLike
@1#bool or 'auto' #bool|Literal['auto']
@1#{'aic', 'bic'}#str|Literal['aic','bic']
1#array-like, shape (n_samples, )#ArrayLike
@1#{'highs-ds', 'highs-ipm', 'highs', 'interior-point',             'revised simplex'}#Literal['highs-ds','highs-ipm','highs','interior-point','revised simplex']
1#list of ndarray of shape (n_outputs,)#Sequence[ArrayLike]
@1#{'nan', 'clip', 'raise'}#Literal['nan','clip','raise']
@1#int or array-like of int#int|Sequence[int]
@1#float or ndarray of shape (n_components,), dtype=float#float|ArrayLike
1#{sparse matrix, dense matrix, LinearOperator}#MatrixLike|LinearOperator
@1#scalar#Scalar
@1#str or sequence of str#str|Sequence[str]
1#a dictionary for environment variables#Mapping[str,str]
@1#int or 'active'#int|Literal['active']
@1#str, list or None#str|Sequence|None
1#array-like, dtype=str#ArrayLike
@1#str, file-like or int#str|IO|int
1#array-like, dtype=str, file-like or int#ArrayLike|IO|int
1#{array-like, sparse matrix}, shape = [n_samples (, n_labels)]#MatrixLike
1#str or file-like in binary mode#str|IO
@1#{'train', 'test', '10_folds'}#Literal['train','test','10_folds']
1#Batch object#Batch
@1#{'SA', 'SF', 'http', 'smtp'}#Literal['SA','SF','http','smtp']
@1#array-like of shape (n_classes,) or (n_classes - 1,)#ArrayLike
1#{'dense', 'sparse'} or False#Literal['dense','sparse']|bool
@1#int or array-like#int|ArrayLike
@1#int or ndarray of shape (n_centers, n_features)#int|MatrixLike
@1#float or array-like of float#float|Sequence[float]
1#tuple of float (min, max)#tuple[float, float]
1#iterable of shape (n_rows, n_cols)#Iterable[MatrixLike]
!1#tuple of shape (n_rows, n_cols)#tuple[int, int]|tuple of shape (N,N)
1#int or array-like or shape (n_row_clusters, n_column_clusters)#Int|MatrixLike
1#{`china.jpg`, `flower.jpg`}#str
@1#list of tuples#Sequence[tuple]
@1#array-like of shape (n_samples,...)#ArrayLike
@1#tuples#tuple
1#dataframe of shape (n_features, n_samples)#MatrixLike
@1#"fmin_l_bfgs_b" or callable#Literal["fmin_l_bfgs_b"]|Callable
1#array-like of shape (n_kernel_params,) default=None#ArrayLike|None
@1#ndarray of shape (n_dims,)#ArrayLike
1#list of Kernels#Sequence[Kernel]
1#float >= 0#float
@1#{"linear", "additive_chi2", "chi2", "poly", "polynomial",               "rbf", "laplacian", "sigmoid", "cosine"} or callable#Literal["linear","additive_chi2","chi2","poly","polynomial","rbf","laplacian","sigmoid","cosine"]|Callable
@1#{'one_vs_rest', 'one_vs_one'}#Literal['one_vs_rest','one_vs_one']
@1#{'threshold', 'k_best'}#str|Literal['threshold','k_best']
1#{array-like, sparse matrix} of shape (n_samples,)#ArrayLike
1#float, 1e-3#float
@1#{None, ndarray}#None|ArrayLike
@1#{None, int}#None|int
@1#None or array of shape (n_samples,)#None|ArrayLike
1#{np.float64, np.float32}#Float
@1#array of shape (n_samples,) or (n_samples, 1)#ArrayLike|MatrixLike
@1#ndarray of shape (height, width) or (height, width, channel)#MatrixLike
@1#ndarray of shape (height, width) or             (height, width, channel), dtype=bool#MatrixLike
@1#ndarray of shape (n_x, n_y, n_z), dtype=bool#MatrixLike
@1#ndarray of shape (image_height, image_width) or         (image_height, image_width, n_channels)#MatrixLike
@1#ndarray of shape (n_patches, patch_height, patch_width) or         (n_patches, patch_height, patch_width, n_channels)#MatrixLike
1#tuple of int (image_height, image_width) or         (image_height, image_width, n_channels)#tuple[int, int]|tuple[int,int,int]
@1#ndarray of shape (n_samples, image_height, image_width) or             (n_samples, image_height, image_width, n_channels)#MatrixLike
@1#string#str
!1#bytes or str#bytes|str
1#sparse matrix of shape n_samples, n_features)#MatrixLike
1#sparse matrix of (n_samples, n_features)#MatrixLike
1#Mapping or iterable over Mappings of shape (n_samples,)#Mapping[str,ArrayLike]|Iterator[Mapping[str,ArrayLike]]
1#numpy dtype#DType
1#iterable over iterable over raw features, length = n_samples#Iterator[Iterator]
!1#pytest config#pytest config
1#list of collected items#Sequence
!1#pytest item#pytest item
1#array-like of shape (n_seeds, n_features) or None#MatrixLike
!1#int, instance of sklearn.cluster model#int|sklearn.cluster
@1#array-like of shape (n_samples, n_samples)#MatrixLike
@1#{'k-means++', 'random'} or callable#Literal['k-means++','random']|Callable
@1#{"lloyd", "elkan"}#Literal["lloyd","elkan"]
@1#{"biggest_inertia", "largest_cluster"}#Literal["biggest_inertia","largest_cluster"]
1#array-like, shape: (n_samples, n_clusters)#MatrixLike
@1#array-like of shape (n_samples, n_clusters)#MatrixLike
1#{array-like, sparse matrix} of shape (n_samples, n_samples)#MatrixLike
1#{None, 'arpack', 'lobpcg', or 'amg'}#None|Literal['arpack', 'lobpcg', 'amg'
1#dict of str to any#Mapping[str, Any]
@1#array-like of shape (n_samples, n_clusters) or (n_clusters,)#MatrixLike|ArrayLike
1#{'k-means++', 'random', or ndarray of shape             (n_clusters, n_features)#Literal['k-means++', 'random']|MatrixLike
1#int or tuple (n_row_clusters, n_column_clusters)#int|tuple[int, int]
@1#{'bistochastic', 'scale', 'log'}#Literal['bistochastic','scale','log']
1#{'k-means++', 'random'} or ndarray of (n_clusters, n_features)#Literal['k-means++','random']|MatrixLike
1#{array-like, sparse (CSR) matrix} of shape (n_samples, n_features) or             (n_samples, n_samples)#MatrixLike
1#ndarray of shape (n_samples, n_features), or                 (n_samples, n_samples) if metric=’precomputed’#MatrixLike
1#ndarray of shape (n_samples, n_features), or             (n_samples, n_samples) if metric='precomputed'#MatrixLike
@1#{"average", "complete", "single"}#str|Literal["average","complete","single"]
@1#{'ward', 'complete', 'average', 'single'}#Literal['ward','complete','average','single']
1#array-like, shape (n_samples, n_features) or                 (n_samples, n_samples)#MatrixLike
@1#{"ward", "complete", "average", "single"}#Literal["ward","complete","average","single"]
1#array-like of shape (n_samples,) or default=None#ArrayLike|None
@1#{'auto', 'predict_proba', 'decision_function', 'predict'}#Literal['auto','predict_proba','decision_function','predict']
@1#ndarray of shape (n_samples_at_node,), dtype=np.uint#ArrayLike
@1#TreeNode#TreeNode
@1#ndarray of shape (n_samples, n_features), dtype=np.uint8#MatrixLike
1#ndarray, dtype=np.uint32#ArrayLike
1#bool or ndarray, dtype=bool#ndarray|bool|ArrayLike
@1#array-like of shape (n_features,), dtype=int#ArrayLike
@1#array-like of floats#ArrayLike
@1#{'squared_error', 'absolute_error', 'poisson', 'quantile'}#Literal['squared_error','absolute_error','poisson','quantile']
@1#{'log_loss', 'auto', 'binary_crossentropy', 'categorical_crossentropy'}#Literal['log_loss','auto','binary_crossentropy','categorical_crossentropy']
1#list of {ndarray, None} of shape (n_features,)#Sequence[None|ArrayLike]
1#ndarray of PREDICTOR_RECORD_DTYPE#ArrayLike
@1#ndarray of shape (n_categorical_features, 8)#MatrixLike
!1#uint8#int
1#ndarray, shape (n_samples, n_target_features)#MatrixLike
1#ndarray, shape (n_target_features)#ArrayLike
1#ndarray, shape (n_samples)#ArrayLike
@1#{'SAMME', 'SAMME.R'}#Literal['SAMME','SAMME.R']
@1#{'linear', 'square', 'exponential'}#Literal['linear','square','exponential']
@1#{'log_loss', 'deviance', 'exponential'}#Literal['log_loss','deviance','exponential']
@1#{'squared_error', 'absolute_error', 'huber', 'quantile'}#Literal['squared_error','absolute_error','huber','quantile']
1#{array-like, sparse matrix, dataframe} of shape                 (n_samples, n_features)#MatrixLike
@1#{'hard', 'soft'}#Literal['hard','soft']
@1#array-like of shape (n_classifiers,)#ArrayLike
@1#array-like of shape (n_regressors,)#ArrayLike
@1#{"squared_error", "absolute_error", "poisson"}#Literal["squared_error","absolute_error","poisson"]
@1#{"squared_error", "absolute_error"}#Literal["squared_error","absolute_error"]
1#ndarray of shape, (n,)#ArrayLike
@1#{'micro', 'samples', 'weighted', 'macro'} or None#Literal['micro','samples','weighted','macro']|None
1#ndarray of shape of (n_samples,)#ArrayLike
@1#{'micro', 'macro', 'samples', 'weighted'} or None#Literal['micro','macro','samples','weighted']|None
1#float > 0 and <= 1#float
@1#{'raise', 'ovr', 'ovo'}#Literal['raise','ovr','ovo']
1#str, type, list of type#str|type|Sequence[type]
@1#array-like of shape (n_samples_Y,) or (n_samples_Y, 1)             or (1, n_samples_Y)#ArrayLike|MatrixLike
@1#array-like of shape (n_samples_X,) or (n_samples_X, 1)             or (1, n_samples_X)#ArrayLike|MatrixLike
1#np.nan or int#np.nan|int
@1#array-like of shape (n_samples_X, 2)#MatrixLike
@1#array-like of shape (n_samples_Y, 2)#MatrixLike
1#{ndarray, sparse matrix} of shape (n_samples_X, n_features)#MatrixLike
1#{ndarray, sparse matrix} of shape (n_samples_Y, n_features)#MatrixLike
@1#ndarray of shape (n_samples_X, n_samples_X) or (n_samples_X, n_features)#MatrixLike
1#float, slope of the pinball loss#float
@1#{'raw_values', 'uniform_average'} or array-like#Literal['raw_values','uniform_average']|ArrayLike
@1#{'raw_values', 'uniform_average', 'variance_weighted'} or             array-like of shape (n_outputs,)#Literal['raw_values','uniform_average','variance_weighted']|ArrayLike
1#{'raw_values', 'uniform_average', 'variance_weighted'},             array-like of shape (n_outputs,) or None#Literal['raw_values','uniform_average','variance_weighted']|ArrayLike|None
@1#ndarray of shape (n_classes, n_classes)#MatrixLike
1#numeric type#int|float
1#{ndarray, sparse matrix} of shape             (n_classes_true, n_classes_pred)#MatrixLike
1#int array, shape = (``n_samples``,)#ArrayLike
1#array, shape = (``n_samples``, )#ArrayLike
@1#array-like of shape (n_samples,), dtype=int#ArrayLike
@1#'jaccard' or callable#Literal['jaccard']|Callable
@1#array-like of shape (n_classes)#ArrayLike
@1#{'linear', 'quadratic'}#Literal['linear','quadratic']
@1#{'micro', 'macro', 'samples', 'weighted',             'binary'} or None#Literal['micro','macro','samples','weighted','binary']|None
!1#"warn", {0.0, 1.0}#Literal[0.0,1.0]|Literal["warn"]
@1#{'binary', 'micro', 'macro', 'samples', 'weighted'}#Literal['binary','micro','macro','samples','weighted']
1#tuple or set, for internal use#tuple|set
@1#array-like of shape (n_labels,)#ArrayLike
1#list of str of shape (n_labels,)#ArrayLike
1#array-like of float, shape = (n_samples, n_classes) or (n_samples,)#MatrixLike|ArrayLike
@1#array of shape (n_samples,) or (n_samples, n_classes)#MatrixLike|ArrayLike
@1#{"squared_error", "friedman_mse", "absolute_error",             "poisson"}#Literal["squared_error","friedman_mse","absolute_error","poisson"]
@1#{"squared_error", "friedman_mse"}#Literal["squared_error","friedman_mse"]
1#decision tree regressor or classifier#Regressor|Classifier
@1#list of strings#Sequence[str]
1#matplotlib axis#Axis
1#decision tree classifier#Classifier
@1#object or str#Any

"""



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