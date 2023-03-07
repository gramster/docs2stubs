
import os


def write_sklearn_native_stubs():
    # Fill in some Cython gaps until we have a better solution
    with open(f'typings/sklearn/utils/_random.pyi', 'w') as f:
            f.write('''
from typing import Literal
import numpy as np
from numpy.random import RandomState

def sample_without_replacement(n_population: int, n_samples: int, random_state: int|RandomState|None=None, method: Literal["auto", "tracking_selection", "reservoir_sampling", "pool"] = "auto") -> np.ndarray: ...
''')
    with open(f'typings/sklearn/utils/murmurhash.pyi', 'w') as f:
            f.write('''import numpy as np

def murmurhash3_32(key:np.int32|bytes|str|np.ndarray, seed: int=0, positive: bool=False)->int: ...
''')
    with open(f'typings/sklearn/_loss/_loss.pyi', 'w') as f:
        f.write("""
class CyLossFunction:
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...


class CyHalfSquaredError(CyLossFunction):
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...


class CyAbsoluteError(CyLossFunction):
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...


class CyPinballLoss(CyLossFunction):
    quantile: float
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...


class CyHalfPoissonLoss(CyLossFunction):
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...


class CyHalfGammaLoss(CyLossFunction):
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...


class CyHalfTweedieLoss(CyLossFunction):
    power: float
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...


class CyHalfTweedieLossIdentity(CyLossFunction):
    power: float
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...

    
class CyHalfBinomialLoss(CyLossFunction):
    def cy_loss(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_gradient(self, y_true: float, raw_prediction: float) -> float: ...
    def cy_grad_hess(self, y_true: float, raw_prediction: float)->tuple[float,float]: ...


""")
    with open(f'typings/sklearn/metrics/_dist_metrics.pyi', 'w') as f:
        f.write("""                 
import numpy as np
from typing import Iterable, Any
from scipy.sparse import csr_matrix


def get_valid_metric_ids(L: Iterable) -> list[str]: ...


class DistanceMetric():

    def __init__(self)->None: ...

    @classmethod
    def get_metric(cls, metric: str, **kwargs) -> DistanceMetric: ...

    def dist(self, x1, x2, size): ...
    def rdist(self, x1, x2, size): ...
    def pdist(self, X, D) -> int: ...
    def cdist(self, X, Y, D) -> int: ...
    def dist_csr(self, x1_data, x1_indices, x2_data, x2_indices, x1_start, x1_end, x2_start, x2_end, size): ...
    def rdist_csr(self, x1_data, x1_indices, x2_data, x2_indices, x1_start, x1_end, x2_start, x2_end, size): ...
    def pdist_csr(self, x1_data, x1_indices, x1_indptr, size, D)->int: ...
    def cdist_csr(self, x1_data, x1_indices, x1_indptr, x2_data, x2_indices, x2_indptr, size, D)->int: ...
    def rdist_to_dist(self, rdist: float) -> float: ...
    def dist_to_rdist(self, dist: float) -> float: ...
    def pairwise(self, X: np.ndarray|csr_matrix, Y: np.ndarray|csr_matrix|None=None)->np.ndarray: ...

    
class DistanceMetric32():

    def __init__(self)->None: ...

    @classmethod
    def get_metric(cls, metric: str, **kwargs) -> DistanceMetric: ...

    def dist(self, x1, x2, size): ...
    def rdist(self, x1, x2, size): ...
    def pdist(self, X, D) -> int: ...
    def cdist(self, X, Y, D) -> int: ...
    def dist_csr(self, x1_data, x1_indices, x2_data, x2_indices, x1_start, x1_end, x2_start, x2_end, size): ...
    def rdist_csr(self, x1_data, x1_indices, x2_data, x2_indices, x1_start, x1_end, x2_start, x2_end, size): ...
    def pdist_csr(self, x1_data, x1_indices, x1_indptr, size, D)->int: ...
    def cdist_csr(self, x1_data, x1_indices, x1_indptr, x2_data, x2_indices, x2_indptr, size, D)->int: ...
    def rdist_to_dist(self, rdist: float) -> float: ...
    def dist_to_rdist(self, dist: float) -> float: ...
    def pairwise(self, X: np.ndarray|csr_matrix, Y: np.ndarray|csr_matrix|None=None)->np.ndarray: ...

class BrayCurtisDistance(DistanceMetric): ...
class BrayCurtisDistance32(DistanceMetric32): ...
class CanberraDistance(DistanceMetric): ...
class CanberraDistance32(DistanceMetric32): ...
class ChebyshevDistance(DistanceMetric): ...
class ChebyshevDistance32(DistanceMetric32): ...
class DiceDistance(DistanceMetric): ...
class DiceDistance32(DistanceMetric32): ...
class EuclideanDistance(DistanceMetric): ...
class EuclideanDistance32(DistanceMetric32): ...
class HammingDistance(DistanceMetric): ...
class HammingDistance32(DistanceMetric32): ...
class HaversineDistance(DistanceMetric): ...
class HaversineDistance32(DistanceMetric32): ...
class JaccardDistance(DistanceMetric): ...
class JaccardDistance32(DistanceMetric32): ...
class KulsinskiDistance(DistanceMetric): ...
class KulsinskiDistance32(DistanceMetric32): ...
class MahalanobisDistance(DistanceMetric): ...
class MahalanobisDistance32(DistanceMetric32): ...
class ManhattanDistance(DistanceMetric): ...
class ManhattanDistance32(DistanceMetric32): ...
class MatchingDistance(DistanceMetric): ...
class MatchingDistance32(DistanceMetric32): ...
class MinkowskiDistance(DistanceMetric): ...
class MinkowskiDistance32(DistanceMetric32): ...
class PyFuncDistance(DistanceMetric): ...
class PyFuncDistance32(DistanceMetric32): ...
class RogersTanimotoDistance(DistanceMetric): ...
class RogersTanimotoDistance32(DistanceMetric32): ...
class RussellRaoDistance(DistanceMetric): ...
class RussellRaoDistance32(DistanceMetric32): ...
class SEuclideanDistance(DistanceMetric): ...
class SEuclideanDistance32(DistanceMetric32): ...
class SokalMichenerDistance(DistanceMetric): ...
class SokalMichenerDistance32(DistanceMetric32): ...
class SokalSneathDistance(DistanceMetric): ...
class SokalSneathDistance32(DistanceMetric32): ...
class WMinkowskiDistance(DistanceMetric): ...
class WMinkowskiDistance32(DistanceMetric32): ...


METRIC_MAPPING: dict[str, Any]


""")
    with open(f'typings/sklearn/metrics/cluster/_expected_mutual_info_fast.pyi', 'w') as f:
        f.write("""  
def expected_mutual_information(contingency, n_samples: int) -> float: ...
""")
    with open(f'typings/sklearn/linear_model/_sag_fast.pyi', 'w') as f:
        f.write("""  
import numpy as np
from ..utils._seq_dataset import SequentialDataset32, SequentialDataset64


def sag32(
    dataset: SequentialDataset32,
    weights_array: np.ndarray,
    intercept_array: np.ndarray,
    n_samples: int,
    n_features: int,
    n_classes: int,
    tol: float,
    max_iter: int,
    loss_function: str,
    step_size: float,
    alpha: float,
    beta: float,
    sum_gradient_init: np.ndarray,
    gradient_memory_init: np.ndarray,
    seen_init: np.ndarray,
    num_seen: int,
    fit_intercept: bool,
    intercept_sum_gradient_init: np.ndarray,
    intercept_decay: float,
    saga: bool,
    verbose: bool
) -> tuple[int, int]: ...


def sag64(
    dataset: SequentialDataset64,
    weights_array: np.ndarray,
    intercept_array: np.ndarray,
    n_samples: int,
    n_features: int,
    n_classes: int,
    tol: float,
    max_iter: int,
    loss_function: str,
    step_size: float,
    alpha: float,
    beta: float,
    sum_gradient_init: np.ndarray,
    gradient_memory_init: np.ndarray,
    seen_init: np.ndarray,
    num_seen: int,
    fit_intercept: bool,
    intercept_sum_gradient_init: np.ndarray,
    intercept_decay: float,
    saga: bool,
    verbose: bool
) -> tuple[int, int]: ...
""")
    with open(f'typings/sklearn/linear_model/_sgd_fast.pyi', 'w') as f:
        f.write("""  
class LossFunction:
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y) -> float: ...


class Regression(LossFunction):
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y) -> float: ...


class Classification(LossFunction):
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y) -> float: ...

    
class ModifiedHuber(Classification):
    \"""Modified Huber loss for binary classification with y in {-1, 1}

    This is equivalent to quadratically smoothed SVM with gamma = 2.

    See T. Zhang 'Solving Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    \"""
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

    
class Hinge(Classification):
    \"""Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by SVM.
        When threshold=0.0, one gets the loss used by the Perceptron.
    \"""
    threshold: float
    def __init__(self, threshold: float=1.0) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...


class SquaredHinge(Classification):
    \"""Squared Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by
        (quadratically penalized) SVM.
    \"""
    threshold: float
    def __init__(self, threshold: float=1.0) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...


class Log(Classification):
    \"""Logistic regression loss for binary classification with y in {-1, 1}\"""
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

    
class SquaredLoss(Regression):
    \"""Squared loss traditional used in linear regression.\"""
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...


class Huber(Regression):
    \"""Huber regression loss

    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    linear in for large errors).

    https://en.wikipedia.org/wiki/Huber_Loss_Function
    \"""
    c: float
    def __init__(self, c: float) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...


class EpsilonInsensitive(Regression):
    \"""Epsilon-Insensitive loss (used by SVR).

    loss = max(0, |y - p| - epsilon)
    \"""
    epsilon: float
    def __init__(self, epsilon: float) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...


class SquaredEpsilonInsensitive(Regression):
    \"""Epsilon-Insensitive loss.

    loss = max(0, |y - p| - epsilon)^2
    \"""
    epsilon: float
    def __init__(self, epsilon: float) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...
""")
    with open(f'typings/sklearn/utils/_seq_dataset.pyi', 'w') as f:
        f.write("""
import numpy as np


class SequentialDataset32():
    current_index: int
    index: np.ndarray
    index_data_ptr: int
    n_samples: int
    seed: int

    def shuffle(self, seed: int) -> None: ...
    def next(self, x_data_ptr, x_ind_ptr, nnz, y, sample_weight) -> None: ...
    def random(self, x_data_ptr, x_ind_ptr, nnz, y, sample_weight) -> int: ...

    
class SequentialDataset64():
    current_index: int
    index: np.ndarray
    index_data_ptr: int
    n_samples: int
    seed: int

    def shuffle(self, seed: int) -> None: ...
    def next(self, x_data_ptr, x_ind_ptr, nnz, y, sample_weight) -> None: ...
    def random(self, x_data_ptr, x_ind_ptr, nnz, y, sample_weight) -> int: ...


class ArrayDataset32(SequentialDataset32):
    def __init__(self, X, Y, sample_weights, seed=1) -> None: ...

    
class ArrayDataset64(SequentialDataset64):
    def __init__(self, X, Y, sample_weights, seed=1) -> None: ...


class CSRDataset64(SequentialDataset64):
    def __init__(self, X_data, X_indptr, X_indices, Y, sample_weights, seed=1) -> None: ...

    
class CSRDataset32(SequentialDataset32):
    def __init__(self, X_data, X_indptr, X_indices, Y, sample_weights, seed=1) -> None: ...

""")
    with open(f'typings/sklearn/tree/_tree.pyi', 'w') as f:
        f.write("""
import numpy as np
from typing import Any
from ._splitter import Splitter


class Node:
    left_child: int
    right_child: int
    feature: int
    threshold: float
    impurity: float
    n_node_samples: int
    weighted_n_node_samples: float


class DepthFirstTreeBuilder(TreeBuilder):
    \"""Build a decision tree in depth-first fashion.\"""

    def __init__(self, splitter: Splitter, min_samples_split: int,
                  min_samples_leaf: int,  min_weight_leaf: float,
                  max_depth: int,
                  min_impurity_decrease: float) -> None: ...

    def build(
        self,
        tree: Tree,
        X: object,
        y: np.ndarray,
        sample_weight: np.ndarray|None=None,
    ) -> None: ...


class BestFirstTreeBuilder(TreeBuilder):
    \"""Build a decision tree in best-first fashion.
    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.
    \"""
    max_leaf_nodes: int

    def __init__(self, splitter: Splitter, min_samples_split: int,
                  min_samples_leaf: int,  min_weight_leaf: float,
                  max_depth: int, max_leaf_nodes: int,
                  min_impurity_decrease: float) -> None: ...

    def build(
        self,
        tree: Tree,
        X: object,
        y: np.ndarray,
        sample_weight: np.ndarray|None=None,
    ) -> None: ...

 

class Tree:

    n_features: int
    n_classes: int
    n_outputs: int
    max_n_classes: int

    max_dept: int
    node_count: int
    capacity: int
    nodes: np.ndarray
    value: np.ndarray
    value_stride: int

    def predict(self, X) -> np.ndarray: ...
    def apply(self, X) -> np.ndarray: ...
    def decision_path(self, X) -> Any: ...
    def compute_node_depths(self): ...
    def compute_feature_importances(self, normalize): ...


class TreeBuilder:

    splitter: Splitter
    min_samples_split: int
    min_samples_leaf: int
    min_weight_leaf: float
    max_depth: int
    min_impurity_decrease: float

    def build(
        self,
        tree: Tree,
        X,
        y,
        sample_weight,
    ): ...

    
def ccp_pruning_path(orig_tree: Tree) -> dict: ...
    
""")
    with open(f'typings/sklearn/tree/_splitter.pyi', 'w') as f:
        f.write("""
import numpy as np
from ._criterion import Criterion


class SplitRecord:
    # Data to track sample split
    feature: int         # Which feature to split on.
    pos: int             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    threshold: float       # Threshold to split at.
    improvement: float     # Impurity improvement given parent node.
    impurity_left: float   # Impurity of the left split.
    impurity_right: float  # Impurity of the right split.


class Splitter:
    \""" The splitter searches in the input space for a feature and a threshold
        to split the samples samples[start:end]. \"""

    criterion: Criterion      # Impurity criterion
    max_features: int         # Number of features to test
    min_samples_leaf: int     # Min samples in a leaf
    min_weight_leaf: float    # Minimum weight in a leaf

    random_state: object             # Random state
    rand_r_state: int           # sklearn_rand_r random number state

    samples: np.ndarray
    n_samples: int
    weighted_n_samples: float
    features: np.ndarray
    constant_features: np.ndarray
    n_features: int
    feature_values: np.ndarray

    start: int
    end: int

    y: np.ndarray
    sample_weight: np.ndarray

    def init(
        self,
        X: object,
        y: np.ndarray,
        sample_weight: np.ndarray
    ) -> int: ...

    def node_reset(
        self,
        start: int,
        end: int,
        weighted_n_node_samples: list[float]
    ) -> int: ...

    def node_split(
        self,
        impurity: float,
        split: SplitRecord,
        n_constant_features: list[int]
    ) -> int: ...

    def node_value(self, dest: list[float]) -> None: ...
    def node_impurity(self) -> float: ...
""")
                
    with open(f'typings/sklearn/tree/_criterion.pyi', 'w') as f:
        f.write("""
import numpy as np


class Criterion:

    y: np.ndarray
    sample_weight: np.ndarray

    sample_indices: np.ndarray
    start: int
    pos: int
    end: int

    n_outputs: int
    n_samples: int
    n_node_samples: int
    weighted_n_samples: float
    weighted_n_node_samples: float
    weighted_n_left: float
    weighted_n_right: float

    def init(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray,
        weighted_n_samples: float,
        sample_indices: np.ndarray,
        start: int,
        end: int
    ) -> int: ...
    def reset(self) -> int: ...
    def reverse_reset(self) -> int: ...
    def update(self, new_pos: int) -> int: ...
    def node_impurity(self) -> float: ...
    def children_impurity(
        self,
        impurity_left: list[float],
        impurity_right: list[float]
    ) -> None: ...
    def node_value(
        self,
        dest: list[float]
    ) -> None: ...
    def impurity_improvement(
        self,
        impurity_parent: float,
        impurity_left: float,
        impurity_right: float
    ) -> float: ...
    def proxy_impurity_improvement(self) -> float: ...


class ClassificationCriterion(Criterion):


    n_classes: np.ndarray
    max_n_classes: int

    sum_total: np.ndarray
    sum_left: np.ndarray
    sum_right: np.ndarray

    
class RegressionCriterion(Criterion):

    sq_sum_total: float

    sum_total: np.ndarray
    sum_left: np.ndarray
    sum_right: np.ndarray
""")
    with open(f'typings/sklearn/utils/sparsefuncs_fast.pyi', 'w') as f:
        f.write("""
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


def csr_row_norms(X: np.ndarray) -> np.ndarray:
    \"""Squared L2 norm of each row in CSR matrix X.\"""
    ...


def csr_mean_variance_axis0(X: csr_matrix, weights:np.ndarray|None=None, return_sum_weights:bool=False) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"""Compute mean and variance along axis 0 on a CSR matrix
    Uses a np.float64 accumulator.
    Parameters
    ----------
    X : CSR sparse matrix, shape (n_samples, n_features)
        Input data.
    weights : ndarray of shape (n_samples,), dtype=floating, default=None
        If it is set to None samples will be equally weighted.
        .. versionadded:: 0.24
    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature.
        .. versionadded:: 0.24
    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means
    variances : float array with shape (n_features,)
        Feature-wise variances
    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if return_sum_weights is True.
    \"""
    ...


def csc_mean_variance_axis0(X: csc_matrix, weights: np.ndarray|None=None, return_sum_weights: bool=False) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"""Compute mean and variance along axis 0 on a CSC matrix
    Uses a np.float64 accumulator.
    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.
    weights : ndarray of shape (n_samples,), dtype=floating, default=None
        If it is set to None samples will be equally weighted.
        .. versionadded:: 0.24
    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature.
        .. versionadded:: 0.24
    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means
    variances : float array with shape (n_features,)
        Feature-wise variances
    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if return_sum_weights is True.
    \"""
    ...

        
def incr_mean_variance_axis0(X: csr_matrix|csc_matrix, last_mean: np.ndarray, last_var: np.ndarray, \
        last_n: np.ndarray, weights: np.ndarray|None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"""Compute mean and variance along axis 0 on a CSR or CSC matrix.
    last_mean, last_var are the statistics computed at the last step by this
    function. Both must be initialized to 0.0. last_n is the
    number of samples encountered until now and is initialized at 0.
    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
      Input data.
    last_mean : float array with shape (n_features,)
      Array of feature-wise means to update with the new data X.
    last_var : float array with shape (n_features,)
      Array of feature-wise var to update with the new data X.
    last_n : float array with shape (n_features,)
      Sum of the weights seen so far (if weights are all set to 1
      this will be the same as number of samples seen so far, before X).
    weights : float array with shape (n_samples,) or None. If it is set
      to None samples will be equall weighted.
    Returns
    -------
    updated_mean : float array with shape (n_features,)
      Feature-wise means
    updated_variance : float array with shape (n_features,)
      Feature-wise variances
    updated_n : int array with shape (n_features,)
      Updated number of samples seen
    \"""
    ...

    
def inplace_csr_row_normalize_l1(X: np.ndarray) -> None:
    \"""Inplace row normalize using the l1 norm\"""
    ...

    
def inplace_csr_row_normalize_l2(X: np.ndarray) -> None:
    \"""Inplace row normalize using the l2 norm\"""
    ...


def assign_rows_csr(
    X: csr_matrix,
    X_rows: np.ndarray,
    out_rows: np.ndarray,
    out: np.ndarray,
) -> None:
    \"""Densify selected rows of a CSR matrix into a preallocated array.
    Like out[out_rows] = X[X_rows].toarray() but without copying.
    No-copy supported for both dtype=np.float32 and dtype=np.float64.
    Parameters
    ----------
    X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)
    X_rows : array, dtype=np.intp, shape=n_rows
    out_rows : array, dtype=np.intp, shape=n_rows
    out : array, shape=(arbitrary, n_features)
    \"""   
    ...
""")

    if not os.path.exists('typings/sklearn/metrics/_pairwise_distances_reduction'):
        os.mkdir('typings/sklearn/metrics/_pairwise_distances_reduction')
    with open(f'typings/sklearn/metrics/_pairwise_distances_reduction/__init__.pyi', 'w') as f:
        f.write("""
from ._dispatcher import (
    BaseDistancesReductionDispatcher,
    ArgKmin,
    RadiusNeighbors,
    sqeuclidean_row_norms,
)

__all__ = [
    "BaseDistancesReductionDispatcher",
    "ArgKmin",
    "RadiusNeighbors",
    "sqeuclidean_row_norms",
]
""")  
    with open(f'typings/sklearn/metrics/_pairwise_distances_reduction/_dispatcher.pyi', 'w') as f:
        f.write("""
from abc import abstractmethod

import numpy as np

from typing import Any, Literal

from scipy.sparse import csr_matrix, spmatrix


def sqeuclidean_row_norms(X: np.ndarray|csr_matrix, num_threads: int) -> np.ndarray:
    \"""Compute the squared euclidean norm of the rows of X in parallel.

    Parameters
    ----------
    X : ndarray or CSR matrix of shape (n_samples, n_features)
        Input data. Must be c-contiguous.

    num_threads : int
        The number of OpenMP threads to use.

    Returns
    -------
    sqeuclidean_row_norms : ndarray of shape (n_samples,)
        Arrays containing the squared euclidean norm of each row of X.
    \"""
    ...


class BaseDistancesReductionDispatcher:
    \"""Abstract base dispatcher for pairwise distance computation & reduction.

    Each dispatcher extending the base :class:`BaseDistancesReductionDispatcher`
    dispatcher must implement the :meth:`compute` classmethod.
    \"""

    @classmethod
    def valid_metrics(cls) -> list[str]: ...

    @classmethod
    def is_usable_for(cls, X: np.ndarray|spmatrix, Y: np.ndarray|spmatrix, metric: str = "euclidean") -> bool:
        \"""Return True if the dispatcher can be used for the
        given parameters.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
            Input data.

        Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features)
            Input data.

        metric : str, default='euclidean'
            The distance metric to use.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        Returns
        -------
        True if the dispatcher can be used, else False.
        \"""
        ...

    @classmethod
    @abstractmethod
    def compute(
        cls,
        X: np.ndarray|csr_matrix,
        Y: np.ndarray|csr_matrix,
        **kwargs,
    ): ...



class ArgKmin(BaseDistancesReductionDispatcher):
    \"""Compute the argkmin of row vectors of X on the ones of Y.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances.

    ArgKmin is typically used to perform
    bruteforce k-nearest neighbors queries.

    This class is not meant to be instanciated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    \"""

    @classmethod
    def compute(
        cls,
        X: np.ndarray|csr_matrix,
        Y: np.ndarray|csr_matrix,
        k: int,
        metric: str = "euclidean",
        chunk_size: int|None = None,
        metric_kwargs: dict[str,Any]|None = None,
        strategy: Literal['auto', 'parallel_on_X', 'parallel_on_Y']|None = None,
        return_distance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        \"""Compute the argkmin reduction.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        k : int
            The k for the argkmin reduction.

        metric : str, default='euclidean'
            The distance metric to use for argkmin.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        return_distance : boolean, default=False
            Return distances between each X vector and its
            argkmin if set to True.

        Returns
        -------
        If return_distance=False:
          - argkmin_indices : ndarray of shape (n_samples_X, k)
            Indices of the argkmin for each vector in X.

        If return_distance=True:
          - argkmin_distances : ndarray of shape (n_samples_X, k)
            Distances to the argkmin for each vector in X.
          - argkmin_indices : ndarray of shape (n_samples_X, k)
            Indices of the argkmin for each vector in X.

        Notes
        -----
        This classmethod inspects the arguments values to dispatch to the
        dtype-specialized implementation of :class:`ArgKmin`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        \"""
        ...


class RadiusNeighbors(BaseDistancesReductionDispatcher):
    \"""Compute radius-based neighbors for two sets of vectors.

    For each row-vector X[i] of the queries X, find all the indices j of
    row-vectors in Y such that:

                        dist(X[i], Y[j]) <= radius

    The distance function `dist` depends on the values of the `metric`
    and `metric_kwargs` parameters.

    This class is not meant to be instanciated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    \"""

    @classmethod
    def compute(
        cls,
        X: np.ndarray | csr_matrix,
        Y: np.ndarray | csr_matrix,
        radius: float,
        metric: str = "euclidean",
        chunk_size: int | None = None,
        metric_kwargs: dict[str,Any] | None = None,
        strategy: Literal['auto', 'parallel_on_X', 'parallel_on_Y']|None = None,
        return_distance: bool = False,
        sort_results: bool = False,
    ):
        \"""Return the results of the reduction for the given arguments.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        radius : float
            The radius defining the neighborhood.

        metric : str, default='euclidean'
            The distance metric to use.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        return_distance : boolean, default=False
            Return distances between each X vector and its neighbors if set to True.

        sort_results : boolean, default=False
            Sort results with respect to distances between each X vector and its
            neighbors if set to True.

        Returns
        -------
        If return_distance=False:
          - neighbors_indices : ndarray of n_samples_X ndarray
            Indices of the neighbors for each vector in X.

        If return_distance=True:
          - neighbors_indices : ndarray of n_samples_X ndarray
            Indices of the neighbors for each vector in X.
          - neighbors_distances : ndarray of n_samples_X ndarray
            Distances to the neighbors for each vector in X.

        Notes
        -----
        This classmethod inspects the arguments values to dispatch to the
        dtype-specialized implementation of :class:`RadiusNeighbors`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        \"""
        ...

""")
    if not os.path.exists("typings/sklearn/__check_build"):
        os.mkdir("typings/sklearn/__check_build")
    with open("typings/sklearn/__check_build/__init__.pyi", "w") as f:
        f.write("""
from ._check_build import check_build
""")
    with open("typings/sklearn/__check_build/_check_build.pyi", "w") as f:
        f.write("""
def check_build() -> None: ...
""")
                    
    with open("typings/sklearn/utils/_readonly_array_wrapper.pyi", "w") as f:
        f.write("""
class ReadonlyArrayWrapper:
    wraps: object

    def __init__(self, wraps: object) -> None:
        ...
""")
    with open("typings/sklearn/utils/_fast_dict.pyi", "w") as f:
        f.write("""
class IntFloatDict:
    cpp_map: dict
""")
    with open("typings/sklearn/cluster/_dbscan_inner.pyi", "w") as f:
        f.write("""
import numpy as np


def dbscan_inner(is_core: np.ndarray, neighborhoods: np.ndarray, labels: np.ndarray) -> None: ...
""")
                
    with open("typings/sklearn/cluster/_k_means_common.pyi", "w") as f:
        f.write("""
CHUNK_SIZE: int = 256
""")
    with open("typings/sklearn/cluster/_k_means_elkan.pyi", "w") as f:
        f.write("""
import numpy as np
from scipy.sparse import spmatrix


def init_bounds_dense(
        X: np.ndarray,
        centers: np.ndarray,
        center_half_distances: np.ndarray,
        labels: np.ndarray,
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
        n_threads: int) -> None:
    \"""Initialize upper and lower bounds for each sample for dense input data.
    Given X, centers and the pairwise distances divided by 2.0 between the
    centers this calculates the upper bounds and lower bounds for each sample.
    The upper bound for each sample is set to the distance between the sample
    and the closest center.
    The lower bound for each sample is a one-dimensional array of n_clusters.
    For each sample i assume that the previously assigned cluster is c1 and the
    previous closest distance is dist, for a new cluster c2, the
    lower_bound[i][c2] is set to distance between the sample and this new
    cluster, if and only if dist > center_half_distances[c1][c2]. This prevents
    computation of unnecessary distances for each sample to the clusters that
    it is unlikely to be assigned to.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The input data.
    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers.
    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        The half of the distance between any 2 clusters centers.
    labels : ndarray of shape(n_samples), dtype=int
        The label for each sample. This array is modified in place.
    upper_bounds : ndarray of shape(n_samples,), dtype=floating
        The upper bound on the distance between each sample and its closest
        cluster center. This array is modified in place.
    lower_bounds : ndarray, of shape(n_samples, n_clusters), dtype=floating
        The lower bound on the distance between each sample and each cluster
        center. This array is modified in place.
    n_threads : int
        The number of threads to be used by openmp.
    \"""
    ...

def init_bounds_sparse(
        X: spmatrix,
        centers: np.ndarray,
        center_half_distances: np.ndarray,
        labels: np.ndarray,
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
        n_threads: int) -> None:
    \"""Initialize upper and lower bounds for each sample for sparse input data.
    Given X, centers and the pairwise distances divided by 2.0 between the
    centers this calculates the upper bounds and lower bounds for each sample.
    The upper bound for each sample is set to the distance between the sample
    and the closest center.
    The lower bound for each sample is a one-dimensional array of n_clusters.
    For each sample i assume that the previously assigned cluster is c1 and the
    previous closest distance is dist, for a new cluster c2, the
    lower_bound[i][c2] is set to distance between the sample and this new
    cluster, if and only if dist > center_half_distances[c1][c2]. This prevents
    computation of unnecessary distances for each sample to the clusters that
    it is unlikely to be assigned to.
    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The input data. Must be in CSR format.
    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers.
    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        The half of the distance between any 2 clusters centers.
    labels : ndarray of shape(n_samples), dtype=int
        The label for each sample. This array is modified in place.
    upper_bounds : ndarray of shape(n_samples,), dtype=floating
        The upper bound on the distance between each sample and its closest
        cluster center. This array is modified in place.
    lower_bounds : ndarray of shape(n_samples, n_clusters), dtype=floating
        The lower bound on the distance between each sample and each cluster
        center. This array is modified in place.
    n_threads : int
        The number of threads to be used by openmp.
    \"""
    ...


def elkan_iter_chunked_dense(
        X: np.ndarray,
        sample_weight: np.ndarray,
        centers_old: np.ndarray,
        centers_new: np.ndarray,
        weight_in_clusters: np.ndarray,
        center_half_distances: np.ndarray,
        distance_next_center: np.ndarray,
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
        labels: np.ndarray,
        center_shift: np.ndarray,
        n_threads: int,
        update_centers: bool=True) -> None:
    \"""Single iteration of K-means Elkan algorithm with dense input.
    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.
    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.
    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.
    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.
    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.
    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        Half pairwise distances between centers.
    distance_next_center : ndarray of shape (n_clusters,), dtype=floating
        Distance between each center its closest center.
    upper_bounds : ndarray of shape (n_samples,), dtype=floating
        Upper bound for the distance between each sample and its center,
        updated inplace.
    lower_bounds : ndarray of shape (n_samples, n_clusters), dtype=floating
        Lower bound for the distance between each sample and each center,
        updated inplace.
    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.
    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.
    n_threads : int
        The number of threads to be used by openmp.
    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    \"""
    ...


def elkan_iter_chunked_sparse(
        X: spmatrix,
        sample_weight: np.ndarray,
        centers_old: np.ndarray,
        centers_new: np.ndarray,
        weight_in_clusters: np.ndarray,
        center_half_distances: np.ndarray,
        distance_next_center: np.ndarray,
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
        labels: np.ndarray,
        center_shift: np.ndarray,
        n_threads: int,
        update_centers: bool=True) -> None:
    \"""Single iteration of K-means Elkan algorithm with sparse input.
    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.
    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        The observations to cluster. Must be in CSR format.
    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.
    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.
    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.
    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.
    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        Half pairwise distances between centers.
    distance_next_center : ndarray of shape (n_clusters,), dtype=floating
        Distance between each center its closest center.
    upper_bounds : ndarray of shape (n_samples,), dtype=floating
        Upper bound for the distance between each sample and its center,
        updated inplace.
    lower_bounds : ndarray of shape (n_samples, n_clusters), dtype=floating
        Lower bound for the distance between each sample and each center,
        updated inplace.
    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.
    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.
    n_threads : int
        The number of threads to be used by openmp.
    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    \"""
    ...
""")
                
                
    with open("typings/sklearn/cluster/_k_means_lloyd.pyi", "w") as f:
        f.write("""
        # lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
""")
                
    if not os.path.exists('typings/sklearn/tests'):
        os.mkdir('typings/sklearn/tests')
    with open(f'typings/sklearn/tests/__init__.pyi', 'w') as f:
        f.write("")

    with open(f'typings/sklearn/tests/random_seed.pyi', 'w') as f:
        f.write("""
import pytest


# Passes the main worker's random seeds to workers
class XDistHooks:
    def pytest_configure_node(self, node) -> None: ...


def pytest_configure(config) -> None: ...
def pytest_report_header(config) -> list[str]|None: ...
""")
                
    with open(f'typings//sklearn/decomposition/_online_lda_fast.pyi', 'w') as f:
        f.write("""
import numpy as np

def mean_change(arr_1: np.ndarray, arr_2: np.ndarray) -> float:
    \"""Calculate the mean difference between two arrays.

    Equivalent to np.abs(arr_1 - arr2).mean().
    \"""
    ...

def psi(x: float) -> float: ...

""")

    with open(f'typings/sklearn/utils/arrayfuncs.pyi', 'w') as f:
        f.write("""
import numpy as np


def min_pos(X: np.ndarray) -> float:
    \"""Find the minimum value of an array over positive values
    Returns the maximum representable value of the input dtype if none of the
    values are positive.
    \"""
    ...



def cholesky_delete(L: np.ndarray, go_out: int) -> None: ...
""")