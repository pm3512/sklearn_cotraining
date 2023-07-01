from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import shuffle as util_shuffle
from math import ceil
from enum import Enum

from sklearn.dummy import check_random_state

class DataGenerationType(Enum):
    """
        Enum for the different types of data generation
    """
    SKLEARN = 1
    RECTS = 2

def rearrange_cols(X: np.ndarray, n_informative: int, n_redundant: int):
    n_features = X.shape[1]
    X1 =  np.concatenate([
        X[:, :(n_informative // 2)],
        X[:, (n_informative): n_informative + n_redundant // 2],
        X[:, (n_informative + n_redundant):((n_features + n_informative + n_redundant) // 2)]
    ], axis=1)

    X2 =  np.concatenate([
        X[:, (n_informative // 2):(n_informative)],
        X[:, n_informative + n_redundant // 2: n_informative + n_redundant],
        X[:, ((n_features + n_informative + n_redundant) // 2):]
    ], axis=1)

    return (X1, X2)


def process_data(
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        n_features: int,
        n_informative: int,
        prob_replace_background: float=0,
        prob_invert_class: float=0,
        random_state: int | None=None,
        permute_cols: bool=True
    ):
    if prob_replace_background + prob_invert_class > 1:
        raise ValueError("prob_replace_background + prob_invert_class must be less than 1")
    np.random.seed(random_state)

    # Rearrange features so that both classifiers have an equal number of
    # informative and redundant features. This eliminates the possibility of
    # one classifier being useless, which would lead to high disagreement
    if permute_cols:
        X1, X2 = rearrange_cols(X, n_informative, n_informative)
    else:
        X1 = X[:, :n_features // 2]
        X2 = X[:, n_features // 2:]

    X = np.concatenate([X1, X2], axis=1)
    # shuffle
    permute = np.random.permutation(n_samples)
    X = X[permute]
    y = y[permute]

    # unlabel half of the data
    y[:n_samples//2] = -1

    return X, y

def gen_rects(
        n_samples: int,
        n_features: int,
        random_state: int | None=None,
        prob_replace: float=0,
    ):
    assert False
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    np.random.seed(random_state)
    X = np.random.uniform(size=(n_samples, n_features))
    # xor-like labelling function
    def choose_label(x, disagree_size):
        if (x[0] >= 1 - disagree_size and x[1] < disagree_size):
            return 0
        if (x[0] < disagree_size and x[1] >= 1 - disagree_size):
            return 1
        return 1 if x[0] + x[1] >= 1 else 0
    label_fn = lambda x: choose_label(x, prob_replace)
    y = np.apply_along_axis(label_fn, 1, X)
    return X, y

def generate_data(
        n_samples: int,
        n_features: int,
        n_informative: int,
        prob_replace_background: float=0,
        prob_invert_class: float=0,
        gen_type: DataGenerationType=DataGenerationType.SKLEARN,
        random_state: int | None=None,
        permute_cols: bool=True
    ):
    np.random.seed(random_state)
    if gen_type == DataGenerationType.SKLEARN:
        X, y = make_classification(
            # should be enough
            n_samples=n_samples,
            n_classes=3,
            n_features=n_features,
            random_state=random_state,
            n_informative=n_informative,
            n_redundant=n_informative,
            shuffle=False,
        )
    elif gen_type == DataGenerationType.RECTS:
        X, y = gen_rects(n_samples * 5, n_features, random_state=random_state, prob_replace=prob_replace)
        prob_replace = 0
    else:
        raise ValueError(f"Unknown data generation type {gen_type}")
    perm = np.random.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    # separate out background class
    return process_data(
        X,
        y,
        n_samples,
        n_features,
        n_informative,
        prob_replace_background,
        prob_invert_class,
        random_state,
        permute_cols
    )

def generate_from_probmatrix(
    mat: np.ndarray,
    n_samples: int,
    n_features: int,
    n_informative: int,
    random_state: int | None=None,
):
    n_classes = mat.shape[0]
    if len(mat.shape) < 3 or mat.shape[1] != n_classes or mat.shape[2] != n_classes:
        raise ValueError("Invalid tensor shape")
    eps = 0.0001
    if abs(mat.sum() - 1) > eps or (mat < 0).any():
        raise ValueError("Not a valid probability tensor")

    np.random.seed(random_state)
    X, y = make_classification(
        prob_tensor=mat,
        n_samples=n_samples,
        n_classes=n_classes,
        n_features=n_features,
        random_state=random_state,
        n_informative=n_informative,
        n_redundant=n_informative,
        shuffle=False,
    )
    X1, X2 = rearrange_cols(X, n_informative, n_informative)
    perm = np.random.permutation(n_samples)
    X1 = X1[perm]
    X2 = X2[perm]
    y = y[perm]

    X = np.concatenate([X1, X2], axis=1)

    y[:n_samples//2] = -1
    return X, y

def fn_to_mat(f: Callable[[int, int, int, int], float], n_classes: int):
    mat = np.zeros((n_classes, n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            for k in range(n_classes):
                mat[i, j, k] = f(i, j, k, n_classes)
                print(i, j, k, '   ', mat[i, j, k])
    #mat /= mat.sum()
    return mat

def identity_fn(i: int, j: int, k: int, n_classes: int):
    return 1 / n_classes if i == j == k else 0

def set_prob_replace_fn(i: int, j: int, k: int, n_classes, non_diagonal_ratio: float):
    non_diagonal = n_classes ** 2 - n_classes
    if i == j == k:
        return (1 - non_diagonal_ratio) / n_classes
    if i != j and (i == k or j == k):
        return non_diagonal_ratio / non_diagonal / 2
    return 0

def make_classification(
    prob_tensor: np.ndarray=fn_to_mat(identity_fn, 2),
    n_samples=100,
    n_features=20,
    *,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None,
):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, default=2
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, default=2
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, default=0
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, default=2
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, default=2
        The number of clusters per class.

    weights : array-like of shape (n_classes,) or (n_classes - 1,),\
              default=None
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1. Note that the actual class proportions will
        not exactly match ``weights`` when ``flip_y`` isn't 0.

    flip_y : float, default=0.01
        The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder. Note that the default setting flip_y > 0 might lead
        to less than ``n_classes`` in y in some cases.

    class_sep : float, default=1.0
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : bool, default=True
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, ndarray of shape (n_features,) or None, default=0.0
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, ndarray of shape (n_features,) or None, default=1.0
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for class membership of each sample.

    See Also
    --------
    make_blobs : Simplified variant.
    make_multilabel_classification : Unrelated generator for multilabel tasks.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.
    """
    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            "Number of informative, redundant and repeated "
            "features must sum to less than the number of total"
            " features"
        )
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(
            msg.format(
                n_classes, n_clusters_per_class, n_informative, 2**n_informative
            )
        )

    if weights is not None:
        if len(weights) not in [n_classes, n_classes - 1]:
            raise ValueError(
                "Weights specified but incompatible with number of classes."
            )
        if len(weights) == n_classes - 1:
            if isinstance(weights, list):
                weights = weights + [1.0 - sum(weights)]
            else:
                weights = np.resize(weights, n_classes)
                weights[-1] = 1.0 - sum(weights[:-1])
    else:
        weights = [1.0 / n_classes] * n_classes

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    # Distribute samples among clusters by weight
    n_samples_per_cluster = [
        int(n_samples * weights[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative, generator).astype(
        float, copy=False
    )
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.uniform(size=(n_clusters, 1))
        centroids *= generator.uniform(size=(1, n_informative))

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.standard_normal(size=(n_samples, n_informative))

    # Create each cluster; a variant of make_blobs
    mat_flat = prob_tensor.flatten()
    choice_indices = [i for i in range(len(mat_flat))]
    for i in range(n_samples):
        # choose a class
        class_idx = np.random.choice(choice_indices, p=mat_flat)
        class_idx = np.unravel_index(class_idx, prob_tensor.shape)
        assert len(class_idx) == 3
        X1_class, X2_class, y_class = class_idx
        X_k = X[i:i+1, :n_informative]  # slice a view of the cluster

        A = 2 * generator.uniform(size=(n_informative, n_informative)) - 1
        X_k[...] = np.dot(X_k, A)  # introduce random covariance
        X_k[:, :n_informative // 2] += centroids[X1_class, :n_informative // 2]
        X_k[:, n_informative // 2: n_informative] += centroids[X2_class, n_informative // 2:n_informative]
        y[i] = y_class

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.uniform(size=(n_informative, n_redundant)) - 1
        X[:, n_informative : n_informative + n_redundant] = np.dot(
            X[:, :n_informative], B
        )

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.uniform(size=n_repeated) + 0.5).astype(np.intp)
        X[:, n : n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        X[:, -n_useless:] = generator.standard_normal(size=(n_samples, n_useless))

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.uniform(size=n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.uniform(size=n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.uniform(size=n_features)
    X *= scale

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Randomly permute features
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]

    return X, y

def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions."""
    if dimensions > 30:
        return np.hstack(
            [
                rng.randint(2, size=(samples, dimensions - 30)),
                _generate_hypercube(samples, 30, rng),
            ]
        )
    out = sample_without_replacement(2**dimensions, samples, random_state=rng).astype(
        dtype=">u4", copy=False
    )
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out

if __name__ == '__main__':
    X, y = generate_from_probmatrix(
        fn_to_mat(identity_fn, 2),
        100,
        10,
        2,
        12344,
    )
    X = X[y != -1]
    y = y[y != -1]
    plt.scatter(X[:, 0], X[:, 5], c=y)
    plt.show()

    X, y = generate_data(
        100,
        10,
        2,
        random_state=12344
    )
    X = X[y != -1]
    y = y[y != -1]
    plt.scatter(X[:, 0], X[:, 5], c=y)
    plt.show()
