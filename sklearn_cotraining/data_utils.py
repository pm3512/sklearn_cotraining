import numpy as np
from sklearn.datasets import make_classification
from math import ceil
from enum import Enum

class DataGenerationType(Enum):
    """
        Enum for the different types of data generation
    """
    SKLEARN = 1
    RECTS = 2

def process_data(
        X: np.ndarray,
        y: np.ndarray,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        X_background: np.ndarray,
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
    X = np.concatenate([X, X_pool], axis=0)

    if permute_cols:
        X1 =  np.concatenate([
            X[:, :(n_informative // 2)],
            X[:, (n_informative):3 * (n_informative // 2)],
            X[:, (n_informative * 2):(n_features // 2 + n_informative)]
        ], axis=1)

        X2 =  np.concatenate([
            X[:, (n_informative // 2):(n_informative)],
            X[:, 3 * (n_informative // 2): (n_informative * 2)],
            X[:, (n_features // 2 + n_informative):]
        ], axis=1)
    else:
        X1 = X[:, :n_features // 2]
        X2 = X[:, n_features // 2:]

    X = np.concatenate([X1, X2], axis=1)
    X_pool = X[n_samples:]
    X1 = X1[:n_samples]
    X2 = X2[:n_samples]

    # randomly select indices where the first or second half of features will
    # be replaced by features from random samples
    r = np.random.uniform(size=n_samples)
    invert_idx_1 = r < prob_invert_class / 2
    invert_idx_2 = (r >= prob_invert_class / 2) & (r < prob_invert_class)
    replace_idx_1 = (r >= prob_invert_class) & (r < prob_invert_class + prob_replace_background / 2)
    replace_idx_2 = (r >= prob_invert_class + prob_replace_background / 2) & (r < prob_invert_class + prob_replace_background)

    invert_idx_1 = np.tile(invert_idx_1, (n_features // 2, 1)).T
    invert_idx_2 = np.tile(invert_idx_2, (n_features // 2, 1)).T
    replace_idx_1 = np.tile(replace_idx_1, (n_features // 2, 1)).T
    replace_idx_2 = np.tile(replace_idx_2, (n_features // 2, 1)).T

    # randomly select samples to invert class
    negative_pool = X_pool[y_pool == 0]
    positive_pool = X_pool[y_pool == 1]

    negative_idx = y == 0
    positive_idx = y == 1

    negative_idx = np.tile(negative_idx, (n_features, 1)).T
    positive_idx = np.tile(positive_idx, (n_features, 1)).T

    pos_invert_idx = np.random.permutation(positive_pool.shape[0]).repeat(ceil(n_samples / positive_pool.shape[0]))[:n_samples]
    neg_invert_idx = np.random.permutation(negative_pool.shape[0]).repeat(ceil(n_samples / negative_pool.shape[0]))[:n_samples]
    replace_idx = np.random.permutation(X_background.shape[0]).repeat(ceil(n_samples / X_background.shape[0]))[:n_samples]

    pos_invert = positive_pool[pos_invert_idx]
    neg_invert = negative_pool[neg_invert_idx]
    replace = X_background[replace_idx]
    invert = positive_idx * neg_invert + negative_idx * pos_invert

    # invert class or replace with background
    X1_invert = invert[:, :(n_features // 2)]
    X2_invert = invert[:, (n_features // 2):]
    X1_replace = replace[:, :(n_features // 2)]
    X2_replace = replace[:, (n_features // 2):]

    X1 = np.where(invert_idx_1, X1_invert, X1)
    X1 = np.where(replace_idx_1, X1_replace, X1)
    X2 = np.where(invert_idx_2, X2_invert, X2)
    X2 = np.where(replace_idx_2, X2_replace, X2)

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
            n_samples=n_samples * 5,
            n_classes=3,
            n_features=n_features,
            random_state=random_state,
            n_informative=n_informative,
            n_redundant=n_informative,
            shuffle=False,
        )
    elif gen_type == DataGenerationType.RECTS:
        X, y = gen_rects(n_samples * 3, n_features, random_state=random_state, prob_replace=prob_replace)
        prob_replace = 0
    else:
        raise ValueError(f"Unknown data generation type {gen_type}")
    perm = np.random.permutation(n_samples * 3)
    X = X[perm]
    y = y[perm]

    # separate out background class
    X_background = X[y == 2]
    X = X[y != 2]
    y = y[y != 2]

    X_pool = X[n_samples:]
    y_pool = y[n_samples:]
    X = X[:n_samples]
    y = y[:n_samples]

    return process_data(
        X,
        y,
        X_pool,
        y_pool,
        X_background,
        n_samples,
        n_features,
        n_informative,
        prob_replace_background,
        prob_invert_class,
        random_state,
        permute_cols
    )