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
        n_samples: int,
        n_features: int,
        n_informative: int,
        prob_replace: float=0,
        random_state: int | None=None,
        permute_cols: bool=True
    ):
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
    replace_idx_1 = r < prob_replace / 2
    replace_idx_2 = (r >= prob_replace / 2) & (r < prob_replace)

    replace_idx_1 = np.tile(replace_idx_1, (n_features // 2, 1)).T
    replace_idx_2 = np.tile(replace_idx_2, (n_features // 2, 1)).T

    # randomly select samples to replace features
    negative_pool = X_pool[y_pool == 0]
    positive_pool = X_pool[y_pool == 1]

    negative_idx = y == 0
    positive_idx = y == 1

    negative_idx = np.tile(negative_idx, (n_features, 1)).T
    positive_idx = np.tile(positive_idx, (n_features, 1)).T

    pos_replace_idx = np.random.permutation(positive_pool.shape[0]).repeat(ceil(n_samples / positive_pool.shape[0]))[:n_samples]
    neg_replace_idx = np.random.permutation(negative_pool.shape[0]).repeat(ceil(n_samples / negative_pool.shape[0]))[:n_samples]

    pos_replace = positive_pool[pos_replace_idx]
    neg_replace = negative_pool[neg_replace_idx]
    replace = positive_idx * neg_replace + negative_idx * pos_replace

    # replace features
    X1_replace = replace[:, :(n_features // 2)]
    X2_replace = replace[:, (n_features // 2):]
    X1 = np.where(replace_idx_1, X1_replace, X1)
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
    ):
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    np.random.seed(random_state)
    X = np.random.uniform(size=(n_samples, n_features))
    # xor-like labelling function
    choose_label = lambda x: 1 if (x[0] < 0.5 and x[1] < 0.5) or (x[0] >= 0.5 and x[1] >= 0.5) else 0
    y = np.apply_along_axis(choose_label, 1, X)
    return X, y

def generate_data(
        n_samples: int,
        n_features: int,
        n_informative: int,
        prob_replace: float=0,
        gen_type: DataGenerationType=DataGenerationType.SKLEARN,
        random_state: int | None=None,
        permute_cols: bool=True
    ):
    np.random.seed(random_state)
    if gen_type == DataGenerationType.SKLEARN:
        X, y = make_classification(
            n_samples=n_samples * 3,
            n_features=n_features,
            random_state=random_state,
            n_informative=n_informative,
            n_redundant=n_informative,
            shuffle=False,
        )
    elif gen_type == DataGenerationType.RECTS:
        X, y = gen_rects(n_samples * 3, n_features, random_state=random_state)
    else:
        raise ValueError(f"Unknown data generation type {gen_type}")
    perm = np.random.permutation(n_samples * 3)
    X = X[perm]
    y = y[perm]

    X_pool = X[n_samples:]
    y_pool = y[n_samples:]
    X = X[:n_samples]
    y = y[:n_samples]

    return process_data(X, y, X_pool, y_pool, n_samples, n_features, n_informative, prob_replace, random_state, permute_cols)