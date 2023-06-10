import numpy as np
from sklearn.datasets import make_classification

def process_data(
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        n_features: int,
        n_informative: int,
        prob_replace: float=0,
        random_state: int | None=None
    ):
    np.random.seed(random_state)

    # Rearrange features so that both classifiers have an equal number of
    # informative and redundant features. This eliminates the possibility of
    # one classifier being useless, which would lead to high disagreement
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

    # randomly select indices where the first or second half of features will
    # be replaced by features from random samples
    r = np.random.uniform(size=n_samples)
    replace_idx_1 = r < prob_replace / 2
    replace_idx_2 = (r >= prob_replace / 2) & (r < prob_replace)

    replace_idx_1 = np.tile(replace_idx_1, (n_features // 2, 1)).T
    replace_idx_2 = np.tile(replace_idx_2, (n_features // 2, 1)).T

    # randomly select samples to replace features
    negative_idx = y == 0
    negative_pool = X[negative_idx]
    positive_idx = y == 1
    positive_pool = X[positive_idx]

    negative_idx = np.tile(negative_idx, (n_features, 1)).T
    positive_idx = np.tile(positive_idx, (n_features, 1)).T

    pos_replace_idx = np.random.randint(0, positive_pool.shape[0], size=n_samples)
    neg_replace_idx = np.random.randint(0, negative_pool.shape[0], size=n_samples)

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
    '''
    permute = np.random.permutation(n_samples)
    X = X[permute]
    y = y[permute]
    '''

    # unlabel half of the data
    y[:n_samples//2] = -1

    return X, y

def generate_data(
        n_samples: int,
        n_features: int,
        n_informative: int,
        prob_replace: float=0,
        random_state: int | None=None
    ):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
        n_informative=n_informative,
        n_redundant=n_informative,
        shuffle=False,
    )
    return process_data(X, y, n_samples, n_features, n_informative, prob_replace, random_state)