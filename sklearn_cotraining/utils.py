import numpy as np

from data_utils import biased_fn, fn_to_mat


def supports_proba(clf):
    """Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
    return hasattr(clf, 'predict_proba')

def compute_conditionals(prob_tensor: np.ndarray) -> np.ndarray:
    marginals = np.sum(prob_tensor, axis=2, keepdims=True)
    return np.divide(prob_tensor, marginals, where=marginals != 0)

# P(view | 1 - view)
def compute_conditionals_one_view(prob_tensor: np.ndarray, view: int) -> np.ndarray:
    assert view in [0, 1]
    p_union = np.sum(prob_tensor, axis=2, keepdims=True)
    p_f1 = np.sum(prob_tensor, axis=(view, 2), keepdims=True)
    conds = np.divide(p_union, p_f1, where=p_f1 != 0).squeeze(axis=2)
    return conds if view == 1 else np.swapaxes(conds, 0, 1)

def compute_posteriors(cond_tensor: np.ndarray, preds1: np.ndarray, preds2: np.ndarray):
    num_classes = cond_tensor.shape[0]
    assert cond_tensor.shape == (num_classes, num_classes, num_classes)
    assert preds1.shape == preds2.shape == (num_classes,)

    outer = np.outer(preds1, preds2)
    outer = np.repeat(outer[np.newaxis, :, :], num_classes, axis=0)
    outer = np.swapaxes(outer, 0, 2)

    return np.sum(cond_tensor * outer, axis=(0, 1))

def compute_posteriors_one_view(cond_tensor: np.ndarray, preds: np.ndarray):
    num_classes = cond_tensor.shape[0]
    assert cond_tensor.shape == (num_classes, num_classes)
    assert preds.shape == (num_classes,)

    return np.dot(cond_tensor, preds) 

if __name__ == '__main__':
    conds = compute_conditionals_one_view(np.array(
        [[[0.05, 0.06], [0.07, 0.08]], [[0.09, 0.1], [0.11, 0.44]]]
    ), view=0)
    preds1 = np.array([0.3, 0.7])
    post_1 = compute_posteriors_one_view(conds, preds1)
    preds2 = np.array([0.4, 0.6])
    print(compute_posteriors(conds, preds1, preds2))